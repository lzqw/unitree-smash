"""Add a partial reference with a complete, verified V23 model-feedback path.

This bank requires state-driven V23 support. The reference itself is not an
open-loop success, and model feedback is not native execution or expert data.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.retry_budgeted_settling_feedback_v1 import sha, write
from scripts.pure_control_state_settling_v23 import PureControlStateSettlingV23
from scripts.recovery_reference_library_v2 import RecoveryReferenceLibraryV2
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.audit_actual_handoff_model_v1 import state_from_record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['parent-library', 'feedback', 'reference', 'run', 'configuration', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--state-key', required=True)
    a = p.parse_args()
    read = lambda path: json.loads(path.read_text())
    if a.output.exists() or read(a.run / 'job.json')['status'] != 'SUCCEEDED':
        raise ValueError('Fresh preparation and completed native initial-state source required')
    stem = a.state_key.replace(':', '_')
    feedback_path = a.feedback / (stem + '_16_state_settling.json')
    baseline_path = a.feedback / (stem + '_16_plan_end.json')
    row, baseline = read(feedback_path), read(baseline_path)
    parent, source, cfg = read(a.parent_library), read(a.reference), read(a.configuration)
    inputs = read(a.feedback / 'inputs.json')
    pins = {}
    for data in [parent, inputs]:
        for path, digest in data['source_sha256'].items():
            if sha(path) != digest or (path in pins and pins[path] != digest):
                raise ValueError('Changed/conflicting input ' + path)
            pins[path] = digest
    if pins.get(str(a.configuration.resolve())) != sha(a.configuration):
        raise ValueError('Different physical configuration')
    initial = row['initial_actual_state']
    if initial != baseline['initial_actual_state'] or initial != source['initial_actual_state']:
        raise ValueError('Different initial observations')
    if row['state_key'] != a.state_key or source['state_key'] != a.state_key:
        raise ValueError('Wrong reference')
    for item in [row, baseline]:
        if item['native_success'] or item['valid_expert_labels'] or item['candidate_scoring_substeps'] != 16 or item['physical_substeps'] != 160:
            raise ValueError('Require declared model-only controller fidelity')
    if (row['mode'] != 'state_settling' or baseline['mode'] != 'plan_end' or row['failure'] is not None
            or not row['full_model_path'] or row['steps'] != 800-initial['tick']
            or row['final_model_stable_s'] < .8 or row['state_settling_capture_tick'] is None):
        raise ValueError('Incomplete state-driven feedback')
    if len(row['trajectory']) != row['steps'] or any(t['tick'] != initial['tick']+i for i,t in enumerate(row['trajectory'])):
        raise ValueError('Incomplete feedback trace')
    capture = row['state_settling_capture_tick'] - initial['tick']
    if capture <= 0 or len(baseline['trajectory']) <= capture:
        raise ValueError('Missing pre-transition comparison')
    differences = {field: [float(np.max(abs(np.asarray(left[field])-right[field])))
        for left,right in zip(row['trajectory'], baseline['trajectory'])] for field in ['q','v','target']}
    if any(max(values[:capture], default=0.) != 0. for values in differences.values()):
        raise ValueError('State or command changed before declared transition')
    if differences['target'][capture] == 0.:
        raise ValueError('No command evidence for transition')
    snapshot = a.feedback / 'candidate_snapshots' / a.reference.name
    if sha(snapshot) != sha(a.reference) or inputs['source_sha256'].get(str(snapshot.resolve())) != sha(snapshot):
        raise ValueError('Feedback reference differs')
    status_raw = (a.feedback / 'status.json').read_bytes()
    status = json.loads(status_raw)
    for expected in [row, baseline]:
        found=[r for r in status['results'] if r['mode']==expected['mode'] and r['candidate_scoring_substeps']==16]
        if len(found)!=1 or any(found[0][k]!=expected[k] for k in ['steps','failure','final_model_stable_s']):
            raise ValueError('Completed per-case result not recorded by worker owner')
    with np.load(a.run/'pure_control_commands.npz') as z:
        arrays={k:z[k].copy() for k in z.files}
    ids=np.flatnonzero((arrays['lane']==initial['case']) & (arrays['tick']==initial['tick']))
    if len(ids)!=1 or any(not np.array_equal(arrays[k][ids[0]],initial[k]) for k in ['q','v','previous','ground_force']):
        raise ValueError('Missing exact actual initial state')
    a.output.mkdir(parents=True)
    snapshot_status=a.output/'observed_feedback_status.json';snapshot_status.write_bytes(status_raw)
    files=[Path(__file__),a.parent_library,a.reference,feedback_path,baseline_path,snapshot_status,
        a.feedback/'inputs.json',a.run/'job.json',a.run/'native_command_inputs.json',
        a.run/'pure_control_commands.npz',a.run/'pure_control_diagnostics.jsonl']
    files += [Path(m.__file__) for name,m in list(sys.modules.items()) if name.startswith('scripts.') and getattr(m,'__file__',None)]
    pins.update({str(f.resolve()):sha(f) for f in files})
    entry=dict(source_case=initial['case'],source_actual_tick=initial['tick'],source_key=sha(a.reference),
        query=dict(q=initial['q'],v=initial['v'],target=initial['previous'],contacts=initial['phase']['ground_contacts']),
        trajectory=source['fine']['trajectory'],source_search=str(a.reference.resolve()),
        source_feedback=str(feedback_path.resolve()),native_success=False,valid_expert_labels=False,
        reference_open_loop_complete=source['fine']['model_path_complete'],
        requires_controller='PureControlStateSettlingV23',model_feedback_stable_s=row['final_model_stable_s'])
    bank=dict(parent,references=[*parent['references'],entry],source_sha256=pins,
        requires_controller='PureControlStateSettlingV23',original_reference_count=len(parent['references']),
        added_reference_count=1,training=False,native_success=False,valid_expert_labels=False)
    library=a.output/'reference_library.json';write(library,bank)
    nominal=read(a.run/'native_command_inputs.json')['default_joint_position_rad']
    c=PureControlStateSettlingV23(PositionPDInterface(**cfg['interface']),nominal,a.parent_library)
    expanded=RecoveryReferenceLibraryV2(library,c.model)
    diagnostics=[json.loads(line) for line in (a.run/'pure_control_diagnostics.jsonl').read_text().splitlines()]
    changed=[];old_matches=0
    for case in sorted(set(arrays['lane'].tolist())):
        i=int(np.flatnonzero(arrays['lane']==case)[0]);diag=diagnostics[i]
        if (diag['lane'],diag['tick']) != (case,int(arrays['tick'][i])) or not diag['phase']['owns_final_recovery_command']:
            raise ValueError('Not a causal release observation')
        record={k:arrays[k][i] for k in ['q','v','previous','ground_force']};record['phase']=diag['phase']
        state=state_from_record(c.model,record);before,_=c.library.match(state);after,score=expanded.match(state)
        if before is not None:
            old_matches+=1
            if after is None or before['source_key']!=after['source_key']:raise ValueError('Changed old selection')
        elif after is not None:
            if case!=initial['case'] or int(arrays['tick'][i])!=initial['tick'] or after['source_key']!=entry['source_key']:
                raise ValueError('Unexpected additional initial-state match')
            test=PureControlStateSettlingV23(PositionPDInterface(**cfg['interface']),nominal,library)
            command,_=test.command(state,diag['phase']['phase'],float(arrays['time_s'][i]))
            error=float(np.max(abs(command-np.asarray(row['trajectory'][0]['target']))))
            if error>2e-6:raise ValueError('Entry command mismatch')
            changed.append(dict(case=case,tick=int(arrays['tick'][i]),score=score,first_command_error_rad=error))
    if len(changed)!=1:raise ValueError('Expected one new causal match')
    receipt=dict(passed=True,old_references_preserved=True,old_initial_matches_preserved=old_matches,
        causal_entries_checked=len(set(arrays['lane'].tolist())),new_matches=changed,
        capture_tick=row['state_settling_capture_tick'],exact_paired_prefix_steps=capture,
        first_command_difference_rad=differences['target'][capture],feedback_final_stable_s=row['final_model_stable_s'],
        feedback_batch_state_at_preparation=status['state'],reference_open_loop_complete=source['fine']['model_path_complete'],
        requires_controller='PureControlStateSettlingV23',native_execution=False,native_success=False,
        valid_expert_labels=False,old_native_success_preservation_not_inferred=True,
        source_sha256={**pins,str(library.resolve()):sha(library)})
    write(a.output/'receipt.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!='source_sha256'}))


if __name__=='__main__':main()
