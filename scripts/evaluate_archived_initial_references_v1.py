"""Recheck archived model references from current actual causal release states.

The old successes only nominate references. Current V15 feedback, actual initial
state and original physical checks decide the new model outcome. No native
teleport, deployment, learning, relaxed matcher or expert labels.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_state_settling_feedback_v1 import evaluate
from scripts.pure_control_stance_foot_reference_v15 import PureControlStanceFootReferenceV15
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.audit_actual_handoff_model_v1 import state_from_record
from scripts.retry_budgeted_settling_feedback_v1 import sha, write


def run_one(cfg, nominal, path, deadline):
    start = time.monotonic()
    result = evaluate(cfg, nominal, path, 'plan_end', min(deadline, start + 1200))
    result['worker_wall_seconds'] = time.monotonic() - start
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['configuration', 'archive-library', 'archive-feedback', 'run', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--cases', nargs='+', type=int, required=True)
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args()
    read = lambda path: json.loads(Path(path).read_text())
    if a.output.exists() or len(set(a.cases)) != len(a.cases):
        raise ValueError('Fresh output and distinct cases required')
    if read(a.run / 'job.json')['status'] != 'SUCCEEDED':
        raise ValueError('Completed native history required')
    cfg = read(a.configuration)
    original = read(cfg['model_reference_library'])
    archive = read(a.archive_library)
    old_results = read(a.archive_feedback / 'result.json')['results']
    pins = {}
    for document in [archive, read(a.archive_feedback / 'inputs.json')]:
        for path, digest in document['source_sha256'].items():
            if sha(path) != digest:
                raise ValueError('Changed archive input ' + path)
            pins[path] = digest
    nominal = read(a.run / 'native_command_inputs.json')['default_joint_position_rad']
    controller = PureControlStanceFootReferenceV15(PositionPDInterface(**cfg['interface']), nominal,
                                                  cfg['model_reference_library'])
    with np.load(a.run / 'pure_control_commands.npz') as z:
        arrays = {key: z[key].copy() for key in z.files}
    diagnostics = [json.loads(line) for line in (a.run / 'pure_control_diagnostics.jsonl').read_text().splitlines()]
    if len(diagnostics) != len(arrays['lane']):
        raise ValueError('Misaligned actual-state evidence')
    first = {}
    for case in sorted(set(arrays['lane'].tolist())):
        i = int(np.flatnonzero(arrays['lane'] == case)[0])
        row = diagnostics[i]
        if (row['lane'], row['tick']) != (case, int(arrays['tick'][i])) or not row['phase']['owns_final_recovery_command']:
            raise ValueError('Expected causal recovery ownership')
        record = {key: arrays[key][i].tolist() for key in ['q', 'v', 'previous', 'ground_force']}
        record.update(case=case, tick=int(arrays['tick'][i]), time_s=float(arrays['time_s'][i]), phase=row['phase'])
        first[case] = record
    selected = []
    for case in a.cases:
        evidence = [r for r in old_results if r['source_case'] == case]
        records = [r for r in archive['references'] if r['source_case'] == case]
        if len(evidence) != 1 or len(records) != 1 or case not in first:
            raise ValueError('Missing unique candidate evidence')
        r = evidence[0]
        if r['failure'] is not None or r['steps'] != r['required_steps'] or r['final_model_stable_s'] < .8:
            raise ValueError('Archive did not complete stable model execution')
        if any(old['source_key'] == records[0]['source_key'] for old in original['references']):
            raise ValueError('Candidate already in the current bank')
        # Re-anchor the candidate at the current measured release. Its entire
        # commanded trajectory stays unchanged; new feedback must validate it.
        # This matches evaluate()'s query and avoids evaluating one query then
        # later deploying the slightly different historical query.
        candidate = copy.deepcopy(records[0])
        candidate['archived_query'] = candidate['query']
        candidate['parent_source_key'] = candidate['source_key']
        record = first[case]
        candidate['query'] = dict(q=record['q'], v=record['v'], target=record['previous'],
                                  contacts=record['phase']['ground_contacts'])
        candidate['source_key'] = hashlib.sha256(json.dumps(dict(
            parent=candidate['parent_source_key'], query=candidate['query']), sort_keys=True).encode()).hexdigest()
        candidate['current_actual_query_reanchored'] = True
        selected.append(candidate)
    combined = copy.deepcopy(original)
    combined['references'].extend(copy.deepcopy(selected))
    # Instantiate the unmodified matcher without publishing a deployable bank.
    matcher = copy.copy(controller.library)
    matcher.records = combined['references']
    changes = []
    old_count = 0
    snapshots = []
    for case, record in first.items():
        state = state_from_record(controller.model, record)
        old, _ = controller.library.match(state)
        new, score = matcher.match(state)
        if old is not None:
            old_count += 1
            if new is None or old['source_key'] != new['source_key']:
                raise ValueError('Archive changes an old selection')
        if new is not None and old is None:
            if case not in a.cases or new['source_case'] != case:
                raise ValueError('Unrequested new match must be evaluated separately')
            key = str(case) + ':' + str(record['tick'])
            errors = {name: float(np.max(abs(np.asarray(record[name]) - new['archived_query'][field])))
                      for name, field in [('q', 'q'), ('v', 'v'), ('previous', 'target')]}
            changes.append(dict(case=case, actual_tick=record['tick'], match_score=score,
                                source_key=new['source_key'], initial_difference_from_archive=errors))
            snapshots.append(dict(state_key=key, initial_actual_state=record,
                fine=dict(trajectory=new['trajectory'], substeps=160),
                archived_source_key=new['parent_source_key'], current_reference_key=new['source_key'],
                current_actual_query_reanchored=True, native_success=False, valid_expert_labels=False))
    if {r['case'] for r in changes} != set(a.cases):
        raise ValueError('Every nominated reference must match a current uncovered release')
    summary = dict(causal_releases=len(first), original_matches=old_count,
        proposed_matches=old_count + len(changes), old_selections_unchanged=True,
        new_matches=changes, native_execution=False, training=False, valid_expert_labels=False)
    if a.check_only:
        print(json.dumps(summary))
        return
    a.output.mkdir(parents=True)
    snapdir = a.output / 'candidate_snapshots'
    snapdir.mkdir()
    for snapshot in snapshots:
        write(snapdir / ('state_' + snapshot['state_key'].replace(':', '_') + '.json'), snapshot)
    write(a.output / 'coverage.json', summary)
    combined['provisional_model_diagnostic_only'] = True
    combined['full_episode_validation_required'] = True
    write(a.output / 'provisional_reference_library.json', combined)
    files = [Path(__file__), a.configuration, a.archive_library, Path(cfg['model_reference_library']),
        a.archive_feedback / 'inputs.json', a.archive_feedback / 'result.json']
    files += [a.run / name for name in ['job.json', 'native_command_inputs.json', 'pure_control_commands.npz', 'pure_control_diagnostics.jsonl']]
    files += list(snapdir.glob('*.json'))
    files += [a.output / 'provisional_reference_library.json']
    files += [Path(m.__file__) for name, m in list(sys.modules.items())
              if name.startswith('scripts.') and getattr(m, '__file__', None)]
    pins.update({str(path.resolve()): sha(path) for path in files})
    write(a.output / 'inputs.json', dict(source_sha256=pins, workers=4, worker_seconds=1200,
        total_seconds=2400, physical_substeps=160, scoring_substeps=16,
        initial_states='current recorded causal releases, not archived queries',
        controller='V15', native_success=False, training=False, acceptance_changed=False))
    start = time.monotonic()
    results = []
    def progress(state):
        write(a.output / 'status.json', dict(state=state, pid=os.getpid(), completed=len(results),
            total=len(snapshots), results=results, elapsed_s=time.monotonic() - start,
            native_success=False, valid_expert_labels=False))
    progress('RUNNING_ARCHIVED_REFERENCE_RECHECK')
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context('spawn')) as pool:
        jobs = [pool.submit(run_one, cfg, nominal, str(path), start + 2400)
                for path in sorted(snapdir.glob('*.json'))]
        for job in as_completed(jobs):
            result = job.result()
            write(a.output / ('state_' + result['state_key'].replace(':', '_') + '.json'), result)
            results.append({k: v for k, v in result.items() if k not in ['trajectory', 'initial_actual_state']})
            progress('RUNNING_ARCHIVED_REFERENCE_RECHECK')
    for path, digest in pins.items():
        if sha(path) != digest:
            raise ValueError('Changed recheck input ' + path)
    write(a.output / 'result.json', dict(state='COMPLETED_ARCHIVED_REFERENCE_RECHECK', results=results,
        coverage=summary, elapsed_s=time.monotonic() - start, native_success=False, valid_expert_labels=False))
    progress('COMPLETED_ARCHIVED_REFERENCE_RECHECK')


if __name__ == '__main__':
    main()
