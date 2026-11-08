"""Export a complete independently replayed foot-space policy for native audit."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_native_early_support_v1 import prepare
from scripts.check_constrained_recovery_suffix_v1 import source_context,state_from_row
from scripts.search_preserved_early_support_v1 import initialize
from scripts.recovery_family_queries_v1 import transfer_reference
from scripts.audit_joint_sequence_proposal_v1 import plain
from quadsmash.rl.episode_trace_v4 import unpack_public


def export(plan_path,search_path,base_bundle,bundle,output):
    read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if bundle.exists() or output.exists():raise ValueError('Fresh preparation and bundle required')
    plan=read(plan_path);search=read(search_path);verified=search.get('verified');case=search['case']
    if not verified or case!=87 or search['native_execution'] or search['valid_expert_labels']:
        raise ValueError('Verified model experiment required')
    x=verified['winner'];y=verified['replay']
    for row in [x,y]:
        r=row['result']
        if (not r['accepted'] or r['failure'] is not None or r['terminal_violations'] or
                r['substeps']!=160 or r['completed_model_steps']!=520 or len(r['trajectory'])!=520 or
                r['control_tick']!=280 or r['original_episode_end_tick']!=800 or r['trailing_model_window_s']<.8):
            raise ValueError('Complete constrained replay required')
        for k,v in [('initial_q','q'),('initial_v','v'),('initial_contacts','contacts'),('initial_previous_command','target')]:
            if not np.array_equal(r[k],search['query'][v]):raise ValueError('Query replaced')
    if not np.array_equal(x['transition_knots'],y['transition_knots']) or any(
            not np.array_equal(a[k],b[k]) for a,b in zip(x['result']['trajectory'],y['result']['trajectory'])
            for k in ['q','v','target','contacts']):raise ValueError('Independent replay differs')
    pins=dict(search['source_sha256'])
    for path,h in pins.items():
        if sha(path)!=h:raise ValueError('Changed source '+path)
    source=read(plan['source']);_,_,_,_,context=source_context(source['configuration']['source'])
    initialize(*context);context,_=transfer_reference(context,state_from_row(search['query']))
    old=read(base_bundle/'manifest.json')
    if old['configuration']!=context[0]:raise ValueError('Native interface/configuration changed')
    trace_path=Path(old['baseline'])/f'fixed_policy_traces/case_{case:03d}.npz'
    trace_manifest=read(trace_path.parent/'manifest.json')
    entry=next(r for r in trace_manifest['episodes'] if r['source_training_index']==case)
    if sha(trace_path)!=entry['data_sha256']:raise ValueError('Baseline trace changed')
    with np.load(trace_path,allow_pickle=False) as z:
        public=unpack_public(z['packed_public'])[280].copy()
        previous=z['dispatched_actions'][279].copy();student=z['dispatched_actions'][280].copy()
    limits=np.asarray(context[0]['interface']['limits']);decoded=limits[:,0]+.5*(previous+1)*np.diff(limits).ravel()
    if np.max(abs(decoded-search['query']['target']))>1e-6 or abs(public[775]-2.8)>1e-5:
        raise ValueError('Actual command history or reference time differs')
    rows=y['result']['trajectory'];key=f'case_{case:03d}_'
    arrays={key+'commands_rad':np.asarray([r['target'] for r in rows]),
        key+'predicted_q':np.asarray([search['query']['q']]+[r['q'] for r in rows]),
        key+'predicted_v':np.asarray([search['query']['v']]+[r['v'] for r in rows]),
        key+'predicted_contacts':np.asarray([search['query']['contacts']]+[r['contacts'] for r in rows],bool),
        key+'query_public':public,key+'previous_action':previous,key+'query_student_action':student,
        'tracking_reference_q':context[4],'tracking_reference_v':context[5]}
    for path in [plan_path,search_path,base_bundle/'manifest.json',trace_path,trace_path.parent/'manifest.json',
        Path(__file__),ROOT/'scripts/native_foot_space_recovery_v1.py',ROOT/'scripts/check_foot_space_dispatch_v1.py']:
        pins[str(path.resolve())]=sha(path)
    for name,module in list(sys.modules.items()):
        if name.startswith('scripts.') and getattr(module,'__file__',None):
            path=str(Path(module.__file__).resolve());pins[path]=sha(path)
    bundle.mkdir(parents=True);archive=bundle/'sequences.npz';np.savez_compressed(archive,**arrays)
    manifest=dict(scope=__doc__,baseline=old['baseline'],policy_sha256=old['policy_sha256'],
        configuration=context[0],control_dt_s=.01,rows=[dict(case=case,query_tick=280,horizon_steps=520)],
        bundle_name=archive.name,bundle_sha256=sha(archive),ground_geometry_sha256=old['ground_geometry_sha256'],
        source_sha256=pins,native_execution=False,valid_expert_labels=False,
        foot_space_definition=dict(contract='verified_foot_space_transition_v1',independent_replay_exact=True,
            original_knots=search['original_knots'],transition_knots=y['transition_knots'],
            braking=search['braking'],tracking_gains=context[6]),
        prediction_contract='Feedback computed on actual state, with complete offline prediction as comparison only.')
    (bundle/'manifest.json').write_text(json.dumps(plain(manifest),indent=2)+'\n')
    prepared=prepare(ROOT,output)
    entry=Path(prepared['entry']);text=entry.read_text()
    before='from scripts.native_early_support_tail_v1 import NativeEarlySupportTailV1 as NativeRecoverySequenceReplay'
    if text.count(before)!=1:raise ValueError('Changed entry template')
    text=text.replace(before,'from scripts.native_foot_space_recovery_v1 import NativeFootSpaceRecoveryV1 as NativeRecoverySequenceReplay')
    # Passive high-frequency monitoring includes the newly intervened lane.
    before='lanes=(3,125),first_frame=89600,last_frame=92000'
    if text.count(before)!=1:raise ValueError('Changed observer template')
    text=text.replace(before,'lanes=(87,125),first_frame=89600,last_frame=92000')
    ast.parse(text);entry.write_text(text)
    audit=Path(prepared['audit']);text=audit.read_text()
    before='from scripts.check_early_support_dispatch_v1 import verify_feedback_dispatch'
    if text.count(before)!=1:raise ValueError('Changed dispatch audit')
    text=text.replace(before,'from scripts.check_foot_space_dispatch_v1 import verify_feedback_dispatch')
    text=text.replace("ROOT/'scripts/check_early_support_dispatch_v1.py'","ROOT/'scripts/check_foot_space_dispatch_v1.py'")
    text=text.replace("ROOT/'scripts/native_early_support_tail_v1.py'","ROOT/'scripts/native_foot_space_recovery_v1.py'")
    ast.parse(text);audit.write_text(text)
    prepared.update(entry_sha256=sha(entry),audit_sha256=sha(audit),bundle=str(bundle),
        bundle_manifest_sha256=sha(bundle/'manifest.json'),case=case,model_independent_replay=True,
        model_search=str(search_path),native_execution=False,teacher_acceptance=False)
    prepared['source_sha256'].update(pins)
    prepared['source_sha256'][str(entry.resolve())]=sha(entry)
    prepared['source_sha256'][str(audit.resolve())]=sha(audit)
    (output/'preparation.json').write_text(json.dumps(plain(prepared),indent=2)+'\n')
    return prepared


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['plan','search','base-bundle','bundle','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();r=export(a.plan.resolve(),a.search.resolve(),a.base_bundle.resolve(),a.bundle.resolve(),a.output.resolve())
    print(json.dumps(dict(case=r['case'],bundle=r['bundle'],entry=r['entry'],native_execution=False)))
