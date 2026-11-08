"""Prepare a ground-start native test of a rejected joint/timing family proposal.

The candidate remains experimental. Ground-only controller contacts, real
command history, head forces and the original task scoring are retained.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.check_constrained_recovery_suffix_v1 import source_context,state_from_row
from scripts.search_preserved_early_support_v1 import initialize
from scripts.recovery_family_queries_v1 import transfer_reference
from scripts.audit_joint_sequence_proposal_v1 import plain
from scripts.native_timed_experimental_recovery_v1 import NativeTimedExperimentalRecoveryV1 as NativeExperimentalRecoveryV2
from quadsmash.rl.episode_trace_v4 import unpack_public


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['plan','search','reference-search','template-plan','output','native-output']:
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or args.native_output.exists():raise ValueError('Fresh evidence paths required')
    read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    setup=read(args.plan);search=read(args.search);reference=read(args.reference_search);template=read(args.template_plan)
    if search['verified'] or search['best']['result']['accepted'] or search['case'] not in (71,85,87,96):
        raise ValueError('Declared failed preselected-family model candidate required')
    if search['best']['plan']['contract'] != 'timed_foot_space_recovery_v1':
        raise ValueError('Joint timing proposal required')
    source=read(setup['source']);_,_,_,_,context=source_context(source['configuration']['source']);initialize(*context)
    context,_=transfer_reference(context,state_from_row(reference['query']))
    context,_=transfer_reference(context,state_from_row(search['query']));initialize(*context)
    command=copy.deepcopy(template['command'])
    old_bundle=Path(command[command.index('--recovery-sequence-bundle')+1]);old=read(old_bundle/'manifest.json')
    if context[0]!=old['configuration']:raise ValueError('Native interface changed')
    case=search['case'];baseline=Path(old['baseline']);trace_path=baseline/f'fixed_policy_traces/case_{case:03d}.npz'
    trace_manifest=read(trace_path.parent/'manifest.json')
    trace_row=next(r for r in trace_manifest['episodes'] if r['source_training_index']==case)
    if sha(trace_path)!=trace_row['data_sha256']:raise ValueError('Baseline trace changed')
    with np.load(trace_path,allow_pickle=False) as z:
        public=unpack_public(z['packed_public']);actions=z['dispatched_actions'].copy()
    limits=np.asarray(context[0]['interface']['limits'])
    previous=limits[:,0]+.5*(actions[279]+1)*np.diff(limits).ravel()
    if np.max(abs(previous-search['query']['target']))>1e-6:raise ValueError('Actual command history differs')
    pins=dict(search['source_sha256'])
    for p,h in pins.items():
        if sha(p)!=h:raise ValueError('Changed model evidence '+p)
    args.output.mkdir(parents=True);bundle=args.output/'bundle';bundle.mkdir()
    archive=bundle/'reference.npz'
    np.savez_compressed(archive,tracking_reference_q=context[4],tracking_reference_v=context[5],
        query_public=public[280],previous_action=actions[279])
    for p in [args.plan,args.search,args.reference_search,args.template_plan,trace_path,
            trace_path.parent/'manifest.json',Path(__file__).resolve(),ROOT/'scripts/native_timed_experimental_recovery_v1.py']:
        pins[str(p.resolve())]=sha(p)
    for name,module in list(sys.modules.items()):
        if name.startswith('scripts.') and getattr(module,'__file__',None):
            p=str(Path(module.__file__).resolve());digest=sha(p)
            if p in pins and pins[p]!=digest:raise ValueError('Source disagreement '+p)
            pins[p]=digest
    manifest=dict(contract='timed_model_rejected_native_diagnostic_v1',case=case,
        baseline=str(baseline),policy_sha256=old['policy_sha256'],configuration=context[0],
        query=search['query'],proposal=search['best']['plan'],model_rejection=search['best']['result']['failure'],
        model_completed_steps=search['best']['result']['completed_model_steps'],model_accepted=False,
        tracking_gains=context[6],bundle_name=archive.name,bundle_sha256=sha(archive),
        source_sha256=pins,valid_expert_labels=False,native_execution=False)
    (bundle/'manifest.json').write_text(json.dumps(plain(manifest),indent=2,allow_nan=False)+'\n')
    # Actual recorded query tests the interface without advancing a simulator.
    check_dir=args.output/'interface_check';check_dir.mkdir()
    adapter=NativeExperimentalRecoveryV2(bundle,case,torch.tensor(limits),old['policy_sha256'],check_dir)
    obs=torch.zeros((128,794));student=torch.zeros((128,18));finished=torch.zeros(128,dtype=torch.bool)
    for tick in range(280):
        student[case]=torch.tensor(actions[tick]);adapter.action(tick,obs,student,finished)
    obs[case]=torch.tensor(public[280]);student[case]=torch.tensor(actions[280])
    state=state_from_row(search['query']);state.measured_force_world=np.zeros((4,3))
    out=adapter.action(280,obs,student,finished,state)
    others=np.arange(128)!=case
    if not torch.equal(out[others],student[others]) or not adapter.query_check['passed']:
        raise ValueError('Recorded query interface check failed')
    interface_check=dict(query_match=adapter.query_check,others_unchanged=True,
        source='recorded native query only; no simulator execution',first_target_rad=adapter.previous.tolist())
    (args.output/'interface_check.json').write_text(json.dumps(interface_check,indent=2)+'\n')
    entry_old=next(Path(v) for v in command if v.endswith('evaluate_early_support.py'))
    text=entry_old.read_text()
    before='from scripts.native_foot_space_recovery_v1 import NativeFootSpaceRecoveryV1 as NativeRecoverySequenceReplay'
    if text.count(before)!=1:raise ValueError('Changed native entry template')
    text=text.replace(before,'from scripts.native_timed_experimental_recovery_v1 import NativeTimedExperimentalRecoveryV1 as NativeRecoverySequenceReplay')
    before='lanes=(87,125),first_frame=89600,last_frame=92000'
    if text.count(before)!=1:raise ValueError('Changed passive observer template')
    text=text.replace(before,f'lanes=({case},125),first_frame=89600,last_frame=92000')
    text=text.replace('measured_state_feedback=True,offline_optimized_reference=True,',
        'measured_state_feedback=True,offline_optimized_reference=True,model_accepted=False,experimental_proposal=True,')
    changes={
        '        sequence_student_commands=[];sequence_final_commands=[];sequence_first_episode=[]':
        '        from scripts.native_ground_support_state_v1 import NativeGroundSupportStateV1\n'
        '        ground_support=NativeGroundSupportStateV1(env,native_replay.case,args.output)\n'
        '        sequence_student_commands=[];sequence_final_commands=[];sequence_first_episode=[]',
        '                replay_state=None':
        '                ground_support.update(tick)\n                replay_state=None',
        '                    replay_state=native_replay.model.state_from_native(env,native_replay.case)':
        '                    replay_state=native_replay.model.state_from_native(env,native_replay.case)\n'
        '                    replay_state=ground_support.apply(replay_state,tick)',
        "    finally:\n        if 'head_observer' in locals():":
        "    finally:\n        if 'ground_support' in locals():\n            ground_support.save()\n"
        "        if 'head_observer' in locals():",
    }
    for before,after in changes.items():
        if text.count(before)!=1:raise ValueError('Changed evaluator boundary '+before)
        text=text.replace(before,after)
    ast.parse(text);entry=args.output/'evaluate_joint_timed_family.py';entry.write_text(text)
    command=[str(entry.resolve()) if v==str(entry_old) else v for v in command]
    for i,v in enumerate(command[:-1]):
        if v=='--output':command[i+1]=str(args.native_output.resolve())
        elif v=='--recovery-sequence-bundle':command[i+1]=str(bundle.resolve())
        elif v=='--recovery-sequence-case':command[i+1]=str(case)
    check_template=ROOT/'scripts/check_native_experimental_recovery_v2.py'
    check_source=check_template.read_text().replace(
        'from scripts.native_experimental_recovery_v2 import NativeExperimentalRecoveryV2',
        'from scripts.native_timed_experimental_recovery_v1 import NativeTimedExperimentalRecoveryV1 as NativeExperimentalRecoveryV2')
    check_source=check_source.replace("ROOT=Path(__file__).resolve().parents[1]",'ROOT=Path('+repr(str(ROOT))+')')
    ast.parse(check_source);check=args.output/'check_fresh_process.py';check.write_text(check_source)
    for p in [entry,check,check_template,bundle/'manifest.json',archive,
            ROOT/'scripts/native_ground_support_state_v1.py',ROOT/'scripts/native_ground_pair_observer_v1.py',
            ROOT/'scripts/run_native_timed_mpc_v1.py',
            ROOT/'scripts/audit_native_experimental_recovery_v1.py']:
        pins[str(p.resolve())]=sha(p)
    plan=dict(scope=__doc__,output=str(args.native_output.resolve()),bundle=str(bundle.resolve()),
        command=command,entry_sha256=sha(entry),source_sha256=pins,
        model_accepted=False,automatic_promotion=False,valid_expert_labels=False,
        check_command=['bash','scripts/python.sh',str(check.resolve()),'--bundle',str(bundle.resolve()),
            '--output',str((args.output/'fresh_process_check').resolve())],
        post_command=['bash','scripts/python.sh','scripts/audit_native_experimental_recovery_v1.py',
            '--run',str(args.native_output.resolve()),'--bundle',str(bundle.resolve()),
            '--output',str((args.output/'native_audit.json').resolve())])
    (args.output/'plan.json').write_text(json.dumps(plain(plan),indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(plan=str(args.output/'plan.json'),interface_check=interface_check)))


if __name__=='__main__':main()
