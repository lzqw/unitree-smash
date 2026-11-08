"""Prepare an explicitly experimental timed-prefix/current-state MPC execution."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['parent-plan','solver-result','service-result','output','native-output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or a.native_output.exists():raise ValueError('Fresh native evidence required')
    read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    parent=read(a.parent_plan);bundle_old=Path(parent['bundle']);m=copy.deepcopy(read(bundle_old/'manifest.json'))
    solved=read(a.solver_result);decision=solved['decision'];service=read(a.service_result)
    if (m['case']!=85 or not decision['accepted'] or decision['control_tick']!=400
            or not decision['independent_replay_exact'] or not service['passed'] or service['service_exit_code']!=0):
        raise ValueError('Verified full-remainder query and real CPU service check required')
    pins=dict(parent['source_sha256'])
    for path,digest in solved['source_sha256'].items():
        if path in pins and pins[path]!=digest:raise ValueError('Model/native inputs disagree '+path)
        pins[path]=digest
    for path in [a.parent_plan,a.solver_result,a.service_result,Path(__file__).resolve(),
            ROOT/'scripts/native_timed_position_mpc_v1.py',ROOT/'scripts/current_state_timed_recovery_service_v4.py',
            ROOT/'scripts/run_native_timed_mpc_v1.py']:
        pins[str(path.resolve())]=sha(path)
    for path,digest in pins.items():
        if sha(path)!=digest:raise ValueError('Pinned evidence changed '+path)
    source_plan=read(a.solver_result.parent/'inputs.json')['configuration']['model_plan']
    m['timed_position_stop_mpc']=dict(execution_contract='timed_position_stop_mpc_v1',
        control_ticks=[400,550,700],max_seconds_per_call=600.,source_plan=str(Path(source_plan).resolve()),
        warm_plan=decision['plan'],seed_query=decision['query'],seed_result_sha256=sha(a.solver_result))
    m['source_sha256']=pins;m['valid_expert_labels']=False
    a.output.mkdir(parents=True);bundle=a.output/'bundle';bundle.mkdir()
    shutil.copyfile(bundle_old/m['bundle_name'],bundle/m['bundle_name'])
    if sha(bundle/m['bundle_name'])!=m['bundle_sha256']:raise ValueError('Reference copy differs')
    (bundle/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    command=copy.deepcopy(parent['command']);old_entry=next(Path(v) for v in command if v.endswith('evaluate_timed_recovery.py'))
    source=old_entry.read_text();old='from scripts.native_timed_experimental_recovery_v1 import NativeTimedExperimentalRecoveryV1 as NativeRecoverySequenceReplay'
    if source.count(old)!=1:raise ValueError('Unexpected timed evaluator template')
    source=source.replace(old,'from scripts.native_timed_position_mpc_v1 import NativeTimedPositionMPCV1 as NativeRecoverySequenceReplay')
    ast.parse(source);entry=a.output/'evaluate_timed_position_mpc.py';entry.write_text(source)
    command=[str(entry.resolve()) if v==str(old_entry) else v for v in command]
    for i,v in enumerate(command[:-1]):
        if v=='--output':command[i+1]=str(a.native_output.resolve())
        elif v=='--recovery-sequence-bundle':command[i+1]=str(bundle.resolve())
        elif v=='--max-seconds':command[i+1]='4200'
    template=ROOT/'scripts/check_native_experimental_recovery_v2.py';check=template.read_text()
    check=check.replace('from scripts.native_experimental_recovery_v2 import NativeExperimentalRecoveryV2',
        'from scripts.native_timed_position_mpc_v1 import NativeTimedPositionMPCV1')
    check=check.replace('NativeExperimentalRecoveryV2(','NativeTimedPositionMPCV1(')
    check=check.replace('ROOT=Path(__file__).resolve().parents[1]','ROOT=Path('+repr(str(ROOT))+')')
    ast.parse(check);check_path=a.output/'check_fresh_process.py';check_path.write_text(check)
    for path in [entry,check_path,template,bundle/'manifest.json',bundle/m['bundle_name']]:pins[str(path.resolve())]=sha(path)
    plan=dict(scope=__doc__,output=str(a.native_output.resolve()),bundle=str(bundle.resolve()),command=command,
        entry_sha256=sha(entry),source_sha256=pins,model_accepted=False,automatic_promotion=False,valid_expert_labels=False,
        check_command=['bash','scripts/python.sh',str(check_path.resolve()),'--bundle',str(bundle.resolve()),'--output',str((a.output/'fresh_process_check').resolve())],
        post_command=['bash','scripts/python.sh','scripts/audit_native_experimental_recovery_v1.py','--run',str(a.native_output.resolve()),
            '--bundle',str(bundle.resolve()),'--output',str((a.output/'native_audit.json').resolve())])
    (a.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(plan=str(a.output/'plan.json'),case=85,planning_ticks=[400,550,700],native_execution=False)))


if __name__=='__main__':main()
