"""Prepare an unchanged recovery replay with a focused native foot-pair view."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['parent-plan','output','native-output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if a.output.exists() or a.native_output.exists():raise ValueError('Fresh output required')
    parent=json.loads(a.parent_plan.read_text());parent_run=Path(parent['output'])
    job=json.loads((parent_run/'job.json').read_text())
    if job['status']!='SUCCEEDED' or job['exit_code']!=0:raise ValueError('Completed parent required')
    pins=dict(parent['source_sha256'])
    for path,h in pins.items():
        if sha(path)!=h:raise ValueError('Pinned parent changed '+path)
    command=copy.deepcopy(parent['command'])
    old=next(Path(x) for x in command if x.endswith('evaluate_joint_timed_family.py'))
    if sha(old)!=parent['entry_sha256']:raise ValueError('Parent entry differs')
    text=old.read_text()
    changes={
        '        head_observer=NativeHeadGroundObservationV1(env,args.output)':
        '        head_observer=NativeHeadGroundObservationV1(env,args.output)\n'
        '        from scripts.native_selected_pair_observer_v1 import NativeSelectedPairObserverV1\n'
        '        filtered_observer=NativeSelectedPairObserverV1(env,args.output,lanes=(71,125),first_frame=50880,last_frame=52480)',
        "    finally:\n        if 'ground_support' in locals():":
        "    finally:\n        if 'filtered_observer' in locals():\n"
        '            filtered_observer.save()\n            filtered_observer.detach()\n'
        "        if 'ground_support' in locals():"}
    for before,after in changes.items():
        if text.count(before)!=1:raise ValueError('Unexpected observer lifecycle')
        text=text.replace(before,after)
    ast.parse(text);a.output.mkdir(parents=True)
    entry=a.output/'evaluate_selected_pair.py';entry.write_text(text)
    command=[str(entry.resolve()) if x==str(old) else x for x in command]
    for i,x in enumerate(command[:-1]):
        if x=='--output':command[i+1]=str(a.native_output.resolve())
        elif x=='--max-seconds':command[i+1]='3000'
    check=copy.deepcopy(parent['check_command']);check[check.index('--output')+1]=str((a.output/'fresh_process_check').resolve())
    for path in [a.parent_plan,Path(__file__),entry,ROOT/'scripts/native_selected_pair_observer_v1.py',
        ROOT/'scripts/native_ground_pair_observer_v1.py',ROOT/'scripts/native_timed_impact_observer_v1.py',
        ROOT/'scripts/native_foot_impact_observer_v1.py',ROOT/'scripts/audit_selected_pair_observation_v1.py',
        ROOT/'scripts/diagnose_frozen_trace_divergence_v1.py',ROOT/'scripts/native_recovery_query_bank_v1.py']:
        pins[str(path.resolve())]=sha(path)
    plan=dict(parent,output=str(a.native_output.resolve()),command=command,check_command=check,
        entry_sha256=sha(entry),source_sha256=pins,observation_parent=str(parent_run.resolve()),
        observer_controls_changed=False,post_command=['bash','scripts/python.sh','scripts/audit_selected_pair_observation_v1.py',
            '--run',str(a.native_output.resolve()),'--parent',str(parent_run.resolve()),'--bundle',parent['bundle'],
            '--output',str((a.output/'native_audit.json').resolve())])
    (a.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(plan=str(a.output/'plan.json'),controls_changed=False,native_execution=False)))


if __name__=='__main__':main()
