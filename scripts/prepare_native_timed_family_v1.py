"""Prepare a full-ground, fixed128 recovery-reference transfer experiment."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['parent-plan','output','native-output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or a.native_output.exists():raise ValueError('Fresh paths required')
    read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    parent=read(a.parent_plan);pins=dict(parent['source_sha256'])
    if read(Path(parent['output'])/'job.json')['status']!='SUCCEEDED':raise ValueError('Completed parent required')
    for name,digest in pins.items():
        if sha(name)!=digest:raise ValueError('Changed input '+name)
    command=copy.deepcopy(parent['command'])
    old=next(Path(v) for v in command if v.endswith('evaluate_timed_recovery.py'))
    if sha(old)!=parent['entry_sha256']:raise ValueError('Wrong source entry')
    source=old.read_text()
    changes={
        'from scripts.native_timed_experimental_recovery_v1 import NativeTimedExperimentalRecoveryV1 as NativeRecoverySequenceReplay':
        'from scripts.native_timed_recovery_family_v1 import NativeTimedRecoveryFamilyV1 as NativeRecoverySequenceReplay',
        '        sequence_student_commands=[];sequence_final_commands=[];sequence_first_episode=[]':
        '        native_replay.set_environment(env)\n'
        '        sequence_student_commands=[];sequence_final_commands=[];sequence_first_episode=[]',
    }
    for before,after in changes.items():
        if source.count(before)!=1:raise ValueError('Unexpected source entry')
        source=source.replace(before,after)
    ast.parse(source);a.output.mkdir(parents=True)
    entry=a.output/'evaluate_timed_family.py';entry.write_text(source)
    command=[str(entry.resolve()) if v==str(old) else v for v in command]
    for i,v in enumerate(command[:-1]):
        if v=='--output':command[i+1]=str(a.native_output.resolve())
    for path in [a.parent_plan,Path(__file__),entry,ROOT/'scripts/native_timed_recovery_family_v1.py',
                 ROOT/'scripts/audit_native_timed_family_v1.py']:
        pins[str(path.resolve())]=sha(path)
    plan=dict(parent,output=str(a.native_output.resolve()),command=command,entry_sha256=sha(entry),
        source_sha256=pins,family_trial=True,new_mpc_solution=False,
        post_command=['bash','scripts/python.sh','scripts/audit_native_timed_family_v1.py',
            '--run',str(a.native_output.resolve()),'--bundle',parent['bundle'],
            '--output',str((a.output/'native_audit.json').resolve())])
    (a.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(plan=str(a.output/'plan.json'),native_execution=False)))


if __name__=='__main__':main()
