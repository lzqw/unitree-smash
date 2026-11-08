"""Version full-remainder MPC with independently filtered ground observations."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['parent-plan','output','native-output','prior-job']:
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or a.native_output.exists():
        raise ValueError('Fresh output paths required')
    read=lambda p:json.loads(Path(p).read_text())
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    parent=read(a.parent_plan)
    if read(Path(parent['output'])/'job.json')['status']!='SUCCEEDED' or read(a.prior_job)['status']!='SUCCEEDED':
        raise ValueError('Completed parent and previous serial native run required')
    pins=dict(parent['source_sha256'])
    for path,digest in pins.items():
        if sha(ROOT/path)!=digest:
            raise ValueError('Changed parent dependency '+path)
    command=copy.deepcopy(parent['command']);old=Path(command[command.index('-u')+1])
    if sha(old)!=parent['entry_sha256']:
        raise ValueError('Changed parent evaluator')
    source=old.read_text()
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
        "    finally:\n        if 'native_replay' in locals():":
        "    finally:\n        if 'ground_support' in locals():\n            ground_support.save()\n"
        "        if 'native_replay' in locals():",
    }
    for before,after in changes.items():
        if source.count(before)!=1:
            raise ValueError('Unexpected entry boundary '+before)
        source=source.replace(before,after)
    ast.parse(source);a.output.mkdir(parents=True)
    entry=a.output/'evaluate_ground_support_mpc.py';entry.write_text(source)
    command=[str(entry.resolve()) if v==str(old) else
        str(a.native_output.resolve()) if v==parent['output'] else v for v in command]
    post=[]
    for row in parent['post_commands']:
        new=[]
        for v in row:
            if v==parent['output']:v=str(a.native_output.resolve())
            elif v.startswith(str(a.parent_plan.parent.resolve())+'/'):
                v=str(a.output.resolve()/Path(v).name)
            new.append(v)
        post.append(new)
    post.insert(0,['bash','scripts/python.sh','scripts/audit_ground_support_mpc_input_v1.py',
        '--run',str(a.native_output.resolve()),'--output',str((a.output/'ground_support_audit.json').resolve())])
    for path in [a.parent_plan,Path(__file__),entry,ROOT/'scripts/native_ground_support_state_v1.py',
            ROOT/'scripts/native_ground_pair_observer_v1.py',ROOT/'scripts/audit_ground_support_mpc_input_v1.py']:
        pins[str(path.resolve())]=sha(path)
    plan=dict(parent,output=str(a.native_output.resolve()),command=command,post_commands=post,
        source_sha256=pins,entry_sha256=sha(entry),previous_process_ids=[],prior_native_job=str(a.prior_job.resolve()),
        scope='Independent ground-only controller contacts; same four full-remainder actual-state replans and continuous feedback.',
        original_task_thresholds_changed=False,student_observation_changed=False,
        controller_contact_source_changed=True,contact_entry_n=1.,contact_release_n=.5,load_reporting_n=5.,
        no_light_contact_exclusion=True,model_accepted=False,automatic_promotion=False,valid_expert_labels=False)
    (a.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(plan=str(a.output/'plan.json'),native_execution=False)))


if __name__=='__main__':main()
