#!/usr/bin/env python3
"""Compare exact live37teacher, updated37and zero-bias38on frozen public inputs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import types
import zipfile
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from quadsmash.control.full_racket_teacher_v4 import FullRacketTeacherV4
from quadsmash.control.full_racket_contact_normal_search_v4 import extend_contact_normal_parameters
from quadsmash.control.full_racket_parameter_layout_v4 import controller_parameter_names


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['run','checkpoint','public','joint-contract','output']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Use a fresh output directory')
    torch.set_num_threads(2)
    module_path='quadsmash/control/full_racket_teacher_v4.py'
    with zipfile.ZipFile(a.run/'source_snapshot.zip') as archive:source=archive.read(module_path)
    old=types.ModuleType('quadsmash.control.previous_contact_normal_teacher');old.__package__='quadsmash.control'
    exec(compile(source,str(a.run/'source_snapshot.zip')+'/'+module_path,'exec'),old.__dict__)
    warm=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    assert len(warm['parameters'])==37
    assert warm['parameter_names']==controller_parameter_names(search_arm_velocity=True,search_recovery_parameters=True)
    contract=json.loads(a.joint_contract.read_text())
    names=[f'{leg}_{joint}_joint' for joint in ['hip','thigh','calf'] for leg in ['FL','FR','RL','RR']]+[f'joint{i}' for i in range(1,7)]
    defaults=contract['defaults_before_neutral_arm_override']
    defaults.update({f'joint{i+1}':q for i,q in enumerate(json.loads((ROOT/'configs/full_racket_reset_v4.json').read_text())['arm_joint_positions_rad'])})
    default=torch.tensor([defaults[n] for n in names]);limits=torch.tensor([contract['limits'][n] for n in names])
    all_commands,variants=[],[]
    for handover in [float(warm['parameters'][3]),1.20]:
        parameters=warm['parameters'].clone();parameters[3]=handover
        extended=extend_contact_normal_parameters(parameters);changed=extended.clone();changed[37]=.04
        values=[x[None].repeat(2,1) for x in [parameters,parameters,extended,changed]]
        teachers=[cls(warm['backbone'],warm['seed'],ROOT/'configs/g07_go2_walk_transfer.json',2,'cpu')
                  for cls in [old.FullRacketTeacherV4,FullRacketTeacherV4,FullRacketTeacherV4,FullRacketTeacherV4]]
        for teacher in teachers:teacher.reset()
        commands=[];changed_ticks=0
        with np.load(a.public) as data,torch.inference_mode():
            for i,observed in enumerate(data['observations']):
                if not data['valid_first_episode'][i].all():break
                obs=torch.tensor(observed)
                actions=[t.action({'policy':obs},v,default,limits) for t,v in zip(teachers,values)]
                assert torch.equal(actions[0],actions[1]) and torch.equal(actions[0],actions[2]),(handover,i)
                assert torch.equal(actions[2][:,:12],actions[3][:,:12]),(handover,i)
                assert all(torch.isfinite(x).all() and (abs(x)<=1).all() for x in actions)
                changed_ticks+=int(not torch.equal(actions[2][:,12:],actions[3][:,12:]))
                commands.append(torch.stack(actions).numpy())
        assert len(commands)>200 and changed_ticks>0
        variants.append(dict(local_handover_time_s=handover,ticks=len(commands),
                             exact_native37_updated37_default38_bitwise_equal=True,
                             altered_normal_bias_changes_arms_only_on_identical_public_inputs=True,
                             changed_arm_command_ticks=changed_ticks))
        all_commands.append(np.stack(commands))
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output/'counterfactual_commands.npz',actions=np.stack(all_commands))
    paths=[a.run/'source_snapshot.zip',a.checkpoint,a.public,a.joint_contract,Path(__file__),ROOT/module_path,
           ROOT/'quadsmash/control/full_racket_contact_normal_search_v4.py',
           ROOT/'quadsmash/control/full_racket_parameter_layout_v4.py',
           ROOT/'quadsmash/control/full_racket_arm_search_v4.py',ROOT/'quadsmash/control/full_racket_recovery_search_v4.py']
    report=dict(created_at=datetime.now(timezone.utc).isoformat(),
        scope='CPU commands on identical frozen public inputs, not changed-state dynamics or observed ball direction/landing/recovery.',
        lanes=2,variants=variants,exact_native_teacher_sha256=hashlib.sha256(source).hexdigest(),
        native_execution=False,formal_acceptance=False,
        source_sha256={str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
    (a.output/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'}),flush=True)


if __name__=='__main__':main()
