#!/usr/bin/env python3
"""Replay public inputs to check optional arm target slew and legacy commands."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import types
import zipfile

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from quadsmash.control.full_racket_rally_controller_v4 import FullRacketRallyControllerV4


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--public',type=Path,required=True)
    p.add_argument('--joint-contract',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists();torch.set_num_threads(2)
    modules=[]
    with zipfile.ZipFile(a.run/'source_snapshot.zip') as z:
        for name in ['full_racket_teacher_v4','full_racket_rally_controller_v4']:
            path=f'quadsmash/control/{name}.py';module=types.ModuleType('legacy_'+name)
            module.__file__=path;module.__package__='quadsmash.control'
            exec(compile(z.read(path),path,'exec'),module.__dict__);modules.append(module)
    modules[1].FullRacketTeacherV4=modules[0].FullRacketTeacherV4
    warm=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    assert warm['seed'].get('arm_target_rate_limit_rad_s') is None
    variants=[warm,warm]
    for rate in [4.5,5.]:
        variant=copy.deepcopy(warm);variant['seed']['arm_target_rate_limit_rad_s']=rate;variants.append(variant)
    controllers=[modules[1].FullRacketRallyControllerV4(variants[0],2,'cpu')]
    controllers += [FullRacketRallyControllerV4(variant,2,'cpu') for variant in variants[1:]]
    contract=json.loads(a.joint_contract.read_text())
    names=[f'{leg}_{j}_joint' for j in ['hip','thigh','calf'] for leg in ['FL','FR','RL','RR']]+[f'joint{i}' for i in range(1,7)]
    defaults=contract['defaults_before_neutral_arm_override'].copy()
    reset=json.loads((ROOT/'configs/full_racket_reset_v4.json').read_text())['arm_joint_positions_rad']
    defaults.update({f'joint{i+1}':value for i,value in enumerate(reset)})
    default=torch.tensor([defaults[name] for name in names]);limits=torch.tensor([contract['limits'][name] for name in names])
    data=np.load(a.public);values=[[] for _ in controllers];clocks=[];initial=None
    for controller in controllers:controller.reset()
    with torch.inference_mode():
        for i,obs in enumerate(data['observations']):
            if not data['valid_first_episode'][i].all():break
            public=torch.tensor(obs)
            if initial is None:initial=default[12:][None]+public[:,21:27]
            clocks.append(obs[:,775].copy())
            for controller,result in zip(controllers,values):
                action=controller.action({'policy':public},default,limits)
                assert action.shape==(2,18) and torch.isfinite(action).all() and (abs(action)<=1).all()
                result.append(action.numpy().copy())
    values=[np.stack(value) for value in values];clocks=np.stack(clocks)
    assert len(clocks)>280 and np.array_equal(values[0],values[1])
    maximum_rates=[]
    for output,rate in zip(values[2:],[4.5,5.]):
        assert np.array_equal(values[1][:,:,:12],output[:,:,:12])
        assert not np.array_equal(values[1][:,:,12:],output[:,:,12:])
        q=limits[12:,0].numpy()+.5*(output[:,:,12:]+1)*(limits[12:,1]-limits[12:,0]).numpy()
        dt=np.r_[np.full((1,2),.01),np.diff(clocks,axis=0)]
        before=np.concatenate([initial.numpy()[None],q[:-1]])
        rates=np.abs(q-before)/dt[:,:,None]
        assert rates.max()<=rate+1e-4,(rate,rates.max())
        maximum_rates.append(float(rates.max()))
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output/'counterfactual_commands.npz',legacy=values[0],disabled=values[1],rate4_5=values[2],rate5=values[3],clock_s=clocks)
    paths=[a.checkpoint,a.public,a.joint_contract,a.run/'source_snapshot.zip',Path(__file__),
        ROOT/'quadsmash/control/full_racket_teacher_v4.py',ROOT/'quadsmash/control/full_racket_arm_target_rate_v4.py']
    report=dict(created_at=datetime.now(timezone.utc).isoformat(),scope='Counterfactual CPU commands on fixed native public observations. Rate-limited robot dynamics, face tracking, contacts, outgoing ball and recovery are unexecuted.',
        ticks=len(clocks),lanes=2,legacy_default_commands_bitwise_equal=True,all_actions_finite_bounded=True,
        software_arm_target_rate_limits_rad_s=[4.5,5.],measured_maximum_command_rates_rad_s=maximum_rates,
        numerical_command_rate_tolerance_rad_s=1e-4,legs_bitwise_unchanged_on_same_inputs=True,
        physical_actuator_limits_unchanged=True,native_execution=False,formal_acceptance=False,
        sha256={str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
    (a.output/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
