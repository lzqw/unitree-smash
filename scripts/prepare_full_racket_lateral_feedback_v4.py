#!/usr/bin/env python3
"""Replay native public inputs and prepare unexecuted lateral-feedback proposals."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from quadsmash.control.full_racket_rally_controller_v4 import FullRacketRallyControllerV4
from quadsmash.control.full_racket_candidate_bank_v4 import validate_candidate_bank
from quadsmash.control.full_racket_teacher_v4 import LOW,HIGH
from quadsmash.control.full_racket_recovery_search_v4 import RECOVERY_LOW,RECOVERY_HIGH
from quadsmash.control.full_racket_contact_normal_search_v4 import NORMAL_LOW,NORMAL_HIGH


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--public',type=Path,required=True)
    p.add_argument('--legacy-commit',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise ValueError('Use a fresh output')
    torch.set_num_threads(2)
    source=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    assert len(source['parameters'])==38
    legacy_text=subprocess.check_output(['git','show',a.legacy_commit+':quadsmash/control/full_racket_teacher_v4.py'],cwd=ROOT,text=True)
    module=types.ModuleType('quadsmash.control._legacy_teacher')
    exec(compile(legacy_text,'legacy_teacher.py','exec'),module.__dict__)
    contract=json.loads((ROOT/'reports/G07_full_racket_v4/CPU_REPLAY_AUTHORED_JOINTS_0724.json').read_text())
    names=[f'{leg}_{joint}_joint' for joint in ['hip','thigh','calf'] for leg in ['FL','FR','RL','RR']]+[f'joint{i}' for i in range(1,7)]
    defaults=contract['defaults_before_neutral_arm_override']
    reset=json.loads((ROOT/'configs/full_racket_reset_v4.json').read_text())['arm_joint_positions_rad']
    defaults.update({f'joint{i+1}':v for i,v in enumerate(reset)})
    default=torch.tensor([defaults[n] for n in names],dtype=torch.float32)
    limits=torch.tensor([contract['limits'][n] for n in names],dtype=torch.float32)
    adaptive=copy.deepcopy(source)
    adaptive['seed']['lateral_residual_feedback']=True
    legacy=FullRacketRallyControllerV4(source,2,'cpu')
    legacy.teacher=module.FullRacketTeacherV4(source['backbone'],source['seed'],ROOT/'configs/g07_go2_walk_transfer.json',2,'cpu')
    controllers=[legacy,FullRacketRallyControllerV4(source,2,'cpu'),FullRacketRallyControllerV4(adaptive,2,'cpu')]
    streams=[[] for _ in controllers]
    data=np.load(a.public)
    with torch.inference_mode():
        for obs,valid in zip(data['observations'],data['valid_first_episode']):
            if not valid.all(): break
            public=torch.tensor(obs)
            before=public.clone()
            for controller,stream in zip(controllers,streams):
                action=controller.action({'policy':public},default,limits)
                assert action.shape==(2,18) and torch.isfinite(action).all() and (action.abs()<=1).all()
                assert torch.equal(public,before), 'Controller mutated its public observation'
                stream.append(action.numpy().copy())
    values=[np.stack(stream) for stream in streams]
    assert len(values[0])>=300
    assert np.array_equal(values[0],values[1]), 'Disabled mode changed legacy commands'
    assert np.array_equal(values[1][:,:,:12],values[2][:,:,:12]), 'Replay leg commands changed'
    assert not np.array_equal(values[1][:,:,12:],values[2][:,:,12:])
    adaptive.pop('training_result',None)
    adaptive.update(native_execution=False,performance_inherited=False,formal_acceptance=False,
                    kind='Unexecuted lateral-residual feedback initialization; requires fresh native search.')
    ref=adaptive['parameters'].numpy().copy()
    lower=np.array(LOW+[.8,-.12]+[-5.]*6+RECOVERY_LOW+NORMAL_LOW,dtype=np.float32)
    upper=np.array(HIGH+[1.45,.12]+[5.]*6+RECOVERY_HIGH+NORMAL_HIGH,dtype=np.float32)
    lower[3]=1.2;lower[22:24],upper[22:24]=-.3,.3
    proposals=[];labels=[]
    for gain in [0.,.15,.30,.50,.70,.85,1.]:
        for lead in [ref[14],ref[14]+.02]:
            row=ref.copy();row[18]=gain;row[14]=np.clip(lead,lower[14],upper[14])
            proposals.append(row.tolist());labels.append(dict(cartesian_gain=gain,arm_lead=float(lead)))
    bank=dict(schema='full-racket-candidate-bank-v4',asset_sha256=adaptive['asset_sha256'],
              seed=adaptive['seed'],parameter_names=adaptive['parameter_names'],reference_parameters=ref.tolist(),
              parameters=proposals,proposal_labels=labels,native_execution=False,formal_acceptance=False)
    validate_candidate_bank(bank,adaptive['seed'],adaptive['asset_sha256'],adaptive['parameter_names'],ref,lower,upper,14)
    a.output.mkdir(parents=True)
    torch.save(adaptive,a.output/'policy.pt')
    (a.output/'CANDIDATE_BANK.json').write_text(json.dumps(bank,indent=2)+'\n')
    (a.output/'legacy_teacher.py').write_text(legacy_text)
    report=dict(scope='CPU command replay only; no dynamics/performance claim',native_execution=False,formal_acceptance=False,
                replay_ticks=len(values[0]),disabled_mode_bitwise_legacy=True,enabled_legs_bitwise_preserved=True,
                enabled_arm_changed_ticks=int(np.any(values[1][:,:,12:]!=values[2][:,:,12:],axis=(1,2)).sum()),
                proposals=len(proposals),legacy_commit=a.legacy_commit,
                source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [a.checkpoint,a.public,Path(__file__),ROOT/'quadsmash/control/full_racket_teacher_v4.py']})
    (a.output/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
