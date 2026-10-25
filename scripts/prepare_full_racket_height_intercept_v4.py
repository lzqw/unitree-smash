#!/usr/bin/env python3
"""Prepare a causal height-planning policy and replay its bounded commands."""
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
from quadsmash.control.full_racket_height_intercept_v4 import validate_height_intercept


def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--public',type=Path,required=True)
    p.add_argument('--legacy-commit',required=True)
    p.add_argument('--height',type=float,default=1.76)
    p.add_argument('--fraction',type=float,default=.5)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise ValueError('Use a new output directory')
    torch.set_num_threads(2)
    source=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    assert source['native_execution'] and source['parameters'].shape==(38,)
    assert source['seed'].get('height_intercept') is None
    cfg=validate_height_intercept(dict(target_height_m=a.height,adaptation_fraction=a.fraction,plane_count=33))
    prepared=copy.deepcopy(source);prepared['seed']['height_intercept']=cfg
    zero=copy.deepcopy(prepared);zero['seed']['height_intercept']['adaptation_fraction']=0.
    path=ROOT/'quadsmash/control/full_racket_rally_controller_v4.py'
    old_text=subprocess.check_output(['git','show',a.legacy_commit+':quadsmash/control/full_racket_rally_controller_v4.py'],cwd=ROOT,text=True)
    old=types.ModuleType('quadsmash.control._height_legacy_rally');old.__file__=str(path)
    exec(compile(old_text,str(path),'exec'),old.__dict__)
    controllers=[old.FullRacketRallyControllerV4(source,2,'cpu'),
                 FullRacketRallyControllerV4(source,2,'cpu'),
                 FullRacketRallyControllerV4(zero,2,'cpu'),
                 FullRacketRallyControllerV4(prepared,2,'cpu')]
    contract=json.loads((ROOT/'reports/G07_full_racket_v4/CPU_REPLAY_AUTHORED_JOINTS_0724.json').read_text())
    names=[f'{leg}_{joint}_joint' for joint in ['hip','thigh','calf'] for leg in ['FL','FR','RL','RR']]+[f'joint{i}' for i in range(1,7)]
    defaults=contract['defaults_before_neutral_arm_override']
    reset=json.loads((ROOT/'configs/full_racket_reset_v4.json').read_text())['arm_joint_positions_rad']
    defaults.update({f'joint{i+1}':v for i,v in enumerate(reset)})
    default=torch.tensor([defaults[n] for n in names],dtype=torch.float32)
    limits=torch.tensor([contract['limits'][n] for n in names],dtype=torch.float32)
    streams=[[] for _ in controllers]
    data=np.load(a.public)
    with torch.inference_mode():
        for obs,valid in zip(data['observations'],data['valid_first_episode']):
            if not valid.all(): break
            public=torch.tensor(obs);before=public.clone()
            for controller,stream in zip(controllers,streams):
                action=controller.action({'policy':public},default,limits)
                assert action.shape==(2,18) and torch.isfinite(action).all() and (abs(action)<=1).all()
                assert torch.equal(public,before)
                stream.append(action.numpy().copy())
    values=[np.stack(s) for s in streams]
    assert len(values[0])>=300
    assert np.array_equal(values[0],values[1]) and np.array_equal(values[0],values[2])
    assert not np.array_equal(values[0],values[3])
    selected=controllers[3].logs[0]
    assert all(.8<=v<=1.45 for v in selected['selected_intercept_plane_x_m'])
    provenance={k:prepared.pop(k) for k in ['selected_source_run','selected_generation','selected_candidate','selected_training_case_index'] if k in prepared}
    prepared.pop('training_result',None)
    prepared.update(native_execution=False,performance_inherited=False,formal_acceptance=False,
        source_policy_provenance=provenance,source_policy_sha256=digest(a.checkpoint),
        kind='Unexecuted causal height-interception initialization; native training required.')
    a.output.mkdir(parents=True)
    torch.save(prepared,a.output/'policy.pt')
    (a.output/'legacy_rally_controller.py').write_text(old_text)
    (a.output/'CPU_PLANS.json').write_text(json.dumps(selected,indent=2)+'\n')
    report=dict(scope='CPU command replay only; no dynamics/success guarantee.',native_execution=False,formal_acceptance=False,
        config=cfg,replay_ticks=len(values[0]),default_and_zero_fraction_bitwise_legacy=True,
        changed_arm_ticks=int(np.any(values[0][:,:,12:]!=values[3][:,:,12:],axis=(1,2)).sum()),
        changed_leg_ticks=int(np.any(values[0][:,:,:12]!=values[3][:,:,:12],axis=(1,2)).sum()),
        legacy_commit=a.legacy_commit,policy_sha256=digest(a.output/'policy.pt'),
        source_sha256={str(p):digest(p) for p in [a.checkpoint,a.public,path,Path(__file__),ROOT/'quadsmash/control/full_racket_height_intercept_v4.py']})
    (a.output/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
