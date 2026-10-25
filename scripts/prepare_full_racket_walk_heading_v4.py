#!/usr/bin/env python3
"""Prepare walking heading feedback after exact legacy command replay."""
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
import quadsmash.control.full_racket_rally_controller_v4 as rally
from quadsmash.control.full_racket_walk_heading_v4 import validate_walk_heading_gain


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--public',type=Path,required=True)
    parser.add_argument('--legacy-commit',required=True)
    parser.add_argument('--gain',type=float,default=1.5)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    if a.output.exists():
        raise ValueError('Use a fresh output directory')
    gain=validate_walk_heading_gain(a.gain)
    if gain==0.:
        raise ValueError('Use a positive gain for this new initialization')
    torch.set_num_threads(2)
    source=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    assert source['native_execution'] and source['parameters'].shape==(38,)
    assert source['seed'].get('walk_heading_gain',0.)==0.
    prepared=copy.deepcopy(source)
    prepared['seed']['walk_heading_gain']=gain
    zero=copy.deepcopy(source);zero['seed']['walk_heading_gain']=0.
    path=ROOT/'quadsmash/control/full_racket_teacher_v4.py'
    legacy_text=subprocess.check_output(['git','show',a.legacy_commit+':quadsmash/control/full_racket_teacher_v4.py'],cwd=ROOT,text=True)
    old=types.ModuleType('quadsmash.control._walk_heading_legacy_teacher');old.__file__=str(path)
    exec(compile(legacy_text,str(path),'exec'),old.__dict__)
    current=rally.FullRacketTeacherV4
    controllers=[]
    for cls,checkpoint in [(old.FullRacketTeacherV4,source),(current,source),(current,zero),(current,prepared)]:
        rally.FullRacketTeacherV4=cls
        try:
            controllers.append(rally.FullRacketRallyControllerV4(checkpoint,2,'cpu'))
        finally:
            rally.FullRacketTeacherV4=current
    # This also refuses to call a merely written, uninstalled module ready.
    assert controllers[-1].teacher.walk_heading_gain==gain
    contract=json.loads((ROOT/'reports/G07_full_racket_v4/CPU_REPLAY_AUTHORED_JOINTS_0724.json').read_text())
    names=[f'{leg}_{joint}_joint' for joint in ['hip','thigh','calf'] for leg in ['FL','FR','RL','RR']]+[f'joint{i}' for i in range(1,7)]
    defaults=dict(contract['defaults_before_neutral_arm_override'])
    reset=json.loads((ROOT/'configs/full_racket_reset_v4.json').read_text())['arm_joint_positions_rad']
    defaults.update({f'joint{i+1}':v for i,v in enumerate(reset)})
    default=torch.tensor([defaults[n] for n in names],dtype=torch.float32)
    limits=torch.tensor([contract['limits'][n] for n in names],dtype=torch.float32)
    streams=[[] for _ in controllers]
    with np.load(a.public) as data:
        with torch.inference_mode():
            for obs,valid in zip(data['observations'],data['valid_first_episode']):
                if not valid.all():
                    break
                public=torch.tensor(obs);before=public.clone()
                for controller,stream in zip(controllers,streams):
                    action=controller.action({'policy':public},default,limits)
                    assert action.shape==(2,18) and torch.isfinite(action).all() and (abs(action)<=1).all()
                    assert torch.equal(public,before)
                    stream.append(action.numpy().copy())
    values=[np.stack(s) for s in streams]
    assert len(values[0])>=300
    assert np.array_equal(values[0],values[1]) and np.array_equal(values[0],values[2])
    assert np.array_equal(values[0][:,:,12:],values[3][:,:,12:])
    changed=np.any(values[0][:,:,:12]!=values[3][:,:,:12],axis=(1,2))
    assert changed.any()
    provenance={k:prepared.pop(k) for k in ['selected_source_run','selected_generation','selected_candidate','selected_training_case_index'] if k in prepared}
    prepared.pop('training_result',None)
    prepared.update(native_execution=False,performance_inherited=False,formal_acceptance=False,
        source_policy_provenance=provenance,source_policy_sha256=digest(a.checkpoint),
        kind='Unexecuted public walking-heading initialization; native training required.')
    a.output.mkdir(parents=True)
    torch.save(prepared,a.output/'policy.pt')
    (a.output/'legacy_teacher.py').write_text(legacy_text)
    report=dict(scope='CPU command replay only. Physical contact/recovery performance is unmeasured.',
        native_execution=False,formal_acceptance=False,performance_inherited=False,
        walk_heading_gain=gain,source_height_intercept=prepared['seed'].get('height_intercept'),
        replay_ticks=len(values[0]),default_and_zero_gain_bitwise_legacy=True,
        changed_leg_ticks=int(changed.sum()),changed_arm_ticks=0,
        changed_tick_min=int(np.flatnonzero(changed)[0]),changed_tick_max=int(np.flatnonzero(changed)[-1]),
        caveat='Unchanged arm commands apply to frozen-input replay only; walking changes the physical body and arm state in closed loop.',
        legacy_commit=a.legacy_commit,policy_sha256=digest(a.output/'policy.pt'),
        source_sha256={str(p):digest(p) for p in [a.checkpoint,a.public,path,Path(__file__),ROOT/'quadsmash/control/full_racket_walk_heading_v4.py']})
    (a.output/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
