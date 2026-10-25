#!/usr/bin/env python3
"""Check zero-residual equivalence on an actual recorded public stream."""
import argparse,copy,hashlib,io,json,subprocess,sys,types
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from quadsmash.control.full_racket_rally_controller_v4 import FullRacketRallyControllerV4
from quadsmash.control.full_racket_residual_v4 import FullRacketResidualControllerV4,SCHEMA


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--public',type=Path,required=True)
    p.add_argument('--legacy-commit',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists();torch.set_num_threads(2)
    source=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    actor=torch.nn.Linear(794,18)
    with torch.no_grad():actor.weight.zero_();actor.bias.zero_()
    buffer=io.BytesIO();torch.jit.save(torch.jit.script(actor.eval()),buffer)
    zero=copy.deepcopy(source);zero.update(schema=SCHEMA,base_controller=copy.deepcopy(source),
        residual_actor_jit=buffer.getvalue(),residual_joint_scale_rad=[.12]*12+[.15]*6,
        native_execution=False,performance_inherited=False,formal_acceptance=False,
        kind='Unexecuted zero-actor CPU command compatibility check only.')
    path=ROOT/'quadsmash/control/full_racket_rally_controller_v4.py'
    old_text=subprocess.check_output(['git','show',a.legacy_commit+':quadsmash/control/full_racket_rally_controller_v4.py'],cwd=ROOT,text=True)
    old=types.ModuleType('quadsmash.control._residual_legacy_rally');old.__file__=str(path)
    exec(compile(old_text,str(path),'exec'),old.__dict__)
    controllers=[old.FullRacketRallyControllerV4(source,2,'cpu'),
        FullRacketRallyControllerV4(source,2,'cpu'),FullRacketResidualControllerV4(zero,2,'cpu')]
    for c in controllers:c.reset()
    c=json.loads((ROOT/'reports/G07_full_racket_v4/CPU_REPLAY_AUTHORED_JOINTS_0724.json').read_text())
    names=[f'{leg}_{joint}_joint' for joint in ['hip','thigh','calf'] for leg in ['FL','FR','RL','RR']]+[f'joint{i}' for i in range(1,7)]
    defaults=dict(c['defaults_before_neutral_arm_override'])
    arm=json.loads((ROOT/'configs/full_racket_reset_v4.json').read_text())['arm_joint_positions_rad']
    defaults.update({f'joint{i+1}':v for i,v in enumerate(arm)})
    default=torch.tensor([defaults[n] for n in names]);limits=torch.tensor([c['limits'][n] for n in names])
    ticks=0
    with np.load(a.public) as data:
        with torch.inference_mode():
            for public,valid in zip(data['observations'],data['valid_first_episode']):
                if not valid.all():break
                obs=torch.tensor(public);before=obs.clone()
                values=[controller.action({'policy':obs},default,limits) for controller in controllers]
                assert torch.equal(values[0],values[1]) and torch.equal(values[0],values[2])
                assert torch.equal(obs,before) and torch.isfinite(values[2]).all() and (values[2].abs()<=1).all()
                ticks+=1
    assert ticks==800
    a.output.mkdir(parents=True);torch.save(zero,a.output/'zero_residual_cpu_check.pt')
    (a.output/'legacy_rally_controller.py').write_text(old_text)
    report=dict(scope='CPU recorded-input command equivalence only; not a native residual training/evaluation result.',
        native_execution=False,formal_acceptance=False,performance_inherited=False,replay_ticks=ticks,
        lanes=2,all18D_actions_bitwise_legacy=True,observation_immutable=True,
        legacy_commit=a.legacy_commit,source_sha256={str(p):sha(p) for p in [a.checkpoint,a.public,path,
            ROOT/'quadsmash/control/full_racket_residual_v4.py',Path(__file__)]})
    (a.output/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
