#!/usr/bin/env python3
"""Prove the old enabled preparation mode is unchanged on saved native states."""
import argparse
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import sys,types,zipfile
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from quadsmash.control.full_racket_joint_preparation_v4 import FullRacketJointPreparation
from quadsmash.control.full_racket_kinematics_v4 import FullRacketKinematicsV4


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--joint-contract',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Use a fresh output directory')
    torch.set_num_threads(2)
    source_zip=a.run/'source_snapshot.zip';module_name='quadsmash/control/full_racket_joint_preparation_v4.py'
    with zipfile.ZipFile(source_zip) as z:source=z.read(module_name)
    old=types.ModuleType('legacy_native_joint_preparation_v1');exec(compile(source,str(source_zip)+'/'+module_name,'exec'),old.__dict__)
    warm=torch.load(a.checkpoint,map_location='cpu',weights_only=True);config=warm['seed']['joint_preparation_guard']
    original=old.FullRacketJointPreparation(config).float();current=FullRacketJointPreparation(config).float()
    assert not current.config['anticipate_measured_motion'] and not current.config['protect_post_hit_net_volume']
    model=FullRacketKinematicsV4().float();contract=json.loads(a.joint_contract.read_text())
    limits=torch.tensor([contract['limits'][f'joint{i}'] for i in range(1,7)])
    defaults=torch.tensor(json.loads((ROOT/'configs/full_racket_reset_v4.json').read_text())['arm_joint_positions_rad'])
    public_path=a.run/'near_contact_all_000.npz';control_path=a.run/'control_window_000.npz';values=[]
    with np.load(public_path) as public,np.load(control_path) as control,torch.inference_mode():
        for i,observed in enumerate(public['observations']):
            obs=torch.tensor(observed);action=torch.tensor(control['actions'][2*i])
            q=limits[:,0]+.5*(action[:,12:]+1)*(limits[:,1]-limits[:,0])
            before=original(q,obs,limits,model);after=current(q,obs,limits,model,defaults[None]+obs[:,21:27])
            assert torch.equal(before,after),i
            values.append(after.numpy())
    a.output.mkdir(parents=True);np.savez_compressed(a.output/'legacy_equivalent_commands.npz',arm_joint_targets_rad=np.stack(values))
    sources=[source_zip,a.checkpoint,a.joint_contract,public_path,control_path,Path(__file__),ROOT/module_name]
    report=dict(created_at=datetime.now(timezone.utc).isoformat(),scope='CPU replay of the old enabled joint preparation mode on saved CEM10native states. No changed-state dynamics.',
        samples=len(values),lanes=len(values[0]),old_enabled_guard_commands_bitwise_equal=True,
        new_anticipation_and_posthit_modes_disabled=True,legacy_module_sha256=hashlib.sha256(source).hexdigest(),
        native_execution=False,formal_acceptance=False,
        sources={str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources})
    (a.output/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
