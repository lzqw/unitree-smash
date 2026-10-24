#!/usr/bin/env python3
"""Counterfactual joint targets on saved native states; no new dynamics."""
import argparse
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import sys
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
    warm=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    guard=FullRacketJointPreparation(warm['seed']['joint_preparation_guard']).float()
    model=FullRacketKinematicsV4().float()
    contract=json.loads(a.joint_contract.read_text())
    limits=torch.tensor([contract['limits'][f'joint{i}'] for i in range(1,7)])
    reset_path=ROOT/'configs/full_racket_reset_v4.json'
    default=torch.tensor(json.loads(reset_path.read_text())['arm_joint_positions_rad'])
    public_path=a.run/'near_contact_all_000.npz';control_path=a.run/'control_window_000.npz'
    values,commands=[],[]
    with np.load(public_path) as saved,np.load(control_path) as control,torch.inference_mode():
        for i in range(len(saved['observations'])):
            j=2*i
            obs=torch.tensor(saved['observations'][i]);valid=saved['valid_first_episode'][i]&control['valid_first_episode'][j]
            assert np.allclose(saved['episode_clock_s'][i,valid],control['clock_s'][j,valid],atol=1e-5)
            action=torch.tensor(control['actions'][j])
            q=limits[:,0]+.5*(action[:,12:]+1)*(limits[:,1]-limits[:,0])
            actual=default[None]+obs[:,21:27]
            corrected=guard(q,obs,limits,model,actual)
            assert torch.isfinite(corrected).all() and (corrected>=limits[:,0]).all() and (corrected<=limits[:,1]).all()
            post_hit=(obs[:,773]>.5) if guard.config['protect_post_hit_net_volume'] else None
            observed_vx=None
            if guard.config['anticipate_measured_motion']:
                observed_point=guard.support_state(actual,obs,model,post_hit,True)[4]
                velocity=obs[:,:3]+torch.linalg.cross(obs[:,3:6],observed_point)
                velocity+=obs[:,51:54]+torch.linalg.cross(obs[:,54:57],observed_point-obs[:,45:48])
                observed_vx=(velocity*obs[:,763:766]).sum(1)
            def support(joints):
                x,*_,point=guard.support_state(joints,obs,model,post_hit)
                v=obs[:,:3]+torch.linalg.cross(obs[:,3:6],point)
                vx=(v*obs[:,763:766]).sum(1)
                if observed_vx is not None:vx=torch.maximum(vx,observed_vx)
                margin=(guard.config['lookahead_s']*vx.clamp_min(0.)).clamp_max(guard.config['maximum_predictive_margin_m'])
                return torch.stack((x,margin),1)
            before,after=support(q),support(corrected)
            mask=valid&(obs[:,773].numpy()<.5)
            values.append(torch.cat((before,after,torch.tensor(mask)[:,None]),1).numpy())
            commands.append(corrected.numpy())
    data=np.stack(values);valid=data[:,:,4]>.5;maximum=guard.config['maximum_x_m']
    before=np.maximum(0,data[:,:,0]+data[:,:,1]-maximum)
    after=np.maximum(0,data[:,:,2]+data[:,:,3]-maximum)
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output/'counterfactual_native_state_targets.npz',support_and_margin=data,
                        corrected_arm_joint_targets_rad=np.stack(commands))
    sources=[a.checkpoint,a.joint_contract,public_path,control_path,reset_path,Path(__file__),
             ROOT/'quadsmash/control/full_racket_joint_preparation_v4.py']
    report=dict(created_at=datetime.now(timezone.utc).isoformat(),
        scope=f'Counterfactual commanded full-tool support on original {a.run.name} native public states and joint commands. No changed-state simulation or native avoidance claimed.',
        samples=len(data),lanes=data.shape[1],valid_unhit_samples=int(valid.sum()),
        unhit_commands_requiring_correction=int(np.sum(valid&(before>.001))),
        unhit_remaining_excess_over5mm=int(np.sum(valid&(after>.005))),
        maximum_initial_excess_m=float(before[valid].max()),maximum_remaining_excess_m=float(after[valid].max()),
        all_corrected_targets_finite_and_within_original_hardware_limits=True,
        configuration=guard.config,native_execution=False,formal_acceptance=False,
        sources={str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources})
    (a.output/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
