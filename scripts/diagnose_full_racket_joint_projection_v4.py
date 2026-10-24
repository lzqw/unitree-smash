#!/usr/bin/env python3
"""Reconstruct native arm targets and compare a proposed projection on fixed states."""
import argparse
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import sys
import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from quadsmash.control.full_racket_rally_controller_v4 import FullRacketRallyControllerV4
from quadsmash.control.full_racket_joint_preparation_v4 import FullRacketJointPreparation


def stats(values,mask):
    values=values[mask]
    return dict(count=len(values),minimum=float(values.min()),median=float(np.median(values)),maximum=float(values.max()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--counterfactual-checkpoint',type=Path,required=True)
    p.add_argument('--joint-contract',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Use a fresh output directory')
    torch.set_num_threads(2)
    warm=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    proposed=torch.load(a.counterfactual_checkpoint,map_location='cpu',weights_only=True)
    assert torch.equal(warm['parameters'],proposed['parameters'])
    assert {k:v for k,v in warm['seed'].items() if k!='joint_preparation_guard'}=={k:v for k,v in proposed['seed'].items() if k!='joint_preparation_guard'}
    assert not warm['seed'].get('causal_arm_prepose_gain',0.),'This reconstruction assumes no mutable prepose offsets'
    state=torch.load(a.snapshot/'partial_search_state.pt',map_location='cpu',weights_only=True)
    contract=json.loads(a.joint_contract.read_text())
    names=[f'{leg}_{joint}_joint' for joint in ['hip','thigh','calf'] for leg in ['FL','FR','RL','RR']]+[f'joint{i}' for i in range(1,7)]
    defaults=contract['defaults_before_neutral_arm_override']
    reset_path=ROOT/'configs/full_racket_reset_v4.json'
    defaults.update({f'joint{i+1}':q for i,q in enumerate(json.loads(reset_path.read_text())['arm_joint_positions_rad'])})
    default=torch.tensor([defaults[n] for n in names]);limits=torch.tensor([contract['limits'][n] for n in names])
    public_path=a.snapshot/'near_contact_all.partial.npz';control_path=a.snapshot/'control_window.partial.npz'
    with np.load(public_path) as saved:count=saved['observations'].shape[1]
    cases=count//len(state['parameters']);assert cases*len(state['parameters'])==count
    controller=FullRacketRallyControllerV4(warm,count,'cpu');controller.reset()
    controller.parameters.copy_(state['parameters'].repeat_interleave(cases,dim=0));controller.planned.fill_(True)
    guard=controller.teacher.joint_preparation;controller.teacher.joint_preparation=None
    alternative=FullRacketJointPreparation(proposed['seed']['joint_preparation_guard']).float()
    plan_path=a.snapshot/'causal_plans.partial.json'
    for log in json.loads(plan_path.read_text()):
        delta=np.asarray(log['position_m'])-controller.nominal_target
        for index,e in enumerate(log['env_ids']):
            event=log['calibrated_event_time_s'][index]
            if log['valid'][index] and np.isfinite(event) and 2.3<=event<=3.3:
                controller.event_time[e]=event
                controller.target_offset_world[e]=torch.tensor(delta[index],dtype=torch.float32)
    values=[];targets=[]
    with np.load(public_path) as saved,np.load(control_path) as control,torch.inference_mode():
        for i,public in enumerate(saved['observations']):
            obs=torch.tensor(public);valid=saved['valid_first_episode'][i]&(public[:,773]<.5)
            assert np.allclose(saved['episode_clock_s'][i,valid],control['clock_s'][2*i,valid],atol=1e-5)
            action=controller.action({'policy':obs},default,limits)
            original=limits[12:,0]+.5*(action[:,12:]+1)*(limits[12:,1]-limits[12:,0])
            native=limits[12:,0]+.5*(torch.tensor(control['actions'][2*i,:,12:])+1)*(limits[12:,1]-limits[12:,0])
            actual=default[12:]+obs[:,21:27];model=controller.teacher.kinematics
            baseline=guard(original,obs,limits[12:],model,actual)
            candidate=alternative(original,obs,limits[12:],model,actual)
            assert torch.isfinite(candidate).all() and (candidate>=limits[12:,0]).all() and (candidate<=limits[12:,1]).all()
            before=guard.support_state(original,obs,model)[0]
            commanded=guard.support_state(native,obs,model)[0]
            observed=guard.support_state(actual,obs,model,observed_center=True)[0]
            proposed_x=alternative.support_state(candidate,obs,model)[0]
            center,normal,*_=model(actual)
            position_error=torch.linalg.vector_norm(center-obs[:,45:48],dim=1)
            normal_error=torch.linalg.vector_norm(normal-obs[:,48:51],dim=1)
            root=saved['link_poses'][i,:,0]
            rotation=Rotation.from_quat(root[:,[4,5,6,3]]).as_matrix()
            known=root[:,:3]+np.einsum('nij,nj->ni',rotation,public[:,760:763])
            known_error=np.linalg.norm(known-np.array([1.1,0.,1.8]),axis=1)
            values.append(np.column_stack((torch.max(abs(baseline-native),dim=1).values.numpy(),
                torch.max(abs(native-original),dim=1).values.numpy(),before.numpy(),commanded.numpy(),
                observed.numpy(),proposed_x.numpy(),position_error.numpy(),normal_error.numpy(),known_error,valid)))
            targets.append(np.stack((original.numpy(),native.numpy(),candidate.numpy())))
    data=np.stack(values);valid=data[:,:,-1]>.5
    a.output.mkdir(parents=True);np.savez_compressed(a.output/'reconstructed_targets.npz',diagnostics=data,
        arm_joint_targets_rad=np.stack(targets),target_order=np.array(['unconstrained','native','counterfactual']))
    sources=[a.checkpoint,a.counterfactual_checkpoint,a.joint_contract,public_path,control_path,
        plan_path,a.snapshot/'partial_search_state.pt',reset_path,Path(__file__),ROOT/'quadsmash/control/full_racket_joint_preparation_v4.py']
    report=dict(created_at=datetime.now(timezone.utc).isoformat(),
        scope='Saved native states and original causal forecasts. Native command reconstruction and a counterfactual arm projection, without changed-state dynamics or new contact/recovery outcomes.',
        valid_unhit_samples=int(valid.sum()),cpu_native_command_error_rad=stats(data[:,:,0],valid),
        modified_native_command_samples=int(np.sum(valid&(data[:,:,1]>.001))),
        native_command_samples_at_original_projection_bound=int(np.sum(valid&(data[:,:,1]>.399))),
        unconstrained_target_support_x_m=stats(data[:,:,2],valid),native_target_support_x_m=stats(data[:,:,3],valid),
        observed_support_x_m=stats(data[:,:,4],valid),observed_minus_native_target_support_m=stats(data[:,:,4]-data[:,:,3],valid),
        measured_model_position_error_m=stats(data[:,:,6],valid),measured_model_normal_error=stats(data[:,:,7],valid),
        known_command_world_error_m=stats(data[:,:,8],valid),
        native_command_targets_beyond_net_plane=int(np.sum(valid&(data[:,:,3]>1.5))),
        counterfactual_command_targets_beyond_net_plane=int(np.sum(valid&(data[:,:,5]>1.5))),
        counterfactual_target_support_x_m=stats(data[:,:,5],valid),
        counterfactual_all_targets_within_original_physical_joint_limits=True,
        counterfactual_configuration=alternative.config,new_dynamics_executed=False,formal_acceptance=False,
        sources={str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources})
    (a.output/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
