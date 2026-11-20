#!/usr/bin/env python3
"""Compare causal joint models on unseen native lanes under measured drive settings."""
import argparse
import json
from pathlib import Path

import torch

from quadsmash.rl.supermimic_actuator_identification import drive_features
from quadsmash.rl.supermimic_physical_dataset_v1 import valid_windows
from quadsmash.rl.supermimic_physical_observation_v1 import JOINT_NAMES
from scripts.evaluate_supermimic_fresh_wm_panel import digest


def basis(state, target, last, torque, age, profile, mode):
    q, qd = state[...,13:31], state[...,31:49]
    base = torch.stack((torch.ones_like(q), q.sin(), q.cos(), qd/profile['velocity_limit']), -1)
    current, features = drive_features(state, target, last, torque, age, profile)
    if mode == 'state_only':
        return base, torch.zeros_like(q)
    if mode == 'executed_target':
        return torch.cat((base,target.unsqueeze(-1)), -1), torch.zeros_like(q)
    return torch.cat((base,features), -1), current


def contraction(x, coefficients):
    return torch.einsum('...jf,jfo->...jo', x, coefficients)


def fit_model(data, ids, profile, mode, dt):
    state, target, last, torque, age, dv, next_torque = [x[ids] for x in data]
    x, demand = basis(state,target,last,torque,age,profile,mode)
    # Positive per-joint effective inverse inertia from the training transitions.
    # The residual accounts for gravity, offsets, damping and multi-body motion.
    energy = (dt*demand).square().sum(0)
    inertia_inv = ((dt*demand*dv).sum(0)/energy.clamp_min(1e-12)).clamp_min(0)
    y = torch.stack((dv-dt*demand*inertia_inv, next_torque/profile['effort_limit']),-1)
    mean=x.mean(0);mean[...,0]=0.
    scale=x.std(0,unbiased=False).clamp_min(1e-5);scale[...,0]=1.
    x=(x-mean)/scale
    xt=x.permute(1,2,0)
    gram=xt@x.transpose(0,1)/len(x)
    ridge=1e-3*torch.eye(gram.shape[-1],device=x.device)
    coef=torch.linalg.solve(gram+ridge,xt@y.transpose(0,1)/len(x))
    return mean,scale,coef,inertia_inv


def predict(state,target,previous,torque,age,profile,mode,weights,dt):
    x,demand=basis(state,target,previous,torque,age,profile,mode)
    mean,scale,coef,inertia_inv=weights
    y=contraction((x-mean)/scale,coef)
    q,qd=state[...,13:31],state[...,31:49]
    new_qd=qd+dt*demand*inertia_inv+y[...,0]
    new_q=q+.5*dt*(qd+new_qd)
    return new_q,new_qd,y[...,1].clamp(-1,1)*profile['effort_limit']


@torch.no_grad()
def main():
    cli=argparse.ArgumentParser(description=__doc__)
    for k in ('packed','calibration-config','drive-contract','validation-split','output'):
        cli.add_argument('--'+k,required=True,type=Path)
    cli.add_argument('--weights-output', type=Path,
                     help='Save train-only PD coefficients for subsequent paired-command validation')
    a=cli.parse_args();torch.set_num_threads(3)
    config=json.loads(a.calibration_config.read_text()); drive=json.loads(a.drive_contract.read_text())
    if (drive['schema']!='native-joint-drive-v1' or drive['joint_names']!=list(JOINT_NAMES)
            or len(drive['profiles'])!=1 or set(drive['lane_profile'])!={0}):
        raise ValueError('This calibration requires a single measured native actuator profile')
    if digest(a.packed)!=config['packed_sha256']:raise ValueError('Packed native source changed')
    package=torch.load(a.packed,weights_only=True,mmap=True,map_location='cpu')
    if drive['asset_sha256']!=package['source_manifest']['task_contract']['asset']['sha256']:
        raise ValueError('Different robot asset')
    data=package['tensors']
    dt=drive['control_dt_s']; params=drive['profiles'][0]
    profile={k:torch.tensor(v,dtype=torch.float64) for k,v in params.items() if k!='position_limits'}
    if (profile['effort_limit']<=0).any() or (profile['velocity_limit']<=0).any():
        raise ValueError('Need positive measured limits')
    if len(drive['all_joint_names'])!=data['torque'].shape[-1]:raise ValueError('Torque column mismatch')
    torque=data['torque'][...,drive['joint_indices']].double()
    target=data['targets'].double();states=data['state'].double();nxt=data['next_state'].double()
    tr=config['train_lanes']; va=json.loads(a.validation_split.read_text())['validation_lanes']
    if not tr or not va or set(tr)&set(va):raise ValueError('Invalid native lane split')
    same=(data['episode'][1:]==data['episode'][:-1]) & ~(data['terminated']|data['timeout'])[:-1]
    age=torch.zeros_like(target)
    for t in range(1,len(target)):
        held=same[t-1,:,None] & ((target[t]-target[t-1]).abs()<=1e-6)
        age[t]=torch.where(held,age[t-1]+1,0.)
    active=torch.zeros_like(same);active[:,tr]=True
    train=(active&same).nonzero().T
    stage=tuple(train)
    velocity_change=nxt[1:,:,31:49]-states[1:,:,31:49]
    arrays=[states[1:],target[1:],target[:-1],torque[:-1],age[1:],velocity_change,torque[1:]]
    valid=valid_windows(data['episode'],data['terminated'],data['timeout'],64)
    valid[0]=False;valid[1:] &= same[:len(valid)-1]
    windows=[]
    for lane in va:
        ticks=torch.where(valid[:,lane])[0]
        if not len(ticks):raise ValueError('Held lane has no complete 64-step command window')
        windows.extend((int(t),lane) for t in ticks[torch.linspace(0,len(ticks)-1,10).long()])
    tick,lane=torch.tensor(windows).T
    result={}
    for mode in ('state_only','executed_target','pd_history'):
        weights=fit_model(arrays,stage,profile,mode,dt)
        if mode == 'pd_history':
            pd_weights=weights
        state=states[tick,lane].clone();prev=torque[tick-1,lane]
        metrics={}
        for i in range(64):
            ti=tick+i
            q,qd,pred_torque=predict(state,target[ti,lane],target[ti-1,lane],prev,age[ti,lane],
                                      profile,mode,weights,dt)
            if i+1 in (1,2,4,8,16,32,64):
                truth=nxt[ti,lane]
                metrics[i+1]=dict(arm_position_mae_rad=float((q[:,12:]-truth[:,25:31]).abs().mean()),
                    arm_velocity_mae_rad_s=float((qd[:,12:]-truth[:,43:49]).abs().mean()),
                    all_joint_position_mae_rad=float((q-truth[:,13:31]).abs().mean()),
                    all_joint_velocity_mae_rad_s=float((qd-truth[:,31:49]).abs().mean()),
                    torque_estimate_mae_nm=float((pred_torque-torque[ti,lane]).abs().mean()))
            state[:,13:31]=q;state[:,31:49]=qd;prev=pred_torque
        result[mode]=dict(horizon=metrics,inverse_inertia=weights[-1].tolist())
    payload=dict(scope='isolated joint identification; fixed executed command sequences; only previous native torque at initialization, predicted torque thereafter',
        measured_actuator_types=drive['actuator_types'],train_lanes=tr,held_lanes=va,train_transitions=train.shape[1],
        validation_windows=len(windows),drive_sha256=digest(a.drive_contract),packed_sha256=config['packed_sha256'],
        results=result,actor_updates=0,contact_bptt_admitted=False)
    if a.weights_output:
        with a.weights_output.open('xb') as stream:
            torch.save(dict(schema='native-joint-pd-history-v1', mode='pd_history', control_dt_s=dt,
                            train_lanes=tr, drive_sha256=payload['drive_sha256'],
                            packed_sha256=config['packed_sha256'], coefficients=pd_weights,
                            task_contract=package['source_manifest']['task_contract']), stream)
    with a.output.open('x') as f:json.dump(payload,f,indent=2,allow_nan=False)


if __name__=='__main__':
    main()
