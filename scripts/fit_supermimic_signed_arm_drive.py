#!/usr/bin/env python3
"""Update an existing measured PD readout using natural transitions and signed identification windows."""
import argparse
import json
from pathlib import Path

import torch

from audit_supermimic_actuator_prior import basis
from evaluate_supermimic_fresh_wm_panel import digest
from quadsmash.rl.supermimic_policy_execution_v1 import contract
from quadsmash.rl.supermimic_training_probe_replay import training_windows


def regressors(data, indices, profile, weights, dt, torque_indices):
    target=data['targets'].double()
    state=data['state']
    torque=data['torque'][...,torque_indices].double()
    same = (data['episode'][1:]==data['episode'][:-1]) & ~(data['terminated']|data['timeout'])[:-1]
    active=torch.zeros_like(same);active[:,indices]=True
    age=torch.zeros_like(target)
    for t in range(1,len(target)):
        age[t]=torch.where(same[t-1,:,None]&((target[t]-target[t-1]).abs()<=1e-6),age[t-1]+1,0.)
    t,l=(same & active).nonzero().T
    t=t+1
    x,u=basis(state[t,l].double(),target[t,l],target[t-1,l],torque[t-1,l],age[t,l],profile,'pd_history')
    mean,scale,_,inverse=weights
    dv=(data['next_state'][t,l,31:49]-state[t,l,31:49]).double()
    y=torch.stack((dv-dt*u*inverse, torque[t,l]/profile['effort_limit']),-1)
    return (x-mean)/scale, y


def signed_rows(window, profile, weights, dt, torque_indices):
    target=window['targets'].double()
    torque=window['applied_joint_torque'][...,torque_indices].double()
    state=window['state'].double()
    dv=window['next_state'][...,31:49].double()-state[...,31:49]
    mean,scale,_,inverse=weights
    age=torch.zeros_like(target[0])
    xs=[];ys=[]
    # The first tick lacks the previous target age. Both branches start from
    # the same prefix; the first measured target and torque seed the next step.
    for t in range(1,len(target)):
        age=torch.where((target[t]-target[t-1]).abs()<=1e-6,age+1,0.)
        x,u=basis(state[t],target[t],target[t-1],torque[t-1],age,profile,'pd_history')
        xs.append((x-mean)/scale)
        ys.append(torch.stack((dv[t]-dt*u*inverse,torque[t]/profile['effort_limit']),-1))
    return torch.cat(xs),torch.cat(ys)


def solve_update(x,y,dx,dy,coefficients):
    """75% natural and 25% signed loss, both normalized in their observed output units."""
    result=coefficients.clone()
    for joint in range(12,18):
        a, b=x[:,joint],dx[:,joint]
        for channel in range(y.shape[-1]):
            baseline=coefficients[joint,:,channel]
            ya,yb=y[:,joint,channel],dy[:,joint,channel]
            natural=ya.square().mean().clamp_min(1.e-10)
            signed=yb.square().mean().clamp_min(.001*natural)
            wa=.75/(len(a)*natural)
            wb=.25/(len(b)*signed)
            lhs=wa*(a.T@a)+wb*(b.T@b)+.01/natural*torch.eye(a.shape[1],dtype=a.dtype)
            rhs=wa*(a.T@(ya-a@baseline))+wb*(b.T@(yb-b@baseline))
            result[joint,:,channel]=baseline+torch.linalg.solve(lhs,rhs)
    if not result.isfinite().all():raise ValueError('Nonfinite signed actuator update')
    return result


def residual_error(x,y,c):
    return ((torch.einsum('njf,jfo->njo',x,c)-y)[:,12:].square().mean((0,1)).sqrt().tolist())


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('packed','actuator','drive','output'):
        ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--collection',action='append',type=Path,required=True)
    a=ap.parse_args();torch.set_num_threads(2)
    fit=torch.load(a.actuator,weights_only=True,mmap=True)
    drive=json.loads(a.drive.read_text())
    if (fit['mode']!='pd_history' or fit['drive_sha256']!=digest(a.drive)
            or fit['packed_sha256']!=digest(a.packed) or len(set(map(Path.resolve,a.collection)))!=len(a.collection)):
        raise ValueError('Mismatched actuator, natural archive or repeated collection')
    data=torch.load(a.packed,weights_only=True,mmap=True)
    if data['source_manifest']['task_contract']!=fit['task_contract']:
        raise ValueError('Mismatched physics')
    weights=fit['coefficients'];dt=fit['control_dt_s'];lanes=fit['train_lanes']
    profile={k:torch.tensor(v,dtype=torch.float64) for k,v in drive['profiles'][0].items() if k!='position_limits'}
    joint=drive['joint_indices']
    x,y=regressors(data['tensors'],lanes,profile,weights,dt,joint)
    differences=[];targets=[];receipts=[]
    for collection in a.collection:
        windows,receipt=training_windows(collection,fit['task_contract'],contract(),set(range(128))-set(lanes))
        for _,pairs in windows:
            branches=[signed_rows(pairs[b],profile,weights,dt,joint) for b in ('minus','plus')]
            differences.append(branches[1][0]-branches[0][0]);targets.append(branches[1][1]-branches[0][1])
        receipts.append(receipt)
    dx,dy=torch.cat(differences),torch.cat(targets)
    coef=solve_update(x,y,dx,dy,weights[2])
    calibration=dict(actuator_parent_sha256=digest(a.actuator),collections=receipts,natural_count=len(x),
                     signed_count=len(dx),train_lanes=lanes,natural_weight=.75,signed_weight=.25,
                     output_units=['rad/s','fraction of effort limit'],first_tick_used=False,
                     natural_before=residual_error(x,y,weights[2]),natural_after=residual_error(x,y,coef),
                     signed_before=residual_error(dx,dy,weights[2]),signed_after=residual_error(dx,dy,coef))
    with a.output.open('xb') as out:
        torch.save(dict(fit,coefficients=(weights[0],weights[1],coef,weights[3]),signed_calibration=calibration),out)
    with a.output.with_suffix('.json').open('x') as out:
        json.dump(dict(calibration,sha256=digest(a.output),actor_updates=0,contact_gradients_admitted=False),out,indent=2,allow_nan=False)


if __name__=='__main__':main()
