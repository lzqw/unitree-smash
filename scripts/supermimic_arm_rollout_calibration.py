"""Local system identification for a six-joint readout on complete recorded command windows."""
import math
import torch

from fit_supermimic_mixed_body_rollout import event_pools
from fit_supermimic_paired_racket_center import take_windows


def calibration_objective(blocks, base):
    """Each source has a fixed objective weight, independent of its row count and units."""
    weights={'paired':.25,'natural':.5,'racket':.1,'recovery':.15}
    result=[]
    for key,value in blocks.items():
        source=key[0]
        terms=sum(k[0]==source for k in blocks)
        denom=base[key].square().mean().sqrt().clamp_min(1e-6)
        result.append(value.flatten() * (math.sqrt(weights[source]/(terms*value.numel()))/denom))
    return torch.cat(result)


def fit_trajectories(world,ball,trainwin,trainlaunch,data,eligible,launch,scale,make_hook):
    from supermimic_paired_body_candidate import fields, signed_rollout
    rng=torch.Generator().manual_seed(6438)
    pools=event_pools(data,eligible,64,True)
    panels={}
    indices={}
    for name,rows in pools.items():
        selected=rows[torch.randperm(len(rows),generator=rng)[:24]]
        if not len(selected):
            raise ValueError(f'No training windows for {name}')
        win=take_windows(data,*selected.T,64)
        panels[name]=({k:v.to(scale.device) for k,v in win.items()},launch[selected[:,1]].to(scale.device))
        indices[name]=selected.tolist()
    panels['paired']=(trainwin,trainlaunch)

    def errors(coef):
        hook=world.register_forward_hook(make_hook(coef))
        blocks={}
        try:
            for name,(win,launches) in panels.items():
                paired=name=='paired'
                horizons=(1,2,4,8,16) if paired else (4,16,32,64)
                for h,state in enumerate(signed_rollout(world,ball,win,launches),1):
                    if h not in horizons:
                        continue
                    truth=win['next_state'][h-1]
                    fk=fields(state,world.geometry)-fields(truth,world.geometry)
                    metrics={'center':fk[:,:3],'normal':fk[:,3:6],'linear':fk[:,6:9],'angular':fk[:,9:12],
                             'arm_speed':state[:,43:49]-truth[:,43:49]}
                    if not paired:
                        metrics['gyro']=state[:,10:13]-truth[:,10:13]
                        metrics['body']=state[:,:49]-truth[:,:49]
                    for metric,error in metrics.items():
                        if paired:
                            lo,hi=error.chunk(2)
                            blocks[name,h,metric,'midpoint']=(hi+lo)*.5
                            blocks[name,h,metric,'signed']=hi-lo
                        else:
                            blocks[name,h,metric]=error
        finally:
            hook.remove()
        return blocks

    zeros=torch.zeros_like(scale)
    base=errors(zeros)
    initial=calibration_objective(base,base)
    columns=[]
    increment=0.05
    for axis in torch.eye(scale.numel(),device=scale.device).reshape(-1,*scale.shape):
        offset=axis*scale*increment
        plus=calibration_objective(errors(offset),base)
        minus=calibration_objective(errors(-offset),base)
        columns.append((plus-minus)/(2*increment))
    jac=torch.stack(columns,dim=1).double()
    ridge=.01*torch.eye(len(columns),dtype=jac.dtype,device=jac.device)
    delta=torch.linalg.solve(jac.T@jac+ridge,-(jac.T@initial.double())).reshape_as(scale).to(scale)
    delta=delta/delta.abs().max().clamp_min(1.)
    best_value=float(initial.square().sum())
    best=zeros
    checks=[]
    for fraction in (1.,.5,.25):
        coef=delta*scale*fraction
        error=errors(coef)
        value=float(calibration_objective(error,base).square().sum())
        checks.append(dict(fraction=fraction,training_objective=value))
        if value<best_value:
            best=coef
            best_value=value
    return best,dict(start=float(initial.square().sum()),selected=best_value,trial_scores=checks,
                     training_windows=indices,paired_cases=len(trainlaunch)//2,coefficients=best.tolist(),
                     regularization=.01,derivative_step=increment)
