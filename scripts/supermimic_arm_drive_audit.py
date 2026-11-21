"""Fit two causal velocity corrections per arm joint on measured action pairs."""
import json
import torch
from fit_supermimic_mixed_body_rollout import event_pools
from fit_supermimic_paired_racket_center import validate


def fit_drive(features, errors):
    """Inputs are [tick, minus then plus, joint, feature] and matching errors."""
    low, high = features.double().chunk(2, dim=1)
    left, right = errors.double().chunk(2, dim=1)
    designs, labels = [], []
    for x, y in (((high+low)/2, (right+left)/2), (high-low, right-left)):
        scale = y.square().mean((0, 1)).sqrt().clamp_min(1e-6)
        designs.append((x / scale[None, None, :, None]).flatten(0, 1))
        labels.append((y / scale).flatten(0, 1))
    matrix = torch.cat(designs).transpose(0, 1)
    rhs = torch.cat(labels).T.unsqueeze(-1)
    # Small CPU SVD supports rank-deficient excitation, including an unexcited joint.
    solve = torch.linalg.lstsq(matrix.cpu(), rhs.cpu(), driver='gelsd')
    return solve.solution.squeeze(-1).to(features), solve.rank.tolist()


@torch.no_grad()
def audit_drive(args,world,ball,trainwin,trainlaunch,devwin,devlaunch,data,eligible,launch,identity):
    from supermimic_paired_body_candidate import fields, signed_rollout
    arm = slice(25,31)
    velocity = slice(43,49)

    def features(state, target):
        return torch.stack((target[...,12:18]-state[...,arm],state[...,velocity]),-1)

    measured=trainwin['state']
    estimates=[]
    for i in range(len(measured)):
        state,_,_=world(measured[i],trainwin['targets'][i],trainwin['previous_action'][i],trainwin['feet'][i],
                        trainwin['present'][i],trainwin['elapsed'][i],trainlaunch)
        estimates.append(state[:,velocity])
    coef,rank=fit_drive(features(measured,trainwin['targets']),trainwin['next_state'][...,velocity]-torch.stack(estimates))

    def corrected(module,inputs,output):
        prior, target=inputs[:2]
        state, logits, active=output
        dv=(features(prior,target)*coef).sum(-1)
        state=state.clone()
        state[:,velocity]+=dv
        state[:,arm]+=(.5*module.dt)*dv
        return state,logits,active

    pools=event_pools(data,~eligible,64,True)
    rng=torch.Generator().manual_seed(9222)
    windows={k:tuple(v[torch.randperm(len(v),generator=rng)[:24]].T) for k,v in pools.items()}
    if any(not len(t) for t,_ in windows.values()):
        raise ValueError('Missing long natural, hit or recovery windows')

    def score(win,launches):
        scores={}
        for h,state in enumerate(signed_rollout(world,ball,win,launches),1):
            if h not in (1,2,4,8,16):
                continue
            truth=win['next_state'][h-1]
            scores[h]={}
            for name,err in [('arm_velocity_rad_s',state[:,velocity]-truth[:,velocity]),
                            ('center_m',fields(state,world.geometry)[:,:3]-fields(truth,world.geometry)[:,:3])]:
                lo,hi=err.chunk(2)
                scores[h][name]=dict(midpoint_rmse=float(((hi+lo)/2).square().mean().sqrt()),
                                     signed_rmse=float((hi-lo).square().mean().sqrt()))
        return scores

    result=dict(identity=identity,train_pairs=len(trainlaunch)//2,development_pairs=len(devlaunch)//2,
                coefficients=coef.tolist(),rank=rank,actor_updates=0,critic_updates=0,
                final_holdout_opened=False,windows={k:torch.stack(v,1).tolist() for k,v in windows.items()})
    for name in ('baseline','calibrated'):
        hook=world.register_forward_hook(corrected) if name=='calibrated' else None
        try:
            result[name]=dict(training=score(trainwin,trainlaunch),development=score(devwin,devlaunch),
                              natural=validate(world,ball,data,windows,launch,args.device))
        finally:
            if hook is not None:
                hook.remove()
    args.output.mkdir(parents=True)
    (args.output/'arm_drive.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
