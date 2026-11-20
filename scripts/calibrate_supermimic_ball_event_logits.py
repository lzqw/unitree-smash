#!/usr/bin/env python3
"""Fit the existing gate's temperature and racket intercept to recorded training transitions."""
import argparse
import json
from pathlib import Path

import torch

from fit_supermimic_mixed_body_rollout import digest
from quadsmash.rl.supermimic_ball_residual_v1 import ball_geometry_features
from quadsmash.rl.supermimic_ball_residual_v2 import BallResidualV2
from quadsmash.rl.supermimic_physical_world_v2 import PhysicalWorldV2


@torch.enable_grad()
def fit_racket_readout(sets, hidden, slope, bias):
    """Calibrate one existing mode against the measured outgoing velocity in m/s."""
    adjustment = torch.zeros_like(hidden['pre'][0]).requires_grad_()
    opt = torch.optim.LBFGS([adjustment], max_iter=45, line_search_fn='strong_wolfe')
    cached = {}
    for name, (logits, responses, target) in sets.items():
        # Aggregate the unaffected four modes into their conditional expected response.
        scaled = slope*logits
        others = scaled[:, [0,2,3,4]]
        other_velocity = (others.softmax(-1)[:, 1:, None]*responses[:, 1:]).sum(1)
        cached[name] = (scaled[:,1]+bias-others.logsumexp(-1), other_velocity,
                        responses[:,0]-other_velocity, target)
    def errors():
        values = {}
        for name,(odds, base, change, target) in cached.items():
            p = (odds + hidden[name] @ adjustment).sigmoid()
            values[name] = (base+p[:,None]*change-target).square().mean(-1)
        return values
    def close():
        opt.zero_grad()
        err = errors()
        # Protect ordinary motion and true strikes alongside the pre-impact tail.
        pre = err['pre']
        loss = (pre.mean() + pre.topk(max(1,len(pre)//5)).values.mean()
                + err['edge'].mean() + err['ordinary'].mean() + err['post'].mean()
                + adjustment.square().sum())
        loss.backward()
        return loss
    opt.step(close)
    values = errors()
    if not all(torch.isfinite(e).all() for e in values.values()):
        raise ValueError('Nonfinite measured ball readout error')
    return adjustment.detach(), dict(trainable='existing racket gate output row',
        after_rmse_mps={k:float(e.detach().mean().sqrt()) for k,e in values.items()},
        row_change_l2=float(adjustment.detach().norm()))


@torch.no_grad()
def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ('body','ball','packed','output'):
        cli.add_argument('--'+key,required=True,type=Path)
    cli.add_argument('--fit-racket-readout',action='store_true',
                     help='Fit the existing racket logit row on measured deviations, including difficult pre-contact frames')
    a=cli.parse_args()
    torch.set_num_threads(3)
    body=torch.load(a.body,map_location='cpu',weights_only=True)
    meta=body['calibration']
    if meta['packed_sha256']!=digest(a.packed) or meta['ball_sha256']!=digest(a.ball):
        raise ValueError('Mixed fit calibration sources differ')
    archive=torch.load(a.packed,map_location='cpu',weights_only=True,mmap=True)
    task=body['source_manifest']['task_contract']
    data=archive['tensors']
    if archive['source_manifest']['task_contract']!=task:
        raise ValueError('Native archive contract differs')
    device=torch.device('cuda:0')
    world=PhysicalWorldV2(task['asset']['sha256']).to(device).eval()
    world.load_state_dict(body['model'])
    checkpoint=torch.load(a.ball,weights_only=True,map_location='cpu')
    ball=BallResidualV2().to(device).eval()
    ball.load_state_dict(checkpoint['model'])
    lanes=meta['train_lanes']
    if not lanes or set(lanes)&set(meta['excluded_lanes']):
        raise ValueError('Invalid training lanes')
    edge=data['next_event_state'][...,1]>data['event_state'][...,1]
    frame=data['next_event_state'][...,6]>data['event_state'][...,6]
    hit=edge|frame
    valid=data['present'].bool() & ~(data['terminated']|data['timeout'])
    eligible=torch.zeros_like(valid)
    eligible[:,lanes]=True
    valid &= eligible
    same=data['episode'][1:]==data['episode'][:-1]
    pre=torch.zeros_like(hit);pre[:-1]=same & hit[1:] & ~hit[:-1]
    post=torch.zeros_like(hit);post[1:]=same & hit[:-1] & ~hit[1:]
    groups={'pre':pre,'edge':hit,'post':post,'ordinary':~(pre|hit|post)}
    sets={}
    hidden={}
    generator=torch.Generator().manual_seed(20261121)
    for name,mask in groups.items():
        indices=(valid & mask).nonzero()
        if name=='ordinary':
            indices=indices[torch.randperm(len(indices),generator=generator)[:4096]]
        if not len(indices):raise ValueError('Missing measured training events in '+name)
        joined=[]
        encoded=[]
        for batch in indices.split(256):
            t,l=batch.T
            state=data['state'][t,l].to(device)
            features=ball.normalized(ball_geometry_features(state,world.geometry))
            representation=ball.gate.net[:-1](features)
            logits=ball.gate.net[-1](representation)
            encoded.append(representation)
            response=(ball.response(features).reshape(-1,4,6)*ball.residual_scale)[:,:,3:]
            _,free_v=world.flight(state[:,49:52],state[:,52:55])
            target=data['next_state'][t,l,52:55].to(device)-free_v
            joined.append((logits,response,target))
        sets[name]=tuple(torch.cat(x) for x in zip(*joined))
        hidden[name]=torch.cat(encoded)
    def scores(slope,bias):
        result={}
        for name,(logits,response,target) in sets.items():
            li=logits*slope
            li[:,1]+=bias
            predicted=(li.softmax(-1)[:,1:,None]*response).sum(1)
            result[name]=float((predicted-target).square().mean().sqrt())
        return result
    original=scores(1.,0.)
    # Balance true contact and adjacent ticks across orders of magnitude. The
    # unaffected ordinary and post-impact groups protect against false impulses.
    def cost(s):
        return sum(w*(s[k]/max(original[k],1e-6))**2 for k,w in
                   [('pre',.4),('edge',.4),('ordinary',.1),('post',.1)])
    best=(cost(original),1.,0.,original)
    for slope in (.5,.75,1.,1.25,1.5,2.,3.,4.):
        for bias in (-3.,-2.,-1.,-.5,0.,.5,1.,2.,3.):
            errors=scores(slope,bias)
            if cost(errors)<best[0]:
                best=(cost(errors),slope,bias,errors)
    _,slope,bias,errors=best
    last=ball.gate.net[-1]
    last.weight.mul_(slope);last.bias.mul_(slope);last.bias[1]+=bias
    tail_fit = None
    if a.fit_racket_readout:
        adjustment, stats = fit_racket_readout(sets, hidden, slope, bias)
        last.weight[1].add_(adjustment)
        errors = stats['after_rmse_mps']
        tail_fit = stats
    info=dict(racket_readout_fit=tail_fit,source_ball_sha256=digest(a.ball),fitted_body_sha256=digest(a.body),
              packed_sha256=meta['packed_sha256'],train_lanes=lanes,excluded_lanes=meta['excluded_lanes'],
              samples={k:len(v[0]) for k,v in sets.items()},before_rmse_mps=original,after_rmse_mps=errors,
              temperature_inverse=slope,racket_logit_offset=bias,contact_bptt_admitted=False,actor_updates=0)
    a.output.mkdir(parents=True,exist_ok=False)
    torch.save(dict(checkpoint,model=ball.cpu().state_dict(),event_calibration=info),a.output/'ball.pending')
    (a.output/'ball.pending').replace(a.output/'ball.pt')
    (a.output/'fit.json').write_text(json.dumps(dict(info,sha256=digest(a.output/'ball.pt')),indent=2)+'\n')


if __name__=='__main__':
    main()
