#!/usr/bin/env python3
"""Fit the existing gate's temperature and racket intercept to recorded training transitions."""
import argparse
import json
from pathlib import Path

import torch

from fit_supermimic_mixed_body_rollout import digest
from quadsmash.rl.supermimic_evidence_binding_v1 import json_sha
from quadsmash.rl.supermimic_ball_residual_v1 import ball_geometry_features
from quadsmash.rl.supermimic_ball_residual_v2 import BallResidualV2, RacketSweepSupportedBallV2
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


def frozen_training_source(folder, task):
    manifest = json.loads((folder/'manifest.json').read_text())
    identity = json.loads((folder.parent/'loaded_runtime_identity.json').read_text())['identity']
    path = folder/'complete_episodes.pt'
    labels = torch.load(folder/'native_critic_labels.pt', map_location='cpu', weights_only=True, mmap=True)
    checksum = digest(path)
    if (identity['task_contract_sha256'] != json_sha(task) or identity['policy_execution'] != manifest['policy_execution']
            or manifest['actor_unchanged'] is not True or manifest['formal_evaluation'] is not False
            or checksum != labels['source_sha256']):
        raise ValueError(f'Incompatible frozen training collection in {folder}')
    archive = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    receipt = archive['receipt']
    if (any(receipt[k] != manifest[k] for k in receipt)
            or any(labels[k] != receipt[k] for k in ('policy_id','actor_sha256','policy_execution'))
            or archive['policy_id'] != receipt['policy_id']):
        raise ValueError(f'Frozen collection identities disagree in {folder}')
    suite = Path(task['suite']['path'])
    if not suite.is_absolute():
        suite = Path(__file__).resolve().parents[1]/suite
    if digest(suite) != task['suite']['sha256']:
        raise ValueError('Training case suite has changed')
    cases = json.loads(suite.read_text())['cases']
    episodes = json.loads((folder/'episodes_summary.json').read_text())
    data = archive['tensors']
    lengths = data['valid_lengths']
    if (len(cases) != len(lengths) or manifest['complete_episodes'] != len(lengths)
            or lengths.tolist() != receipt['lengths'] or not ((lengths > 0) & (lengths <= len(data['state']))).all()
            or sorted(row['env_id'] for row in episodes) != list(range(len(lengths)))):
        raise ValueError('Incomplete or duplicated frozen lane identities')
    for row in episodes:
        lane = row['env_id']
        if (row['case_id'] != cases[lane]['case_id'] or row['episode_index'] != receipt['episode_ids'][lane]
                or not (data['episode'][:lengths[lane],lane] == row['episode_index']).all()):
            raise ValueError('Frozen episode and fixed case identities disagree')
    return data, dict(path=str(path), sha256=checksum, policy_id=receipt['policy_id'],
                      actor_sha256=receipt['actor_sha256'], suite_sha256=task['suite']['sha256'])


def training_groups(data, lanes):
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
    if 'valid_lengths' in data:
        inside = torch.arange(len(hit))[:, None] < data['valid_lengths'][None, :]
        valid &= inside
        pre[:-1] &= inside[1:]
        post[1:] &= inside[:-1]
    return {k:valid & v for k,v in {'pre':pre,'edge':hit,'post':post,'ordinary':~(pre|hit|post)}.items()}


@torch.no_grad()
def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ('body','ball','packed','output'):
        cli.add_argument('--'+key,required=True,type=Path)
    cli.add_argument('--fit-racket-readout',action='store_true',
                     help='Fit the existing racket logit row on measured deviations, including difficult pre-contact frames')
    cli.add_argument('--fit-swept-support', action='store_true',
                     help='Calibrate a causal racket-only geometric support on the same training lanes')
    cli.add_argument('--frozen-refresh', action='append', default=[], type=Path,
                     help='Complete scored frozen training collection from the same case suite (repeatable)')
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
    sweep = RacketSweepSupportedBallV2().to(device).eval() if a.fit_swept_support else None
    gaps = {}
    lanes=meta['train_lanes']
    if not lanes or set(lanes)&set(meta['excluded_lanes']):
        raise ValueError('Invalid training lanes')
    sources=[(data,dict(path=str(a.packed),sha256=meta['packed_sha256'],source='initial'))]
    checksums={meta['packed_sha256']}
    for folder in a.frozen_refresh:
        tensors, info = frozen_training_source(folder,task)
        if info['sha256'] in checksums:
            raise ValueError('Duplicate calibration source')
        checksums.add(info['sha256'])
        sources.append((tensors,dict(info,source='frozen_refresh')))
    sets={}
    hidden={}
    generator=torch.Generator().manual_seed(20261121)
    collected={name:([],[],[]) for name in ('ordinary','edge','pre','post')}
    for data, info in sources:
        info['samples']={}
        for name, mask in training_groups(data,lanes).items():
            indices=mask.nonzero()
            if name=='ordinary':
                indices=indices[torch.randperm(len(indices),generator=generator)[:4096]]
            info['samples'][name]=len(indices)
            joined, encoded, gap_batches=collected[name]
            for batch in indices.split(256):
                if not len(batch):
                    continue
                t,l=batch.T
                state=data['state'][t,l].to(device)
                geometry=ball_geometry_features(state,world.geometry)
                if sweep is not None:
                    gap_batches.append(sweep.min_racket_gap(geometry))
                features=ball.normalized(geometry)
                representation=ball.gate.net[:-1](features)
                logits=ball.gate.net[-1](representation)
                encoded.append(representation)
                response=(ball.response(features).reshape(-1,4,6)*ball.residual_scale)[:,:,3:]
                _,free_v=world.flight(state[:,49:52],state[:,52:55])
                target=data['next_state'][t,l,52:55].to(device)-free_v
                joined.append((logits,response,target))
    for name,(joined,encoded,gap_batches) in collected.items():
        if not joined:
            raise ValueError('Missing measured training events in '+name)
        sets[name]=tuple(torch.cat(x) for x in zip(*joined))
        hidden[name]=torch.cat(encoded)
        if gap_batches:
            gaps[name] = torch.cat(gap_batches)
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
    support_fit = None
    if sweep is not None:
        # Score clearance and transition width only on the archived training episodes.
        probability = {name: last(h).softmax(-1) for name, h in hidden.items()}
        candidates = []
        for bias_mm in (-6., -4., -2., 0., 2., 4., 6.):
            for width_mm in (1., 2., 4., 8.):
                squared = {}
                for name, (_, responses, target) in sets.items():
                    p = sweep.probability_with_support(probability[name], gaps[name],
                                                       .001*bias_mm, .001*width_mm)
                    pred = (p[:,1:,None]*responses).sum(1)
                    squared[name] = (pred-target).square().mean(-1)
                rmse = {k:float(v.mean().sqrt()) for k,v in squared.items()}
                tail = squared['pre'].topk(max(1,len(squared['pre'])//5)).values.mean()
                candidates.append((cost(rmse) + .1*float(tail)/max(original['pre']**2,1e-10),
                                   bias_mm, width_mm, rmse))
        _, bias_mm, width_mm, errors = min(candidates, key=lambda row:row[0])
        state = sweep.state_dict()
        state.update(ball.state_dict())
        sweep.load_state_dict(state, strict=True)
        sweep.gap_bias_m.fill_(.001*bias_mm)
        sweep.width_m.fill_(.001*width_mm)
        support_fit = dict(gap_bias_m=float(sweep.gap_bias_m), width_m=float(sweep.width_m),
                           geometry_hashes=sweep.preview.geometry_hashes, train_candidates=len(candidates))
        ball = sweep
    info=dict(swept_racket_support=support_fit,racket_readout_fit=tail_fit,source_ball_sha256=digest(a.ball),fitted_body_sha256=digest(a.body),
              calibration_sources=[info for _,info in sources],unique_training_lanes=len(set(lanes)),
              packed_sha256=meta['packed_sha256'],train_lanes=lanes,excluded_lanes=meta['excluded_lanes'],
              samples={k:len(v[0]) for k,v in sets.items()},before_rmse_mps=original,after_rmse_mps=errors,
              temperature_inverse=slope,racket_logit_offset=bias,contact_bptt_admitted=False,actor_updates=0)
    a.output.mkdir(parents=True,exist_ok=False)
    torch.save(dict(checkpoint,schema=ball.schema,model=ball.cpu().state_dict(),event_calibration=info),a.output/'ball.pending')
    (a.output/'ball.pending').replace(a.output/'ball.pt')
    (a.output/'fit.json').write_text(json.dumps(dict(info,sha256=digest(a.output/'ball.pt')),indent=2)+'\n')


if __name__=='__main__':
    main()
