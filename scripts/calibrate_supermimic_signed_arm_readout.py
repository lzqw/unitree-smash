#!/usr/bin/env python3
"""Fit the existing arm position and velocity readout to natural motion and paired executed targets."""
import argparse
import json
from pathlib import Path

import torch

from fit_supermimic_mixed_body_rollout import digest
from quadsmash.rl.supermimic_physical_world_v2 import PhysicalWorldV2
from quadsmash.rl.supermimic_policy_execution_v1 import contract
from quadsmash.rl.supermimic_training_probe_replay import training_windows


def arm_rows(model, data, ticks, columns, launches):
    state = data['state'][ticks, columns]
    features = model.features(state, *(data[k][ticks, columns] for k in
                             ('targets', 'previous_action', 'feet', 'present', 'elapsed')), launches)
    hidden = model.body.net[:-1]((features - model.input_mean) / model.input_scale)
    layer = model.body.net[-1]
    # The physical ordering is leg12, arm6, for both q and qdot.
    arm = torch.arange(25, 31)
    rows = torch.cat((arm, arm + 18))
    velocity = state[:, arm + 18]
    prior = torch.cat((state[:, arm] + model.dt * velocity, velocity), dim=1)
    scale = model.delta_scale[rows]
    native = (data['next_state'][ticks, columns][:, rows] - prior - model.delta_mean[rows]) / scale
    error = native - layer(hidden)[:, rows]
    return torch.cat((hidden, torch.ones_like(hidden[:, :1])), dim=1), error, native


class Moments:
    def __init__(self, width, outputs):
        self.gram = torch.zeros(width, width, dtype=torch.float64)
        self.cross = torch.zeros(width, outputs, dtype=torch.float64)
        self.energy = torch.zeros(outputs, dtype=torch.float64)
        self.sse = torch.zeros(outputs, dtype=torch.float64)
        self.n = 0

    def add(self, x, errors, targets):
        x, e, y = (v.double() for v in (x, errors, targets))
        self.gram.addmm_(x.T, x)
        self.cross.addmm_(x.T, e)
        self.energy += y.square().sum(0)
        self.sse += e.square().sum(0)
        self.n += len(x)

    def rmse(self, update):
        sse = self.sse - 2 * (self.cross * update).sum(0) + (update * (self.gram @ update)).sum(0)
        return (sse.clamp_min(0) / self.n).sqrt()


@torch.no_grad()
def rollout_inputs(model,ball,window,launch):
    """Only the initial state and the observed command sequence enter the rollout."""
    from quadsmash.rl.supermimic_ball_sequence_v1 import ball_transition
    state,feet,present=(window[k][0] for k in ('state','feet','present'))
    states=[]; contacts=[]; presence=[]
    for t,target in enumerate(window['targets']):
        states.append(state)
        contacts.append(feet)
        presence.append(present)
        state,feet,present=ball_transition(model,ball,state,target,window['previous_action'][t],
                                          feet,present,window['elapsed'][t],launch)
    return dict(window,state=torch.stack(states),feet=torch.stack(contacts),present=torch.stack(presence))


def mix_absolute_moments(natural,excited):
    # 80/20 inside the 75% pointwise term gives 60% natural, 15% excited, 25% signed.
    for name in ('gram','cross','energy','sse'):
        moments=getattr(natural,name)
        moments.mul_(.8).add_(getattr(excited,name),alpha=.2*natural.n/excited.n)


def solve(natural, signed, ridge):
    # Normalize by the calibration-only per-channel signal power. A 1% RMS floor
    # keeps a nearly unexcited paired output from overwhelming natural motion.
    a = .75 / natural.n
    nat_energy = (natural.energy / natural.n).clamp_min(1.e-12)
    sig_energy = (signed.energy / signed.n).clamp_min(1.e-4 * nat_energy)
    b = .25 / signed.n * (nat_energy / sig_energy)
    lhs = a * natural.gram + ridge * torch.eye(natural.gram.shape[0], dtype=natural.gram.dtype)
    return torch.stack([torch.linalg.solve(lhs + w * signed.gram, a * n + w * s)
                        for w, n, s in zip(b, natural.cross.T, signed.cross.T, strict=True)], dim=1)


def integrated_arm_observations(model, data, ticks, columns, launches):
    features,error,actual=arm_rows(model,data,ticks,columns,launches)
    rows=[*range(25,31),*range(43,49)]
    mean,scale=model.delta_mean[rows],model.delta_scale[rows]
    prediction=(actual-error)*scale+mean
    observed=actual*scale+mean
    return torch.stack((prediction[:,:6],prediction[:,6:],observed[:,:6],observed[:,6:]),dim=1).double()


def fit_arm_integration(natural,signed,dt):
    """Calibrate a bounded within-tick quadrature and blend using native training rows."""
    def project(x,y,w,default):
        denom=(w*x.square()).sum(0)
        numer=(w*x*y).sum(0)
        return torch.where(denom>1e-12,numer/denom.clamp_min(1e-12),default).clamp(0.,1.)
    # Normalize each joint's signed error by training response power. Retain
    # the same 1% RMS floor and 75/25 natural/excitation balance as the arm fit.
    yn=natural[:,2].square().mean(0).clamp_min(1e-12)
    ys=signed[:,2].square().mean(0).clamp_min(yn*1e-4)
    weights=torch.cat((torch.ones_like(natural[:,2])*.75/(yn*len(natural)),
                       torch.ones_like(signed[:,2])*.25/(ys*len(signed))))
    predicted_q,predicted_dv,actual_q,actual_dv=torch.cat((natural,signed)).unbind(1)
    fraction=project(dt*actual_dv,actual_q,weights,.5)
    difference=dt*fraction*predicted_dv-predicted_q
    blend=project(difference,actual_q-predicted_q,weights,0.)
    return fraction,blend


@torch.no_grad()
def fold_arm_integration(model,fraction,blend):
    """Fold physical-unit position quadrature into the six existing normalized readout rows."""
    q=slice(25,31)
    v=slice(43,49)
    layer=model.body.net[-1]
    fraction,blend=fraction.to(layer.weight),blend.to(layer.weight)
    multiplier=blend*model.dt*fraction*model.delta_scale[v]/model.delta_scale[q]
    bias_shift=blend*(model.dt*fraction*model.delta_mean[v]-model.delta_mean[q])/model.delta_scale[q]
    layer.weight[q].mul_(1-blend[:,None]).add_(multiplier[:,None]*layer.weight[v])
    layer.bias[q].mul_(1-blend).add_(multiplier*layer.bias[v]+bias_shift)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for k in ('body', 'packed', 'training-collection', 'output'):
        parser.add_argument('--' + k, type=Path, required=True)
    parser.add_argument('--position-quadrature',action='store_true',help='Calibrate the physical relation between arm acceleration and displacement')
    parser.add_argument('--rollout-ball',type=Path,help='Fit absolute and signed arm corrections on free-running 16-step Body WM states')
    args = parser.parse_args()
    if args.position_quadrature and args.rollout_ball:
        parser.error('Select either quadrature or the free-running readout fit')
    torch.set_num_threads(4)
    ck = torch.load(args.body, map_location='cpu', weights_only=True)
    meta = ck['calibration']
    task = ck['source_manifest']['task_contract']
    packed = torch.load(args.packed, weights_only=True, mmap=True)
    if digest(args.packed) != meta['packed_sha256'] or packed['source_manifest']['task_contract'] != task:
        raise ValueError('Different natural archive or physics contract')
    suite = Path(task['suite']['path'])
    if digest(suite) != task['suite']['sha256']:
        raise ValueError('Different launch suite')
    launch = torch.tensor([c['launch']['position_m'] + c['launch']['velocity_mps']
                           for c in json.loads(suite.read_text())['cases']])
    lanes = set(meta['train_lanes'])
    if lanes & set(meta['excluded_lanes']):
        raise ValueError('Training lanes intersect the holdout')
    windows, receipt = training_windows(args.training_collection, task, contract(), set(range(len(launch))) - lanes)
    if receipt not in meta.get('sequence_receipts', [meta.get('sequence_receipt')]):
        raise ValueError('Paired data differ from the parent calibration')
    model = PhysicalWorldV2(task['asset']['sha256']).eval()
    model.load_state_dict(ck['model'])
    layer = model.body.net[-1]
    rows = list(range(25, 31)) + list(range(43, 49))
    rollout_ball=None
    if args.rollout_ball:
        from quadsmash.rl.supermimic_ball_residual_v2 import load_ball_residual
        ball_ck=torch.load(args.rollout_ball,map_location='cpu',weights_only=True)
        if (ball_ck.get('event_calibration',{}).get('fitted_body_sha256')!=digest(args.body) or
                ball_ck['event_calibration']['packed_sha256']!=meta['packed_sha256'] or
                ball_ck['event_calibration']['train_lanes']!=meta['train_lanes']):
            raise ValueError('Ball checkpoint does not bind the starting Body WM')
        rollout_ball=load_ball_residual(ball_ck).eval()
    position_data={'natural':[],'signed':[]}
    natural, signed = [Moments(layer.in_features + 1, len(rows)) for _ in range(2)]
    excited = Moments(layer.in_features+1,len(rows)) if rollout_ball is not None else None
    data = packed['tensors']
    eligible = torch.zeros(len(launch), dtype=torch.bool)
    eligible[sorted(lanes)] = True
    ticks, cols = torch.where(eligible & ~(data['terminated'] | data['timeout']))
    for t, c in zip(ticks.split(512), cols.split(512), strict=True):
        if args.position_quadrature:
            position_data['natural'].append(integrated_arm_observations(model,data,t,c,launch[c]))
        else:
            natural.add(*arm_rows(model, data, t, c, launch[c]))
    for ids, branches in windows:
        minus, plus = (branches[key] for key in ('minus', 'plus'))
        if rollout_ball is not None:
            minus,plus=(rollout_inputs(model,rollout_ball,d,launch[ids]) for d in (minus,plus))
        t, c = torch.where(torch.ones_like(minus['terminated'], dtype=torch.bool))
        for times, columns in zip(t.split(512), c.split(512), strict=True):
            launches = launch[torch.tensor(ids)[columns]]
            if args.position_quadrature:
                m,p=(integrated_arm_observations(model,d,times,columns,launches) for d in (minus,plus))
                position_data['signed'].append(p-m)
                continue
            m, p = (arm_rows(model, d, times, columns, launches) for d in (minus, plus))
            signed.add(*(y - x for y, x in zip(p, m, strict=True)))
            if excited is not None:
                excited.add(*m)
                excited.add(*p)
    if args.position_quadrature:
        data={k:torch.cat(v) for k,v in position_data.items()}
        fraction,blend=fit_arm_integration(data['natural'],data['signed'],model.dt)
        fold_arm_integration(model,fraction,blend)
        info=dict(parent_sha256=digest(args.body),calibration='arm-position-quadrature',
                  fraction=fraction.tolist(),blend=blend.tolist(),dt_s=model.dt,paired_weight=.25,
                  training_collection=receipt,actor_updates=0,critic_updates=0)
        for dataset,obs in data.items():
            raw,dv,actual,_=obs.unbind(1)
            revised=raw+blend*(model.dt*fraction*dv-raw)
            info[dataset]=dict(rows=len(raw),before_rmse_rad=(raw-actual).square().mean(0).sqrt().tolist(),
                               after_rmse_rad=(revised-actual).square().mean(0).sqrt().tolist())
    else:
        if excited is not None:
            mix_absolute_moments(natural,excited)
        ridge = .1
        step = solve(natural, signed, ridge)
        if not torch.isfinite(step).all():
            raise ValueError('Nonfinite readout')
        # Integer-list indexing returns a copy; assign explicitly and verify after serialization.
        layer.weight[rows] += step[:-1].T.to(layer.weight)
        layer.bias[rows] += step[-1].to(layer.bias)
        info = dict(parent_sha256=digest(args.body), natural_rows=natural.n, paired_rows=signed.n,
                    paired_weight=.25, ridge=ridge, readout_rows=rows, training_collection=receipt,
                    physical_units=['rad']*6+['rad/s']*6, actor_updates=0)
        if excited is not None:
            info.update(rollout_ball_sha256=digest(args.rollout_ball),excited_rows=excited.n,
                        source_weights=dict(natural=.60,excited_absolute=.15,signed=.25))
        for key, stats in [('absolute_mixture' if excited else 'natural', natural), ('paired', signed)]:
            scale = model.delta_scale[rows]
            info[key + '_rmse_before'] = (stats.rmse(torch.zeros_like(step)) * scale).tolist()
            info[key + '_rmse_after'] = (stats.rmse(step) * scale).tolist()
    output = args.output
    output.mkdir(exist_ok=False, parents=True)
    ck.update(model=model.state_dict(), calibration=dict(meta, signed_arm_readout=info), policy_training_admitted=False)
    pending = output / 'world.pending'
    torch.save(ck, pending)
    restored = PhysicalWorldV2(task['asset']['sha256']).eval()
    restored.load_state_dict(torch.load(pending, weights_only=True)['model'])
    torch.testing.assert_close(restored.body.net[-1].weight, layer.weight, rtol=0, atol=0)
    pending.replace(output / 'world.pt')
    (output / 'fit.json').write_text(json.dumps(dict(info, sha256=digest(output/'world.pt')), indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
