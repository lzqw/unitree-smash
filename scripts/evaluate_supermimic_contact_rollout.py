#!/usr/bin/env python3
"""64-control-tick contact transfer with frozen Body WM and recorded commands."""
import math
import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from calibrate_supermimic_pilot_oracle import ROOT, BallImpulsePriorV1, _quat_rotate, _quat_rotate_inverse, sliding_speed_change
from quadsmash.rl.supermimic_compound_racket import compound_step
from quadsmash.rl.supermimic_physical_dataset_v1 import decode_physical, valid_windows
from quadsmash.rl.supermimic_physical_imagination_adapter_v1 import matrix_quaternion
from quadsmash.rl.supermimic_physical_world_v2 import PhysicalWorldV2


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def racket(state, geometry):
    s = decode_physical(state)
    bodies = geometry(s['joint_position'], s['joint_velocity'], s['root_position_w'],
                      s['root_quaternion_w'], s['root_link_velocity_w'], s['root_angular_velocity_w'])
    i = geometry.ids['Racket']
    return (bodies['position'][:, i], matrix_quaternion(bodies['rotation'][:, i]),
            bodies['velocity'][:, i], bodies['angular_velocity'][:, i])


def tick(prior, ball, start, end, fitted, offset, previous_patch=None, *, measured_path=None,
         racket_material=None, initial_spin=None, event_trace=None, compound_racket=False, return_spin=False, local_mobility=None, shuttle_mass=None):
    p, v = ball[:, :3], ball[:, 3:]
    if local_mobility is not None and (not compound_racket or local_mobility.shape!=(len(p),6,6)
            or shuttle_mass is None or not 0<shuttle_mass):
        raise ValueError('A per-lane articulated racket and positive shuttle mass are required')
    if compound_racket and (racket_material is None or previous_patch is not None):
        raise ValueError('Compound rollout needs measured shape materials and supports multiple contacts per substep')
    if event_trace is not None:
        if event_trace:
            raise ValueError('An empty event trace is required for a new control interval')
        dims = (prior.steps, len(ball))
        event_trace.update(racket=torch.full(dims,-1,device=ball.device,dtype=torch.int64),
                           ground=torch.zeros(dims,device=ball.device,dtype=torch.bool),
                           net=torch.zeros(dims,device=ball.device,dtype=torch.bool))
    spin = torch.zeros_like(v) if compound_racket and initial_spin is None else initial_spin
    if spin is not None and spin.shape != v.shape:
        raise ValueError('Spin must specify the incoming three-vector for each shuttle')
    if racket_material is not None:
        n_shapes = len(prior.preview.racket_box_centers) + len(prior.preview.racket_round_centers)
        if racket_material.shape != (n_shapes, 3):
            raise ValueError('Expected static friction, dynamic friction and restitution for each shape')
    if compound_racket and event_trace is not None:
        event_trace['near_shape_count'] = torch.zeros(dims,dtype=torch.int32,device=p.device)
        event_trace['shape_regime'] = torch.zeros((*dims,n_shapes),dtype=torch.uint8,device=p.device)
    if measured_path is not None:
        center,q,linear,omega=measured_path
        if any(x.shape[:2]!=(prior.steps,len(ball)) for x in measured_path):
            raise ValueError('Native racket samples must start at each physics substep')
    else:
        c0, q0, v0, w0 = start
        c1, q1, v1, w1 = end
        u = torch.arange(prior.steps, dtype=p.dtype, device=p.device)[:, None, None] / prior.steps
        center = c0 + (3*u.square()-2*u**3)*(c1-c0) + prior.dt*((u**3-2*u.square()+u)*v0 + (u**3-u.square())*v1)
        linear = (6*u-6*u.square())*(c1-c0)/prior.dt + (3*u.square()-4*u+1)*v0 + (3*u.square()-2*u)*v1
        q1 = torch.where((q0*q1).sum(-1, keepdim=True) < 0, -q1, q1)
        q = F.normalize(q0 + u*(q1-q0), dim=-1)
        omega = w0 + u*(w1-w0)
    g = prior.preview
    racket_radius = torch.maximum((g.racket_box_centers.norm(dim=-1) + g.racket_box_halves.norm(dim=-1)).max(),
                                  (g.racket_round_centers.norm(dim=-1) + g.racket_round_halves.flatten()
                                   + g.racket_round_radii.flatten()).max()) + g.radius + offset
    net_lo = torch.minimum((g.net_centers[:, 0]-g.net_halves[:, 0]).min(),
                           (g.post_centers[:, 0]-g.post_radii.flatten()).min()) - g.radius - offset
    net_hi = torch.maximum((g.net_centers[:, 0]+g.net_halves[:, 0]).max(),
                           (g.post_centers[:, 0]+g.post_radii.flatten()).max()) + g.radius + offset
    hit = torch.zeros(len(p), dtype=torch.bool, device=p.device)
    spin_limit = fitted.get('pre_solve_spin_limit_rad_s')
    if spin_limit is not None and (not math.isfinite(spin_limit) or spin_limit<=0 or spin is None):
        raise ValueError('Pre-solve spin limiting requires a positive speed and a propagated spin')
    for i in range(prior.steps):
        if spin_limit is not None:
            spin = spin * (spin_limit/spin.norm(dim=-1,keepdim=True).clamp_min(1.e-12)).clamp(max=1.)
        close = (p-center[i]).norm(dim=-1) <= racket_radius
        delta = torch.zeros_like(v)
        inside = torch.zeros_like(hit)
        impact_displacement = None
        if compound_racket and close.any():
            lane = close.nonzero().flatten()
            load_trace = {} if event_trace is not None else None
            normal, tangent, next_spin, active, impact_time = compound_step(
                prior, p[lane], v[lane], spin[lane], center[i, lane], q[i, lane],
                linear[i, lane], omega[i, lane], racket_material, offset, fitted['normal_impulse_scale'],
                separation_gain=fitted.get('separation_velocity_gain', 0.),
                impact_geometry_fraction=fitted.get('impact_geometry_fraction', 0.), return_impact_time=True, load_trace=load_trace,
                local_mobility=local_mobility[lane] if local_mobility is not None else None,shuttle_mass=shuttle_mass,
                temporal_solver=fitted.get('temporal_solver'),patch_inertia=fitted.get('patch_inertia'))
            if event_trace is not None:
                regime = load_trace['regime']
                event_trace['shape_regime'][i,lane] = regime
                event_trace['near_shape_count'][i,lane] = load_trace['near_shape_count'].int()
                event_trace['racket'][i,lane] = (regime>0).sum(-1)-1
            if fitted.get('impact_geometry_fraction', 0.):
                impact_displacement = torch.zeros_like(p).index_copy(0,lane,impact_time[:, None]*(normal+tangent))
            delta = delta.index_copy(0, lane, normal + tangent)
            spin = spin.index_copy(0, lane, next_spin)
            hit[lane] |= active > 0
        elif close.any():
            lane = close.nonzero().flatten()
            local = _quat_rotate_inverse(q[i, lane], p[lane]-center[i, lane])
            gap = g.racket_gap(local)
            near = gap <= offset
            if near.any():
                lane = lane[near]
                local = local[near]
                nlocal, shape = prior.racket_surface(local)
                n = _quat_rotate(q[i, lane], nlocal)
                arm = _quat_rotate(q[i, lane], local - g.radius*nlocal)
                rel = v[lane] - linear[i, lane] - torch.linalg.cross(omega[i, lane], arm)
                vn = (rel*n).sum(-1)
                active = (vn < 0) & (gap[near] + prior.sub_dt*vn <= 0)
                if racket_material is None:
                    mu_static = mu_dynamic = fitted['friction']
                    restitution = prior.restitution
                else:
                    mu_static, mu_dynamic, restitution = racket_material[shape].unbind(-1)
                normal = -(1+restitution)*fitted['normal_impulse_scale']*vn.clamp(max=0)
                tan = rel - vn[:, None]*n
                shuttle_arm = -g.radius*n
                if spin is not None:
                    tan = tan + torch.linalg.cross(spin[lane], shuttle_arm)
                sticking = fitted.get('sticking_gain', 2/7)*tan.norm(dim=-1)
                friction = torch.where(sticking <= mu_static*normal, sticking, mu_dynamic*normal)
                delta[lane] = active[:, None]*(normal[:, None]*n - friction[:, None]*F.normalize(tan, dim=-1))
                if previous_patch is not None:
                    inside[lane] = gap[near] <= 0
                    slip = sliding_speed_change(gap[near], vn, tan, prior.sub_dt, mu_dynamic)
                    slip = torch.where((~previous_patch[lane]) & (slip*.0052/prior.sub_dt > .001), slip, 0.)
                    delta[lane] -= slip[:, None]*F.normalize(tan, dim=-1)
                if spin is not None:
                    spin = spin.index_add(0, lane, torch.linalg.cross(shuttle_arm, delta[lane]) / (.4*g.radius**2))
                if event_trace is not None:
                    loaded = delta[lane].norm(dim=-1) > 0.
                    event_trace['racket'][i,lane[loaded]] = shape[loaded]
                hit[lane] |= active
        if previous_patch is not None:
            previous_patch.copy_(inside)
        gap = p[:, 2]-g.radius
        ground = (gap <= offset) & (v[:, 2] < 0) & (gap + prior.sub_dt*v[:, 2] <= 0)
        if event_trace is not None:
            event_trace['ground'][i] = ground
        delta[ground, 2] -= (1+prior.restitution)*v[ground, 2]
        at_net = (p[:, 0] >= net_lo) & (p[:, 0] <= net_hi)
        if at_net.any():
            lane = at_net.nonzero().flatten()
            net_gap = g.net_gap(p[lane])
            near = net_gap <= offset
            if near.any():
                lane = lane[near]
                n, _ = prior.net_surface(p[lane])
                vn = (v[lane]*n).sum(-1)
                active = (vn < 0) & (net_gap[near] + prior.sub_dt*vn <= 0)
                if event_trace is not None:
                    event_trace['net'][i,lane] = active
                delta[lane] -= ((1+prior.restitution)*vn*active)[:, None]*n
        relative = v-prior.wind
        v = v + prior.sub_dt*(prior.gravity-prior.drag*relative.norm(dim=-1, keepdim=True)*relative) + delta
        p = p + prior.sub_dt*v
        if impact_displacement is not None:
            p = p - impact_displacement
    result = torch.cat((p, v), -1), hit
    return (*result, spin) if return_spin else result


@torch.inference_mode()
def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('packed', 'checkpoint', 'calibration', 'output'):
        cli.add_argument('--'+name, type=Path, required=True)
    candidate = cli.add_mutually_exclusive_group()
    candidate.add_argument('--body-candidate',type=Path,help='Mixed-rollout Body WM replayed on its parent split')
    candidate.add_argument('--online-checkpoint', type=Path,
                     help='Replay the online Body WM on the same cases; prior training overlap is possible')
    cli.add_argument('--start', type=int, required=True)
    cli.add_argument('--initial-slip', action='store_true', help='Apply sliding response once when entering a separating overlap')
    args = cli.parse_args()
    cfg = json.loads((args.checkpoint.parent/'config.json').read_text())
    source_hash = sha(args.packed)
    if source_hash != cfg['source_sha256']['packed']:
        raise ValueError('Packed evaluation archive differs from the declared whole-lane split source')
    lanes = torch.tensor(cfg['provenance']['validation_lanes'])
    if len(lanes.unique()) != len(lanes) or set(lanes.tolist()) & set(cfg['provenance']['train_lanes']):
        raise ValueError('Invalid Body WM holdout')
    pack = torch.load(args.packed, weights_only=True, mmap=True, map_location='cpu')
    d = pack['tensors']
    start = args.start
    if not valid_windows(d['episode'], d['terminated'], d['timeout'], 64)[start, lanes].all():
        raise ValueError('Window crosses an episode boundary')
    if not d['present'][start, lanes].all():
        raise ValueError('Window must start after launch')
    checkpoint = torch.load(args.checkpoint, weights_only=True, map_location='cpu')
    contract = checkpoint['source_manifest']['task_contract']
    base = PhysicalWorldV2(contract['asset']['sha256']).eval()
    weights = checkpoint['model']
    parent_hash = sha(args.checkpoint)
    if args.online_checkpoint:
        online = torch.load(args.online_checkpoint, weights_only=False, mmap=True, map_location='cpu')
        metadata = online['metadata']
        if (metadata['task_contract'] != contract or
                metadata['physical_imagination']['checkpoint_sha256']['physical'] != parent_hash):
            raise ValueError('Online body has different physical model or task ancestry')
        weights = {k.removeprefix('physical_model.'):v for k,v in online['world'].items()
                   if k.startswith('physical_model.')}
    elif args.body_candidate:
        fitted = torch.load(args.body_candidate,weights_only=True,map_location='cpu')
        cal = fitted['calibration']
        if (fitted['schema']!=base.schema or cal['source_sha256']!=parent_hash or
                cal['packed_sha256']!=source_hash or set(lanes.tolist()) & set(cal['train_lanes'])):
            raise ValueError('Different parent or overlapping mixed calibration/evaluation lanes')
        weights = fitted['model']
    base.load_state_dict(weights, strict=True)
    suite = ROOT/contract['suite']['path']
    if sha(suite) != contract['suite']['sha256']:
        raise ValueError('Launch suite identity mismatch')
    launch = torch.tensor([r['launch']['position_m'] + r['launch']['velocity_mps']
                           for r in json.loads(suite.read_text())['cases']])[lanes]
    prior = BallImpulsePriorV1().eval()
    frozen = json.loads(args.calibration.read_text())
    result = {}
    column = 0
    for field in pack['task_schema']['fields']:
        if field['path'] == 'rally/contact_count':
            break
        column += field['width']
    else:
        raise ValueError('Missing native racket contact counter')
    for mode in ('recorded_robot_endpoints', 'body_wm_one_tick', 'body_wm'):
        state = d['state'][start, lanes].clone()
        feet = d['feet'][start, lanes]
        present, clock = d['present'][start, lanes], d['elapsed'][start, lanes]
        observed, predicted, metrics, center_error = [], [], {}, []
        patch = torch.zeros_like(present, dtype=torch.bool) if args.initial_slip else None
        for k in range(64):
            t = start+k
            if mode == 'body_wm_one_tick':
                state = torch.cat((d['state'][t, lanes, :49], state[:, 49:]), -1)
                feet = d['feet'][t, lanes]
            if mode != 'recorded_robot_endpoints':
                robot, logits, _ = base(state, d['targets'][t, lanes], d['previous_action'][t, lanes],
                                        feet, present, clock, launch)
                feet = logits.sigmoid()
            else:
                robot = d['next_state'][t, lanes]
            endpoint = racket(robot, base.geometry)
            center_error.append((endpoint[0]-racket(d['next_state'][t, lanes], base.geometry)[0]).norm(dim=-1))
            ball, contact = tick(prior, state[:, 49:], racket(state, base.geometry), endpoint,
                                 frozen['fitted'], frozen['solver']['pair_contact_offset_m'], patch)
            state = torch.cat((robot[:, :49], ball), -1)
            if not torch.isfinite(state).all():
                raise FloatingPointError(f'Nonfinite rollout at tick {k+1}')
            if not ((d['next_valid'][t, lanes, column] == 1) & (d['valid'][t, lanes, column] == 1)).all():
                raise ValueError('Missing native contact count')
            observed.append(d['next_values'][t, lanes, column] > d['values'][t, lanes, column])
            predicted.append(contact)
            clock = clock+prior.dt
            if k+1 in (1, 8, 16, 32, 64):
                error = ball - d['next_state'][t, lanes, 49:]
                metrics[str(k+1)] = dict(p95_position_error_m=float(error[:, :3].norm(dim=-1).quantile(.95)),
                                         p95_velocity_error_mps=float(error[:, 3:].norm(dim=-1).quantile(.95)),
                                         max_ball_speed_mps=float(ball[:, 3:].norm(dim=-1).max()))
        measured = torch.stack(observed).any(0)
        inferred = torch.stack(predicted).any(0)
        hit_error = torch.stack(center_error)[torch.stack(observed)]
        result[mode] = dict(metrics=metrics, native_hit_lanes=int(measured.sum()),
                            p95_racket_center_error_at_native_hit_m=float(hit_error.quantile(.95)) if len(hit_error) else None,
                            recovered=int((inferred & measured).sum()), missed=int((~inferred & measured).sum()),
                            native_no_hit_lanes=int((~measured).sum()), false_positive_lanes=int((inferred & ~measured).sum()))
    report = dict(scope='64 control ticks autoregressive ball; recorded final joint commands; original whole-lane Body WM holdout',
                  start_tick=start, initial_slip=args.initial_slip, lanes=lanes.tolist(), results=result, checkpoint_sha256=sha(args.checkpoint),
                  packed_sha256=source_hash, calibration_sha256=sha(args.calibration), policy_training_admitted=False)
    if args.online_checkpoint:
        report.update(scope='Online Body WM on its parent model panel; online exposure possible, diagnostic only',
                      parent_checkpoint_sha256=parent_hash, checkpoint_sha256=sha(args.online_checkpoint))
    elif args.body_candidate:
        report.update(scope='Mixed body candidate compared on unchanged parent holdout with recorded native commands',
                      parent_checkpoint_sha256=parent_hash, checkpoint_sha256=sha(args.body_candidate))
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
