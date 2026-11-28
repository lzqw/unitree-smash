"""Fit independent force members on native post-release rollouts and score H32 touchdown."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from quadsmash.rl.dynamimic_lagrangian_v1 import rotation
from quadsmash.rl.dynamimic_recovery_wm_v1 import DynaMimicRecoveryWorldModel, local_robot_state
from quadsmash.rl.supermimic_full_body_geometry_v1 import FullBodyGeometryV1


def read_data(root, extra_fields=(), *, expected_schema='dynamimic-native-recovery-v1'):
    dest = root/'trajectories'
    manifest = json.loads((dest/'manifest.json').read_text())
    if manifest['schema'] != expected_schema:
        raise ValueError('Unexpected trajectory schema')
    names = ('before','after','gate','live','body','landed','feet','final_joint_targets', *extra_fields)
    blocks = {key: [] for key in names}
    for name in manifest['chunks']:
        chunk = torch.load(dest/name, weights_only=True, mmap=True)
        for key in names:
            blocks[key].append(chunk[key])
    data = {key:torch.cat(value) for key, value in blocks.items()}
    origins = torch.tensor(manifest['origins'])
    for key in ('before','after'):
        data[key][..., :3] -= origins
        data[key][..., 49:52] -= origins
    return data, manifest


def verify_replay_cases(archives):
    """Keep repeated cases and launches within their original train/dev/holdout partition."""
    known_ids, known_launches = {}, {}
    for cases in archives:
        if len(cases) != 256:
            raise ValueError('Each native archive requires 256 lanes')
        for lane, case in enumerate(cases):
            slot = lane % 16
            split = 'holdout' if slot < 2 else 'development' if slot < 4 else 'training'
            identifier = case['case_id']
            launch = json.dumps(case['launch'], sort_keys=True)
            if known_ids.setdefault(identifier, (launch, split)) != (launch, split):
                raise ValueError('Historical case changed launch or partition')
            if known_launches.setdefault(launch, split) != split:
                raise ValueError('Historical launch crossed native partitions')


def sampling_pools(data, held, horizon):
    eligible = data['live'] & data['gate']
    usable = eligible.unfold(0, horizon, 1).all(-1)
    landed = data['landed'].bool()
    # At least one sample on each side of the rising edge of the touchdown latch.
    crossing = usable & ~landed[:len(usable)] & landed[horizon-1:]
    return (usable & ~held).nonzero(), (crossing & ~held).nonzero(), (crossing & held).nonzero()


def crossing_panel(rows, lane_mask, latest_length):
    rows = rows[(rows[:, 0] + 32 <= latest_length) & lane_mask[rows[:, 1]]]
    lanes = rows[:, 1].unique()
    if not len(lanes):
        raise ValueError('No independent crossing windows for panel')
    selected = []
    for lane in lanes:
        windows = rows[rows[:, 1] == lane]
        selected.append(windows[len(windows)//2])
    return torch.stack(selected)


def crossing_metrics(sequence):
    last = sequence[-1]
    return dict(tilt_deg=float(last[:, 0].median()), com_height_mae_m=float(last[:, 1].mean()),
                contact_accuracy=float(sequence[..., 2].mean()), joint_speed_mae=float(last[:, 3].mean()))


def center_z(geometry, q, v):
    rot = rotation(q[:, 3:7])
    return geometry(q[:, 7:], v[:, 6:], q[:, :3], q[:, 3:7],
                    (rot @ v[:, :3,None]).squeeze(-1),
                    (rot @ v[:, 3:6,None]).squeeze(-1))['center'][:, 2]


def rollout(wm, member, data, rows, horizon, device, profiles, geometry=None):
    tick, lane = rows.T
    times = tick[None] + torch.arange(horizon)[:, None]
    q, v = local_robot_state(data['before'][tick, lane].to(device))
    targets = data['final_joint_targets'][times, lane].to(device)
    truth = data['after'][times, lane].to(device)
    feet = data['feet'][times, lane].to(device).float()
    ground = data['body'][times, lane].to(device).float()
    profile = profiles[lane].to(device)
    loss = q.new_zeros(())
    results = []
    for t in range(horizon):
        if q.is_cuda and horizon >= 16 and torch.is_grad_enabled():
            q, v, aux = checkpoint(wm, q, v, targets[t], member, profile,
                                   use_reentrant=False, preserve_rng_state=False)
        else:
            q, v, aux = wm(q, v, targets[t], member, profile)
        q_ref, v_ref = local_robot_state(truth[t])
        r, r_ref = rotation(q[:, 3:7]), rotation(q_ref[:, 3:7])
        p = aux['foot_contact'].clamp(1e-6, 1-1e-6)
        step = (20*F.smooth_l1_loss(q[:, :3], q_ref[:, :3]) +
                5*F.mse_loss(r, r_ref) + F.smooth_l1_loss(v[:, :6], v_ref[:, :6]) +
                5*F.smooth_l1_loss(q[:, 7:], q_ref[:, 7:]) +
                F.smooth_l1_loss(v[:, 6:], v_ref[:, 6:]) +
                F.binary_cross_entropy(p, feet[t]) +
                F.binary_cross_entropy_with_logits(aux['body_ground_logit'], ground[t]) +
                .001*aux['residual_force'].square().mean())
        loss = loss + step/horizon
        if geometry is not None:
            # Angle between gravity directions is invariant to yaw and bounds tilt error.
            tilt = torch.rad2deg(torch.acos((r[:, 2]*r_ref[:, 2]).sum(-1).clamp(-1,1)))
            results.append(torch.stack((tilt, (center_z(geometry,q,v)-center_z(geometry,q_ref,v_ref)).abs(),
                ((p >= .5) == feet[t].bool()).float().mean(-1),
                (v[:, 6:]-v_ref[:, 6:]).abs().mean(-1)), -1))
    return loss, results


def paired_response_loss(wm, members, baseline, other, starts, lanes, horizon, device, profiles, *, impact=False, ground_geometry=None):
    """Compare both executed command sequences from identical native starting states."""
    times = starts[lanes][None] + torch.arange(horizon)[:, None]
    q0, v0 = local_robot_state(baseline['before'][starts[lanes], lanes].to(device))
    def outputs(q, v):
        body = (rotation(q[:,3:7])[:,2], v[:,3:6], q[:,7:])
        values = (*body, q[:,2:3], v[:,:3]) if impact else body
        if ground_geometry is not None:
            shapes, geometry = ground_geometry
            values = (*values, shapes.monitored_gaps(geometry, q, v))
        return values

    states = []
    truth = []
    for branch in (baseline, other):
        q, v = q0, v0
        commands = branch['final_joint_targets'][times, lanes].to(device)
        profile = profiles[lanes].to(device)
        for command in commands:
            q, v, _ = checkpoint(wm, q, v, command, members, profile,
                                 use_reentrant=False, preserve_rng_state=False)
        states.append(outputs(q,v))
        q, v = local_robot_state(branch['after'][times[-1], lanes].to(device))
        truth.append(outputs(q,v))
    losses = []
    floors = (.01,.1,.01,.001,.02) if impact else (.01,.1,.01)
    if ground_geometry is not None:
        floors = (*floors, .001)
    for a,b,x,y,floor in zip(*states, *truth, floors, strict=True):
        actual = y-x
        scale = actual.square().mean().sqrt().clamp_min(floor)
        losses.append(F.smooth_l1_loss((b-a)/scale, actual/scale))
    loss = sum(losses)/len(losses)
    if ground_geometry is not None:
        # Native FK constrains the common offset that cancels from the branch difference.
        for predicted, measured in zip(states, truth, strict=True):
            loss = loss + .125 * F.smooth_l1_loss(predicted[-1]/.01, measured[-1]/.01)
    return loss


def trunk_innovation(model, data, end_tick, lane, profile, member, steps=4):
    """World-frame change over the final segment from this branch's measured state."""
    start = end_tick - steps
    before = data['before'][start, lane].to(profile.device)
    q,v = local_robot_state(before)
    for offset in range(steps):
        command = data['final_joint_targets'][start+offset,lane].to(profile.device)
        q,v,_ = model(q,v,command,member,profile)
    rot = rotation(q[:,3:7])
    world = torch.cat([(rot @ vel.unsqueeze(-1)).squeeze(-1) for vel in (v[:,:3],v[:,3:6])], -1)
    return world - before[:,7:13], data['after'][end_tick-1,lane,7:13].to(profile.device) - before[:,7:13]


def innovation_loss(wm, member, baseline, paired, end_tick, lanes, profile):
    left, right = [trunk_innovation(wm, d, end_tick, lanes, profile, member) for d in (baseline, paired)]
    predicted, measured = right[0]-left[0], right[1]-left[1]
    losses = []
    for p, y, floor in zip(predicted.split(3, dim=-1), measured.split(3, dim=-1), (.005, .025)):
        scale = y.square().mean().sqrt().clamp_min(floor)
        losses.append(F.smooth_l1_loss(p/scale, y/scale))
    return torch.stack(losses).mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('native','asset','output'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--steps-per-horizon', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--history-native', type=Path, action='append', default=[])
    parser.add_argument('--paired-native', type=Path)
    parser.add_argument('--pulse-audit', type=Path,
                        help='Use exact-prefix H4/H8/H16 native intervention pairs and development response selection')
    parser.add_argument('--fit-ground-response', action='store_true',
                        help='Fit native absolute and paired minimum and rear thigh surface gaps')
    parser.add_argument('--fit-trunk-innovation', action='store_true',
                        help='Also fit four-tick paired changes from both native branch states')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.steps_per_horizon < 1 or args.batch_size < 2:
        parser.error('Positive updates and at least two starts per batch required')
    if args.pulse_audit and not (args.resume and args.paired_native):
        parser.error('Intervention fitting requires the nominal ensemble and its paired archive')
    if args.fit_trunk_innovation and not args.pulse_audit:
        parser.error('Trunk response fitting requires audited native intervention pairs')
    if args.fit_ground_response and not args.pulse_audit:
        parser.error('Ground response fitting requires an audited native intervention pair')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(7372)
    data, manifest = read_data(args.native, ('active',) if args.pulse_audit else ())
    contract = json.loads((args.native/'actuator_contract.json').read_text())
    if len(args.history_native) > 3 or len(set([args.native, *args.history_native])) != 1+len(args.history_native):
        raise ValueError('Expected up to four distinct native archives')
    paired = None
    if args.pulse_audit:
        pulse = json.loads(args.pulse_audit.read_text())
        paired, intervention = read_data(args.paired_native, ('active',),
                                         expected_schema='dynamimic-native-recovery-pulse-v1')
        if (pulse['incumbent'] != manifest['checkpoint_sha256'] or intervention['pulse'] != pulse['pulse']
                or intervention['pulse']['duration_ticks'] != 16
                or any(manifest[k] != intervention[k] for k in
                       ('cases','origins','dt','checkpoint_sha256','companion_sha256'))
                or contract != json.loads((args.paired_native/'actuator_contract.json').read_text())):
            raise ValueError('Intervention identity differs from the native training archive')
        accepted = set(pulse['eligible_case_ids'])
        same = torch.tensor([c['case_id'] in accepted for c in manifest['cases']])
        from scripts.dynamimic_intervention_timing import pulse_ticks
        starts = pulse_ticks(intervention['pulse'],len(same)).clamp_min(0)
        accepted_lanes = same.nonzero().flatten()
        if (int(same.sum()) != len(accepted) or len(accepted) != pulse['eligible']
                or not torch.equal(data['before'][starts[accepted_lanes],accepted_lanes],paired['before'][starts[accepted_lanes],accepted_lanes])):
            raise ValueError('Native intervention does not reproduce the audited starting states')
        lanes = torch.arange(len(same))
        train_pairs, dev_pairs = {}, {}
        nominal_active = data.pop('active')
        for horizon in (4,8,16):
            times = starts + torch.arange(horizon)[:,None]
            valid = same & (nominal_active[times,lanes] & paired['active'][times,lanes]).all(0)
            train_pairs[horizon] = lanes[valid & (lanes%16>=4)]
            dev_pairs[horizon] = lanes[valid & (lanes%16//2==1)]
            if min(len(train_pairs[horizon]),len(dev_pairs[horizon])) < 2:
                raise ValueError(f'Insufficient independent intervention pairs at H{horizon}')
        del nominal_active
        pair_lanes = train_pairs[16]
    elif args.paired_native:
        from scripts.compare_dynamimic_native_behavior import compare, first_tick
        audit = compare(args.native, args.paired_native)
        paired, _ = read_data(args.paired_native, ('active',))
        starts = first_tick(data['live'] & data['gate'])
        idx = starts.clamp_max(min(len(data['live']),len(paired['live']))-1)
        lanes = torch.arange(len(starts))
        same = torch.tensor([r['paired_active'] and all(r['prefix_max_abs_'+k] == 0 for k in
                            ('before','request','final_joint_targets')) for r in audit['cases']])
        same &= (starts == first_tick(paired['active'])) & (starts == idx)
        same &= (data['before'][idx,lanes] == paired['before'][idx,lanes]).all(-1)
        valid = same & (lanes%16 >= 4) & (starts+48 <= min(len(data['live']),len(paired['live'])))
        times = (idx[None]+torch.arange(48)[:,None]).clamp_max(min(len(data['live']),len(paired['live']))-1)
        valid &= (paired['active'][times,lanes] & data['live'][times,lanes] & ~data['body'][times,lanes]).all(0)
        pair_lanes = lanes[valid]
        if len(pair_lanes) < 2:
            raise ValueError('No usable independent training pairs')
    latest_length = len(data['live'])
    prior_data = []
    archive_cases = [manifest['cases']]
    for directory in args.history_native:
        history, history_manifest = read_data(directory)
        other_contract = json.loads((directory/'actuator_contract.json').read_text())
        if other_contract != contract:
            raise ValueError('Historical actuator contract changed')
        if (history['live'][0] & history['gate'][0]).any():
            raise ValueError('A historical rollout window could cross the native reset')
        archive_cases.append(history_manifest['cases'])
        prior_data.append(history)
    verify_replay_cases(archive_cases)
    if prior_data:
        data = {k:torch.cat([h[k] for h in [data,*prior_data]]) for k in data}
    profiles = torch.tensor(contract['lane_profile'])
    held = torch.arange(len(profiles)) % 16 < 2
    wm = DynaMimicRecoveryWorldModel(args.asset, contract).to(args.device)
    if args.resume:
        wm.load_state_dict(torch.load(args.resume, weights_only=True, map_location=args.device))
    geometry = FullBodyGeometryV1(wm.body.asset_sha256).to(args.device)
    geometry.enable_depth_batches()
    ground_geometry = None
    if args.fit_ground_response:
        from quadsmash.rl.dynamimic_ground_clearance import Go2GroundClearance
        ground_geometry = (Go2GroundClearance(geometry).to(args.device), geometry)
    # Reserve separate cases for choosing curriculum checkpoints and for the final gate.
    development = torch.arange(len(profiles)) % 16 // 2 == 1
    pools = {k:sampling_pools(data, held | development, k) for k in (4, 16, 32)}
    validation = crossing_panel(pools[32][2], held, latest_length)
    lanes = validation[:, 1]
    dev_rows = crossing_panel(pools[32][2], development, latest_length)
    selected = {}
    selection_log = []
    metrics = []
    horizon_errors = []
    per_case = []
    members = tuple(range(len(wm.members)))
    generators = [torch.Generator().manual_seed(2315+i) for i in members]
    optimizer = torch.optim.AdamW(wm.members.parameters(), lr=3e-5 if paired is not None else 3e-4)
    def select_on_development(horizon):
        with torch.no_grad():
            _, by_tick = rollout(wm, members, data, dev_rows.repeat(len(members), 1),
                                 32, args.device, profiles, geometry)
            for member, sequence in enumerate(torch.stack(by_tick).chunk(len(members), dim=1)):
                result = crossing_metrics(sequence)
                # Minimize the worst normalized error, then the sum in case of a tie.
                errors = [result['tilt_deg']/4, result['com_height_mae_m']/.02,
                          (1-result['contact_accuracy'])/.1, result['joint_speed_mae']/1.2]
                if not all(math.isfinite(e) for e in errors):
                    raise ValueError('Non-finite development score')
                score = (max(errors), sum(errors))
                if args.pulse_audit:
                    delta = float(paired_response_loss(wm, member, data, paired, starts,
                                  dev_pairs[16], 16, args.device, profiles, impact=True, ground_geometry=ground_geometry))
                    if not math.isfinite(delta):
                        raise ValueError('Nonfinite development response score')
                    result['paired_h16'] = delta
                    score = (max(0., max(errors)-1), delta, *score)
                    if args.fit_trunk_innovation:
                        local = [innovation_loss(wm, member, data, paired, starts[dev_pairs[h]]+h,
                                                 dev_pairs[h], profiles[dev_pairs[h]].to(args.device))
                                 for h in (4,8,16)]
                        innovation = float(torch.stack(local).mean())
                        if not math.isfinite(innovation):
                            raise ValueError('Nonfinite local trunk response score')
                        result['four_tick_innovation'] = innovation
                        score = (score[0], innovation, *score[1:])
                selection_log.append(dict(horizon=horizon, member=member, **result))
                if member not in selected or score < selected[member][0]:
                    state = {key: value.detach().cpu().clone() for key, value in
                             wm.members[member].state_dict().items()}
                    selected[member] = (score, state, horizon)
        (args.output/'development.json').write_text(json.dumps(selection_log, indent=2)+'\n')

    if args.pulse_audit:
        select_on_development(0)
    for horizon, (all_rows, crossing, _) in pools.items():
        if not len(crossing):
            raise ValueError(f'No train-side touchdown support at H{horizon}')
        for i in range(args.steps_per_horizon):
            n = args.batch_size//2
            rows = torch.cat([pool[torch.randint(len(pool), (size,), generator=generator)]
                              for generator in generators
                              for pool, size in ((crossing, n), (all_rows, args.batch_size-n))])
            optimizer.zero_grad(set_to_none=True)
            loss, _ = rollout(wm, members, data, rows, horizon, args.device, profiles)
            if paired is not None:
                if args.pulse_audit:
                    paired_horizon = {4:4,16:8,32:16}[horizon]
                    pair_lanes = train_pairs[paired_horizon]
                else:
                    paired_horizon = {4:16,16:32,32:48}[horizon]
                sampled = torch.cat([pair_lanes[torch.randint(len(pair_lanes), (n,), generator=g)] for g in generators])
                loss = loss + .2 * paired_response_loss(wm, members, data, paired, starts, sampled,
                                                       paired_horizon, args.device, profiles, impact=bool(args.pulse_audit),
                                                       ground_geometry=ground_geometry)
                if args.fit_trunk_innovation:
                    loss = loss + innovation_loss(wm, members, data, paired, starts[sampled]+paired_horizon,
                                                  sampled, profiles[sampled].to(args.device))
            # Batched mean includes all members. Restore each independent batch's
            # gradient scale before clipping its parameters separately.
            (loss * len(members)).backward()
            for net in wm.members:
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            if i == 0 or (i+1) % 10 == 0 or i+1 == args.steps_per_horizon:
                print(f'H{horizon} step {i+1}/{args.steps_per_horizon} per member, '
                      f'mean train loss {float(loss):.6g}', flush=True)
        select_on_development(horizon)
    for member in members:
        wm.members[member].load_state_dict(selected[member][1])
        with torch.no_grad():
            _, per_tick = rollout(wm, member, data, validation, 32, args.device, profiles, geometry)
            # Terminal H32 body errors and all 32 steps of per-foot contact agreement.
            sequence = torch.stack(per_tick)
            horizon_errors.append({str(h):dict(tilt_median_deg=float(sequence[h-1,:,0].median()),
                com_height_mae_m=float(sequence[h-1,:,1].mean()), contact_accuracy=float(sequence[h-1,:,2].mean()),
                joint_speed_mae=float(sequence[h-1,:,3].mean())) for h in (1,4,8,16,32)})
            scores = per_tick[-1]
            per_case.append([dict(case_id=manifest['cases'][lane]['case_id'], start_tick=int(row[0]),
                                  tilt_deg=float(score[0]), height_absolute_error_m=float(score[1]),
                                  foot_accuracy=float(sequence[:,idx,2].mean()), joint_speed_mae=float(score[3]))
                             for idx,(lane,row,score) in enumerate(zip(lanes.tolist(),validation,scores))])
            metrics.append(crossing_metrics(sequence))
    passed = all(m['tilt_deg']<4 and m['com_height_mae_m']<.02 and m['contact_accuracy']>.9 and
                 m['joint_speed_mae']<1.2 for m in metrics)
    torch.save(wm.state_dict(), args.output/'ensemble.pt')
    (args.output/'validation.json').write_text(json.dumps(dict(passed=passed, members=metrics, ensemble_sha256=hashlib.sha256((args.output/'ensemble.pt').read_bytes()).hexdigest(), native_checkpoint=manifest['checkpoint_sha256'],
        fit_trunk_innovation=args.fit_trunk_innovation, fit_ground_response=args.fit_ground_response, paired_train_by_horizon={k:len(v) for k,v in train_pairs.items()} if args.pulse_audit else None,
        paired_development_by_horizon={k:len(v) for k,v in dev_pairs.items()} if args.pulse_audit else None,
        paired_native=str(args.paired_native) if paired is not None else None,
        paired_training_cases=len(pair_lanes) if paired is not None else 0,
        sources=[str(args.native), *map(str,args.history_native)], horizon_errors=horizon_errors,
        selected_horizons=[selected[i][2] for i in members],
        development_cases=[manifest['cases'][int(l)]['case_id'] for l in dev_rows[:,1]],
        per_case=per_case, train_pool_sizes={h:[len(a),len(b)] for h,(a,b,_) in pools.items()},
        held_cases=[manifest['cases'][i]['case_id'] for i in lanes.tolist()]),indent=2)+'\n')


if __name__ == '__main__':
    main()
