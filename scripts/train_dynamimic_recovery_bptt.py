"""Recovery imagination objectives and sampling from audited native execution snapshots."""
import math
from functools import lru_cache
from time import perf_counter

import torch
from torch.nn import functional as F

from quadsmash.rl.dynamimic_lagrangian_v1 import mv, rotation


from scripts.fit_dynamimic_recovery_critic import FALL_COST, NATIVE_RETURN_SCHEMA, ROOT_FAILURE_HEIGHT, root_termination


def root_survival(previous, height_w):
    # Smooth the terminal boundary over +/- 1 cm. Latching the lowest survival
    # prevents paying the same fall cost again or rewarding an imagined rebound.
    x = ((height_w - ROOT_FAILURE_HEIGHT + .01) / .02).clamp(0., 1.)
    return torch.minimum(previous, x.square() * (3. - 2. * x))


def expected_live_reward(previous_survival, next_survival, reward):
    return next_survival * reward - FALL_COST * (previous_survival - next_survival)


def candidate_steps(measure):
    """Apply each interpolated policy before yielding its measured action-space step."""
    alpha = 1.
    previous_rms = math.inf
    for label, budget in [('moderate', .035), ('conservative', .020)]:
        for _ in range(25):
            rms = measure(alpha)
            if math.isfinite(rms) and 0 < rms <= budget and rms < previous_rms:
                break
            alpha *= .5
        else:
            raise ValueError(f'No distinct nonzero {label} proposal within the action budget')
        yield label, budget, alpha, rms
        previous_rms = rms
        alpha *= .5


def recovery_start_groups(data, snapshot_ticks, held_lanes):
    """Classify (tick, lane) using native events; future falls only select training starts."""
    ticks = torch.arange(len(data['live']))[:, None]
    landed = data['landed'].bool()
    # Both landed and body are captured at the END of a control step.
    had_landed = F.pad(landed[:-1], (0, 0, 1, 0))
    landing_events = landed & ~had_landed & data['live']
    touchdown = torch.where(landing_events, ticks+1, len(ticks)+1).amin(dim=0)
    eligible = data['active'] & ~held_lanes
    eligible &= data['obs'][..., 775] - data['release_time'] >= .04
    eligible &= torch.isin(ticks, torch.as_tensor(snapshot_ticks))
    future_fall = F.pad((data['body'] | data['root_terminated']) & data['live'], (0,0,0,49)).unfold(0,50,1).any(dim=-1)
    age = ticks - touchdown
    near = had_landed & ((age < 30) | future_fall)
    stable = F.pad(data['strict'][:-1], (0, 0, 1, 0))
    masks = (~had_landed, near, had_landed & ~near & ~stable & (age >= 30) & (age <= 250))
    return tuple((eligible & mask).nonzero() for mask in masks)


def native_segments(rows, horizon, active, snapshots, segment_ticks=16):
    segments = []
    for offset in range(0,horizon,16):
        tick = rows[:,0]+offset
        lane = rows[:,1]
        exists = (tick<len(active)) & torch.isin(tick, torch.as_tensor(snapshots))
        exists &= active[tick.clamp_max(len(active)-1),lane]
        if exists.any():
            segments.append((torch.stack((tick[exists],lane[exists]),1),min(segment_ticks,horizon-offset)))
    return segments


def imminent_root_rows(data):
    root_only = data['root_terminated'] & data['live'] & ~data['body'] & ~data['net']
    # G07 snapshots are 16 ticks apart. This includes the last recoverable
    # snapshot while the measured dynamics still resolve the height cutoff.
    return F.pad(root_only, (0, 0, 0, 15)).unfold(0, 16, 1).any(-1)


def body_fault_window_counts(rows, landing_tick, first_fault_tick, horizon, archive_ticks):
    """Count sampled starts and independent lanes whose rollout reaches first body contact."""
    tick, lane = rows.T
    fault = first_fault_tick[lane]
    landing = landing_tick[lane]
    reached = (fault < archive_ticks) & (tick <= fault) & (fault < tick + horizon)
    delay = fault - landing
    counts = {}
    for name, condition in (('before_landing', delay < 0), ('first_second', (delay >= 0) & (delay < 100)),
                            ('later', delay >= 100)):
        covered = lane[reached & condition]
        counts[name] = dict(starts=len(covered), distinct_cases=len(covered.unique()))
    return counts


def late_body_windows(data, horizon):
    ticks = torch.arange(len(data['live']))[:, None]
    end = len(ticks)
    landing = torch.where(data['landed'] & data['live'], ticks + 1, end).amin(0)
    fault = torch.where(data['body'] & data['live'], ticks, end).amin(0)
    return ((ticks >= landing + 100) & (ticks <= fault) & (fault < ticks + horizon)
            & (fault < end) & data['active'])


def stratified_starts(groups, count, generator, imminent=None, late_body=None):
    if not 128 <= count <= 256:
        raise ValueError('Recovery expects 128 to 256 native starts')
    sizes = [round(.3*count), round(.4*count)]
    sizes.append(count-sum(sizes))
    sampled = []
    taken = 0
    for phase, (pool, size) in enumerate(zip(groups, sizes, strict=True)):
        # Odd phase sizes alternate their extra slot across the complete batch.
        primary_count = (taken+size+1)//2 - (taken+1)//2
        taken += size
        for policy, needed in enumerate((primary_count, size-primary_count)):
            cases = pool[pool[:,1] % 2 == policy]
            if len(cases) < needed:
                raise ValueError(f'Approach {policy} has {len(cases)} starts for {needed} requested')
            if phase == 1 and imminent is not None:
                urgent = imminent[cases[:, 0], cases[:, 1]]
                # Reserve one quarter of the touchdown allocation for these
                # short windows. Spill either way if a policy has few failures.
                n_urgent = min(int(urgent.sum()), max(round(needed/4), needed-int((~urgent).sum())))
                if n_urgent:
                    candidates = cases[urgent]
                    ids = torch.randperm(len(candidates), generator=generator)[:n_urgent]
                    sampled.extend(candidates[ids])
                    needed -= n_urgent
                    cases = cases[~urgent]
            if phase == 1 and late_body is not None and needed:
                late = cases[late_body[cases[:, 0], cases[:, 1]]]
                # Each reserved slot covers a different measured fall for this Approach.
                lanes = late[:, 1].unique()
                lanes = lanes[torch.randperm(len(lanes), generator=generator)[:round(needed / 3)]]
                for lane in lanes:
                    choices = late[late[:, 1] == lane]
                    row = choices[torch.randint(len(choices), (), generator=generator)]
                    sampled.append(row)
                    cases = cases[~(cases == row).all(1)]
                needed -= len(lanes)
            ordered = cases[torch.randperm(len(cases), generator=generator)]
            cursors = {}
            tiers = []
            for row in ordered:
                lane = int(row[1])
                rank = cursors.get(lane, 0)
                cursors[lane] = rank + 1
                if rank == len(tiers):
                    tiers.append([])
                tiers[rank].append(row)
            sampled.extend([row for tier in tiers for row in tier][:needed])
    return torch.stack(sampled)


def recovery_reward(q, v, contacts, geometry, landed, body_fault=None, absorbing_body=False, fk=None):
    """Smooth physical margins. Link origins are calibrated surrogates for the ground barrier."""
    rot = rotation(q[:, 3:7])
    linear, angular = mv(rot, v[:, :3]), mv(rot, v[:, 3:6])
    if fk is None:
        fk = geometry(q[:, 7:], v[:, 6:], q[:, :3], q[:, 3:7], linear, angular)
    # Squared sine/cosine terms have a finite gradient at exact upright.
    tilt_sq = 2 * (1 - rot[:, 2, 2].clamp(-1, 1))
    speed = linear[:, :2].norm(dim=-1)
    omega = angular.norm(dim=-1)
    max_leg_speed = v[:, 6:18].abs().amax(dim=-1)
    max_joint_speed = v[:, 6:].abs().amax(dim=-1)
    support = contacts.mean(-1)
    stable = (torch.sigmoid((math.radians(15)**2 - tilt_sq)/.012) *
              torch.sigmoid((.3-speed)/.05) * torch.sigmoid((1.-omega)/.15) *
              torch.sigmoid((1.-max_joint_speed)/.15) * (1-(1-contacts).prod(-1)))
    upright = torch.exp(-tilt_sq/math.radians(12)**2)
    ground_reward = (.03 + .05*torch.exp(-tilt_sq/math.radians(30)**2) + .05*upright +
                     .03*torch.exp(-((q[:, 2]-.34)/.07).square()) + .03*support +
                     .06*torch.exp(-max_leg_speed.square()) + .03*torch.exp(-(speed/.30).square()) +
                     .03*torch.exp(-omega.square()) + .08*stable)
    air_reward = .015*torch.exp(-tilt_sq/math.radians(20)**2) + .01*torch.exp(-omega.square())
    sites = (geometry.root_id, *geometry.heads,
             *(geometry.ids[name+'_hip'] for name in ('FL','FR','RL','RR')))
    h_min = fk['position'][:,sites,2].amin(-1)
    net = (fk['position'][:,geometry.ids['Racket'],0]-1.38).relu().square() + (q[:,0]-1.06).relu().square()
    if absorbing_body:
        ground_cost = 0.
    elif body_fault is None:
        ground_cost = 20*(.14-h_min).relu().square()
    else:
        from scripts.dynamimic_body_fault import body_fault_score
        ground_cost = .15 * body_fault_score(q, v, contacts, body_fault)
    return landed*ground_reward + (1-landed)*air_reward - ground_cost - 20*net


def saturation_cost(raw):
    return .08 * (raw.abs()-.90).relu().square().mean()


def focus_cost(weight, anchor):
    diff = (weight-anchor).square()
    history = diff[:, :759].reshape(len(weight), 11, 69)
    return (12*history[:,:,61:67].sum() + 10*diff[:,776:794].sum())/diff.numel()


def mixed_request(primary, secondary, obs, use_secondary, release):
    # Both policies have the same base, gate and frozen arm reference. Use the
    # shared trainable Recovery exactly once, including in companion lanes.
    gate = torch.ones(len(obs), device=obs.device, dtype=torch.bool)
    mean = primary._gated_distribution(obs, gate, release_time=release).loc
    z = primary.actor.normalizer(obs)
    extra = secondary.actor.approach(z) - primary.actor.approach(z)
    return mean + use_secondary[:, None]*primary.approach_action_gate(obs)*primary.actor.approach_scale*extra


def command_context(obs, q, v):
    cmd = obs[:, 759:770]
    rot = rotation(q[:, 3:7])
    relative = mv(rot, cmd[:, 1:4])
    velocity = mv(rot, v[:, :3] + cmd[:, 7:10]) + torch.linalg.cross(mv(rot, v[:, 3:6]), relative)
    return dict(goal_enabled=cmd[:, 10].bool(), goal_position_w=q[:, :3]+relative,
                goal_normal_w=mv(rot, cmd[:, 4:7]), goal_velocity_w=velocity,
                goal_time_s=obs[:, 775]+cmd[:, 0])


def released_ball_path(ball, horizon):
    from types import SimpleNamespace
    from quadsmash.physics.court_v4 import load_court
    from quadsmash.physics.native import physics_contract
    from quadsmash.rl.supermimic_outgoing_flight import predict_flight_events
    physical = physics_contract()
    prior = SimpleNamespace(gravity=ball.new_tensor([0.,0.,-9.81]),
                            wind=ball.new_tensor(physical['wind_m_s']),
                            restitution=physical['restitution'], drag=physical['quadratic_drag_k_per_m'])
    times = [step*.01 for step in range(1,horizon+1)]
    flight = predict_flight_events(ball[:, :3],ball[:, 3:],prior,load_court(),duration=times[-1],
        dt=.01,radius=physical['collision_radius_m'],sample_times=times)
    path = []
    for t in times:
        if t in flight['samples']:
            row = flight['samples'][t]
            pos = row['position']
            vel = torch.where(row['airborne'][:,None], row['velocity'], 0.)
        else:
            pos, vel = path[-1]
        path.append((pos,vel))
    return path


def imagine(starts, memory, primary, secondary, preview, wm, geometry, observer, tail, member, horizon, body_fault=None, absorbing_body=False, gradient_terms=False, tail_gradient_scale=1., ground_shapes=None):
    from quadsmash.rl.dynamimic_recovery_wm_v1 import local_robot_state
    from quadsmash.rl.supermimic_physical_observation_v1 import PhysicalPolicyInputs
    from torch.utils.checkpoint import checkpoint
    obs = starts['obs']
    q, v = local_robot_state(starts['physical'])
    joint_default = q[:, 7:] - obs[:, 9:27]
    default = joint_default[0]
    if not torch.allclose(joint_default, default.expand_as(joint_default), atol=1e-5, rtol=0):
        raise ValueError('Native G07 joint defaults inconsistent')
    context = command_context(obs, q, v)
    path = released_ball_path(starts['physical'][:,49:], horizon)
    landed = starts['landed'].to(q.dtype)
    elapsed = obs[:, 775]
    present = obs[:, 67].bool()
    ever_airborne = (obs[:, 773:775].sum(-1)>0) | starts['landed']
    survival = q.new_ones(len(q))
    root_alive = survival
    body_alive = survival
    discount = 1.
    value = q.new_zeros(len(q))
    saturation = q.new_zeros(())
    disagreement = q.new_zeros(())
    all_active = torch.ones_like(present)
    history = obs[:,:759].reshape(-1,11,69)

    def body_step(q_in, v_in, command):
        indices = (member, (member + 1) % 3, (member + 2) % 3)
        all_q, all_v, aux = wm(q_in.repeat(3, 1), v_in.repeat(3, 1), command.repeat(3, 1),
                               indices, starts['profile'].repeat(3))
        ensemble = torch.cat((all_q, all_v), -1).unflatten(0, (3, len(q_in)))
        variance = ensemble.var(dim=0, unbiased=False).mean()
        return all_q[:len(q_in)], all_v[:len(q_in)], aux['foot_contact'][:len(q_in)], variance

    for step in range(horizon):
        request = mixed_request(primary, secondary, obs, starts['companion'], starts['release_time'])
        saturation = saturation + saturation_cost(request)/horizon
        command, memory, _ = preview.forward(obs, request.clamp(-1,1), default, starts['limits'], memory)
        if step == 0 and starts.get('audit_target') is not None:
            torch.testing.assert_close(command.final_joint_targets.detach(), starts['audit_target'], atol=2e-5, rtol=2e-5)
        q, v, contacts, variance = checkpoint(body_step, q, v, command.final_joint_targets, use_reentrant=False)
        disagreement = disagreement + variance/horizon
        landed = torch.maximum(landed, contacts.amax(-1))
        root_alive = root_survival(root_alive, q[:, 2] + starts['origin_z'])
        fk = None
        if ground_shapes is not None:
            from quadsmash.rl.dynamimic_ground_clearance import ground_survival
            rot = rotation(q[:,3:7])
            fk = geometry(q[:,7:], v[:,6:], q[:,:3], q[:,3:7], mv(rot,v[:,:3]), mv(rot,v[:,3:6]))
            gaps = ground_shapes(fk['position'], fk['rotation'])
            body_alive = ground_survival(body_alive, gaps)
        elif absorbing_body:
            from scripts.dynamimic_body_fault import body_fault_score
            body_alive = body_alive * (1 - body_fault_score(q, v, contacts, body_fault))
        next_survival = root_alive * body_alive
        step_reward = recovery_reward(q, v, contacts, geometry, landed, body_fault, absorbing_body or ground_shapes is not None, fk)
        if ground_shapes is not None:
            step_reward = step_reward + landed * ground_shapes.thigh_margin_reward(gaps)
        value = value + discount * expected_live_reward(survival, next_survival, step_reward)
        survival = next_survival
        discount *= .995
        elapsed = elapsed + .01
        rot = rotation(q[:,3:7])
        ball_p, ball_v = path[step]
        physical = PhysicalPolicyInputs(root_position_w=q[:,:3], root_quaternion_w=q[:,3:7],
            root_link_velocity_w=mv(rot,v[:,:3]), root_angular_velocity_w=mv(rot,v[:,3:6]),
            joint_position=q[:,7:], joint_velocity=v[:,6:], joint_default=joint_default,
            foot_contact=contacts, ball_position_w=ball_p, ball_velocity_w=ball_v, ball_present=present,
            airborne=contacts.amax(-1)<.5, airborne_seen=ever_airborne, hit_seen=all_active,
            elapsed_s=elapsed, final_action=command.final_normalized, **context)
        obs, history = observer(physical, history)
    terminal = tail(obs)
    continuation = discount * survival * terminal
    if gradient_terms:
        # Disjoint observation groups give the exact first derivative at the recorded endpoint.
        # The derivative of the value with respect to its input is held fixed for these VJPs.
        sensitivity, = torch.autograd.grad(
            -(tail_gradient_scale * discount * survival.detach() * terminal).mean(), obs, retain_graph=True)
        attribution = obs * sensitivity.detach()
        frames = attribution[:, :759].reshape(len(obs), 11, 69)
        return {'running_reward': -value.mean(),
                'tail_proprioception': frames[:, :, :45].sum() + frames[:, :, 57:61].sum(),
                'tail_racket': frames[:, :, 45:57].sum(), 'tail_ball_and_flags': frames[:, :, 61:].sum(),
                'tail_command_phase_clock': attribution[:, 759:776].sum(),
                'tail_last_action': attribution[:, 776:].sum(),
                'tail_survival': -(tail_gradient_scale * discount * survival * terminal.detach()).mean(),
                'saturation': saturation, 'disagreement': disagreement}
    if tail_gradient_scale != 1:
        continuation = continuation.detach() + tail_gradient_scale * (continuation - continuation.detach())
    return -(value+continuation).mean() + saturation + disagreement


def main():
    import argparse
    import hashlib
    import json
    from pathlib import Path
    from quadsmash.control.g07_shared_target_residual_v1 import make_g07_residual_controller
    from quadsmash.rl.dynamimic_recovery_wm_v1 import DynaMimicRecoveryWorldModel
    from quadsmash.rl.supermimic_compact_starts import compact_start
    from quadsmash.rl.supermimic_full_body_geometry_v1 import FullBodyGeometryV1
    from quadsmash.rl.supermimic_g07_memory_preview_v1 import G07MemoryPreviewV1, pack_planned_execution
    from quadsmash.rl.supermimic_physical_capture_v1 import restore_archive_execution
    from quadsmash.rl.supermimic_physical_loop_loading_v1 import RECOVERY_NAMES, freeze_execution_modules
    from quadsmash.rl.supermimic_physical_observation_v1 import PhysicalPolicyObservationV1
    from quadsmash.rl.supermimic_policy_execution_v1 import resolve_contract
    from quadsmash.rl.supermimic_v1 import load_ppo_initialized_actor
    from scripts.fit_dynamimic_recovery_critic import make_tail
    from scripts.fit_dynamimic_recovery_wm import read_data
    from scripts.run_supermimic_recovery_baseline import verify, dump

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('native','checkpoint','companion','asset','wm','critic','output'):
        parser.add_argument('--'+name, required=True, type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--native-body-fault', action='store_true')
    group.add_argument('--authored-ground', action='store_true')
    group.add_argument('--absorbing-body-fault', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--horizon', type=int, choices=(48,64), default=48)
    parser.add_argument('--audit-segment-gradients', action='store_true',
                        help='Measure H16 objective gradients on native starts and exit before updating')
    parser.add_argument('--tail-gradient-scale', type=float, default=1.,
                        help='Scale the gradient through the discounted continuation value, preserving its forward value')
    parser.add_argument('--sample-late-body', action='store_true')
    parser.add_argument('--native-segments', action='store_true', help='Re-anchor every 16 ticks to measured state and G07 memory')
    parser.add_argument('--updates', type=int, choices=range(15,26), default=20)
    parser.add_argument('--starts', type=int, default=128)
    parser.add_argument('--microbatch', type=int, default=64)
    parser.add_argument('--segment-ticks', type=int, choices=(4,8,16), default=16,
                        help='Open-loop ticks per native G07 snapshot, spaced 16 ticks apart')
    args = parser.parse_args()
    if args.segment_ticks != 16 and not args.native_segments:
        parser.error('Short segment lengths require --native-segments')
    if not 0 <= args.tail_gradient_scale <= 1:
        parser.error('Tail gradient scale must lie in [0, 1]')
    if not 1 <= args.microbatch <= args.starts:
        parser.error('Microbatch must fit the full native start batch')
    report = json.loads((args.wm/'validation.json').read_text())
    if not report['passed'] or not all(m['tilt_deg']<4 and m['com_height_mae_m']<.02 and
                                       m['contact_accuracy']>.9 and m['joint_speed_mae']<1.2 for m in report['members']) or len(report['members']) != 3:
        raise ValueError('H32 crossing touchdown validation failed')
    if report['ensemble_sha256'] != hashlib.sha256((args.wm/'ensemble.pt').read_bytes()).hexdigest():
        raise ValueError('World model weights differ from the scored ensemble')
    args.output.mkdir(parents=True, exist_ok=False)
    verify([args.checkpoint,args.companion], synchronize_recovery=True)
    data, manifest = read_data(args.native, ('active','obs','release_time','strict','relaxed','net','request','done'))
    if report['native_checkpoint'] != manifest['checkpoint_sha256']:
        raise ValueError('World model acceptance refers to a different behavior policy')
    for key, path in [('checkpoint_sha256',args.checkpoint),('companion_sha256',args.companion)]:
        if manifest[key] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError('Native rollout identity differs from BPTT start')
    device = torch.device(args.device)
    body_fault = None
    if args.absorbing_body_fault:
        from scripts.dynamimic_body_fault import calibrate
        ranking, calibration = calibrate(data)
        body_fault = ranking * calibration[0]
        body_fault[-1] += calibration[1]
        if not body_fault.isfinite().all() or calibration[0] <= 0:
            raise ValueError('Invalid native first-contact incidence calibration')
        dump(args.output/'body_fault.json', dict(calibration=calibration.tolist(), coefficients=body_fault.tolist(),
                                                 native=manifest['checkpoint_sha256'], terminal_cost=FALL_COST))
        body_fault = body_fault.to(device)
    elif args.native_body_fault:
        from scripts.dynamimic_body_fault import fit_native_body_fault
        body_fault, fault_metrics = fit_native_body_fault(data)
        dump(args.output/'body_fault.json', dict(held=fault_metrics, coefficients=body_fault.tolist(),
                                                native=manifest['checkpoint_sha256'], cost_scale=.15))
        body_fault = body_fault.to(device)
    contract = json.loads((args.native/'actuator_contract.json').read_text())
    wm = DynaMimicRecoveryWorldModel(args.asset, contract).to(device)
    wm.load_state_dict(torch.load(args.wm/'ensemble.pt',map_location=device,weights_only=True))
    wm.requires_grad_(False).eval()
    geometry = FullBodyGeometryV1(wm.body.asset_sha256).to(device)
    geometry.enable_depth_batches()
    ground_shapes = None
    if args.authored_ground:
        from quadsmash.rl.dynamimic_ground_clearance import Go2GroundClearance
        ground_shapes = Go2GroundClearance(geometry).to(device)
    observer = PhysicalPolicyObservationV1().to(device=device, dtype=wm.kp.dtype)
    tail, source_sha = make_tail(args.checkpoint, device)
    value_state = torch.load(args.critic/'critic.pt', weights_only=True, map_location=device)
    critic_fit = json.loads((args.critic/'fit.json').read_text())
    if value_state['source'] != source_sha or critic_fit['native_checkpoint'] != manifest['checkpoint_sha256']:
        raise ValueError('Native critic identity mismatch')
    if value_state.get('return_schema') != NATIVE_RETURN_SCHEMA:
        raise ValueError('Native tail must use the same low-root terminal cost as imagination')
    tail.load_native_fit(value_state)
    for param in tail.parameters():
        param.requires_grad_(False)
    ckpt = torch.load(args.checkpoint, mmap=True,map_location='cpu',weights_only=False)
    metadata = ckpt['metadata']['init_policy_checkpoint']
    source = Path(metadata['path'])
    policies = []
    for path in (args.checkpoint, args.companion):
        policy,_ = load_ppo_initialized_actor(source,device)
        payload = torch.load(path,mmap=True,map_location='cpu',weights_only=False)
        policy.load_state_dict(payload['actor'], strict=True)
        policy.configure_approach_gate(resolve_contract(payload)[0].get('arm_preparation_seconds'))
        for name,param in policy.named_parameters():
            param.requires_grad_(not policies and name in RECOVERY_NAMES)
        policies.append(policy.eval())
    primary, secondary = policies
    params = {k:p for k,p in primary.named_parameters() if p.requires_grad}
    if set(params) != set(RECOVERY_NAMES) or sum(p.numel() for p in params.values()) != 111186:
        raise ValueError('Recovery head parameter contract differs')
    anchor = {k:p.detach().clone() for k,p in params.items()}
    optimizer = torch.optim.Adam(params.values(), lr=5e-5)
    controller = make_g07_residual_controller(torch.load(source,map_location='cpu',weights_only=False),
                                              256,device,metadata['g07_execution_root'])
    freeze_execution_modules(controller)
    preview = G07MemoryPreviewV1(controller)
    preview.mixer.to(device)
    held = torch.arange(256)%16 < 2
    directory = args.native/'trajectories'
    ticks = sorted(int(p.stem.split('_')[-1]) for p in directory.glob('g07_*.pt'))
    origins_z = torch.as_tensor(manifest['origins'])[:, 2]
    data['root_terminated'] = root_termination(data['done'], data['after'][..., 2] + origins_z)
    pools = recovery_start_groups(data, ticks, held)
    imminent = imminent_root_rows(data)
    late_body = late_body_windows(data, args.horizon)
    end_tick = len(data['live'])
    control_tick = torch.arange(end_tick)[:, None]
    landing_tick = torch.where(data['live'] & data['landed'], control_tick, end_tick).amin(0)
    first_fault_tick = torch.where(data['live'] & data['body'], control_tick, end_tick).amin(0)
    generator = torch.Generator().manual_seed(9917)
    success = ((data['relaxed']|data['strict']) & data['live']).any(0) & ~(
        (data['body']|data['net']|data['root_terminated']) & data['live']).any(0)
    keep = (data['active'] & success & ~held).nonzero()
    if not len(keep):
        raise ValueError('No original successful training trajectories for retention')
    profiles = torch.tensor(contract['lane_profile'])
    limits = torch.tensor(contract['profiles'][0]['position_limits'], device=device)
    if any(profile['position_limits'] != contract['profiles'][0]['position_limits'] for profile in contract['profiles']):
        raise ValueError('G07 expects shared physical joint limits')

    @lru_cache(maxsize=32)
    def archived_execution(tick):
        return torch.load(directory/f'g07_{tick:04d}.pt', weights_only=False, mmap=True, map_location='cpu')

    def context(rows, audit):
        branches, sizes, ordered = [], [], []
        for tick in rows[:,0].unique().tolist():
            subset = rows[rows[:,0]==tick]
            ids = subset[:,1].sort().values
            mask = torch.zeros(256,dtype=torch.bool)
            mask[ids] = True
            snap = archived_execution(tick)
            row,_ = compact_start(dict(before={},observation=data['obs'][tick],executor_before=snap),
                                  mask, torch.zeros(256,6))
            branches.append(restore_archive_execution(controller,row['executor_before'],device=device,
                                                      array_plan_observations=True))
            sizes.append(len(ids))
            ordered.append(torch.stack((torch.full_like(ids,tick),ids),1))
        t,l = torch.cat(ordered).T
        start = dict(physical=data['before'][t,l].to(device), obs=data['obs'][t,l].to(device),
                     release_time=data['release_time'][t,l].to(device), companion=(l%2==1).to(device),
                     profile=profiles[l].to(device), origin_z=origins_z[l].to(device), landed=data['landed'][(t-1).clamp_min(0),l].to(device),
                     limits=limits, audit_target=data['final_joint_targets'][t,l].to(device) if audit else None)
        return start,pack_planned_execution(tuple(branches),tuple(sizes))

    if args.audit_segment_gradients:
        near = pools[1]
        late = late_body[near[:,0], near[:,1]]
        eligible = data['active'] & ~held & success
        eligible &= torch.isin(torch.arange(len(eligible))[:,None],torch.as_tensor(ticks))
        eligible &= data['landed'] & (data['obs'][...,775] - data['release_time'] >= .04)
        groups = [('early_impact_or_prefall',near[~late]), ('late_prefall',near[late]),
                  ('successful_landed',eligible.nonzero())]
        reports = {}
        for label, group in groups:
            sample = []
            for parity in (0,1):
                subset = group[group[:,1] % 2 == parity]
                ids = subset[:,1].unique()
                for lane in ids[torch.randperm(len(ids),generator=generator)[:4]]:
                    rows = subset[subset[:,1]==lane]
                    sample.append(rows[torch.randint(len(rows),(),generator=generator)])
            rows = torch.stack(sample)
            members = []
            for member in range(3):
                start, memory = context(rows, audit=True)
                parts = imagine(start,memory,primary,secondary,preview,wm,geometry,observer,tail,member,
                                args.segment_ticks,body_fault,args.absorbing_body_fault,gradient_terms=True, tail_gradient_scale=args.tail_gradient_scale, ground_shapes=ground_shapes)
                gradients = {}
                for i,(name, loss) in enumerate(parts.items()):
                    gs = torch.autograd.grad(loss, tuple(params.values()), retain_graph=i+1<len(parts))
                    gradients[name] = torch.cat([g.detach().flatten() for g in gs]).double()
                total = sum(gradients.values())
                members.append({k:dict(norm=float(g.norm()), projected_share=float(g@total/total.square().sum()))
                                for k,g in gradients.items()})
            reports[label] = dict(starts=rows.tolist(), members=members)
        dump(args.output/'gradients.json',reports)
        return

    evidence = []
    checks = stratified_starts(pools,args.starts,generator)
    t,l = checks.T
    def actions(rows):
        t,l = rows.T
        return mixed_request(primary, secondary, data['obs'][t,l].to(device), (l%2==1).to(device),
                             data['release_time'][t,l].to(device)).clamp(-1,1)
    with torch.no_grad():
        original = actions(checks)
    for step in range(args.updates):
        started = perf_counter()
        rows = stratified_starts(pools,args.starts,generator,imminent=imminent,
                                 late_body=late_body if args.sample_late_body else None)
        optimizer.zero_grad(set_to_none=True)
        total = 0.
        segments = [(rows,args.horizon)]
        if args.native_segments:
            segments = native_segments(rows, args.horizon, data['active'], ticks, args.segment_ticks)
        weight_sum = sum(len(batch) for batch,_ in segments)
        for batch,length in segments:
            for group in batch.split(args.microbatch):
                start, memory = context(group, audit=step==0)
                loss = imagine(start,memory,primary,secondary,preview,wm,geometry,observer,tail,
                               step%3,length,body_fault,args.absorbing_body_fault,tail_gradient_scale=args.tail_gradient_scale, ground_shapes=ground_shapes)
                weighted = loss*len(group)/weight_sum
                weighted.backward()
                total += float(weighted.detach())
        retention_rows = keep[torch.randint(len(keep),(args.starts,),generator=generator)]
        rt,rl = retention_rows.T
        retention = (actions(retention_rows)-data['request'][rt,rl].to(device)).square().mean()
        name = 'actor.recovery.0.weight'
        regularization = .1*focus_cost(params[name],anchor[name]) + retention
        regularization.backward()
        grad = torch.nn.utils.clip_grad_norm_(params.values(),.5,error_if_nonfinite=True)
        if float(grad) == 0:
            raise ValueError('Recovery gradient is zero')
        optimizer.step()
        evidence.append(dict(step=step+1, objective=total, gradient_norm=float(grad),
                             segment_counts=[len(batch) for batch,_ in segments], seconds=perf_counter()-started, microbatch=args.microbatch,
                             peak_cuda_mib=torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else None,
                             unique_native_cases=len(rows[:,1].unique()),
                             native_first_body_in_window=body_fault_window_counts(
                                 rows, landing_tick, first_fault_tick, args.horizon, end_tick),
                             late_body_prefall_starts=int(late_body[rows[:,0],rows[:,1]].sum()),
                             imminent_root_starts=int(imminent[rows[:,0],rows[:,1]].sum()),
                             h8d_starts=int((rows[:,1]%2).sum()),
                             retention=float(retention.detach())))
        print(evidence[-1],flush=True)
    with torch.no_grad():
        final = {k:p.clone() for k,p in params.items()}
        def measure(alpha):
            for k, p in params.items():
                p.copy_(torch.lerp(anchor[k], final[k], alpha))
            return float((actions(checks)-original).square().mean().sqrt())

        for label, budget, alpha, rms in candidate_steps(measure):
            state = {k:v.cpu().clone() for k,v in primary.state_dict().items()}
            if state.keys() != ckpt['actor'].keys() or any(not torch.equal(val,ckpt['actor'][k])
                    for k,val in state.items() if k not in RECOVERY_NAMES):
                raise ValueError('Frozen weights changed')
            torch.save(dict(ckpt, actor=state, actor_updates=ckpt['actor_updates']+args.updates),
                       args.output/(label+'.pt'))
            dump(args.output/(label+'.json'),dict(alpha=alpha,rms=rms,budget=budget))
    dump(args.output/'training.json',dict(updates=evidence, pools=[len(p) for p in pools], retained_cases=int(success.sum()),
                                         tail_gradient_scale=args.tail_gradient_scale, horizon=args.horizon, authored_ground=args.authored_ground, native_segments=args.native_segments, segment_ticks=args.segment_ticks, native_body_fault=args.native_body_fault, absorbing_body_fault=args.absorbing_body_fault, native=manifest['checkpoint_sha256']))


if __name__ == '__main__':
    main()
