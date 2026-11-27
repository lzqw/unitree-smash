"""Recovery imagination objectives and sampling from audited native execution snapshots."""
import math
from functools import lru_cache
from time import perf_counter

import torch
from torch.nn import functional as F

from quadsmash.rl.dynamimic_lagrangian_v1 import mv, rotation


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
    future_body = F.pad(data['body'] & data['live'], (0,0,0,49)).unfold(0,50,1).any(dim=-1)
    age = ticks - touchdown
    near = had_landed & ((age < 30) | future_body)
    stable = F.pad(data['strict'][:-1], (0, 0, 1, 0))
    masks = (~had_landed, near, had_landed & ~near & ~stable & (age >= 30) & (age <= 250))
    return tuple((eligible & mask).nonzero() for mask in masks)


def stratified_starts(groups, count, generator):
    if not 128 <= count <= 256:
        raise ValueError('Recovery expects 128 to 256 native starts')
    sizes = [round(.3*count), round(.4*count)]
    sizes.append(count-sum(sizes))
    sampled = []
    taken = 0
    for pool, size in zip(groups, sizes, strict=True):
        # Odd phase sizes alternate their extra slot across the complete batch.
        primary_count = (taken+size+1)//2 - (taken+1)//2
        taken += size
        for policy, needed in enumerate((primary_count, size-primary_count)):
            cases = pool[pool[:,1] % 2 == policy]
            if len(cases) < needed:
                raise ValueError(f'Approach {policy} has {len(cases)} starts for {needed} requested')
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


def recovery_reward(q, v, contacts, geometry, landed):
    """Smooth physical margins. Link origins are calibrated surrogates for the ground barrier."""
    rot = rotation(q[:, 3:7])
    linear, angular = mv(rot, v[:, :3]), mv(rot, v[:, 3:6])
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
    return landed*ground_reward + (1-landed)*air_reward - 20*(.14-h_min).relu().square() - 20*net


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


def imagine(starts, memory, primary, secondary, preview, wm, geometry, observer, tail, member, horizon):
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
        value = value + discount * recovery_reward(q, v, contacts, geometry, landed)
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
    return -(value+discount*tail(obs)).mean() + saturation + disagreement


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
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--horizon', type=int, choices=(48,64), default=48)
    parser.add_argument('--updates', type=int, choices=range(15,26), default=20)
    parser.add_argument('--starts', type=int, default=128)
    parser.add_argument('--microbatch', type=int, default=64)
    args = parser.parse_args()
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
    data, manifest = read_data(args.native, ('active','obs','release_time','strict','relaxed','net','request'))
    if report['native_checkpoint'] != manifest['checkpoint_sha256']:
        raise ValueError('World model acceptance refers to a different behavior policy')
    for key, path in [('checkpoint_sha256',args.checkpoint),('companion_sha256',args.companion)]:
        if manifest[key] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError('Native rollout identity differs from BPTT start')
    device = torch.device(args.device)
    contract = json.loads((args.native/'actuator_contract.json').read_text())
    wm = DynaMimicRecoveryWorldModel(args.asset, contract).to(device)
    wm.load_state_dict(torch.load(args.wm/'ensemble.pt',map_location=device,weights_only=True))
    wm.requires_grad_(False).eval()
    geometry = FullBodyGeometryV1(wm.body.asset_sha256).to(device)
    geometry.enable_depth_batches()
    observer = PhysicalPolicyObservationV1().to(device=device, dtype=wm.kp.dtype)
    tail, source_sha = make_tail(args.checkpoint, device)
    value_state = torch.load(args.critic/'critic.pt', weights_only=True, map_location=device)
    critic_fit = json.loads((args.critic/'fit.json').read_text())
    if value_state['source'] != source_sha or critic_fit['native_checkpoint'] != manifest['checkpoint_sha256']:
        raise ValueError('Native critic identity mismatch')
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
    pools = recovery_start_groups(data, ticks, held)
    generator = torch.Generator().manual_seed(9917)
    success = ((data['relaxed']|data['strict']) & data['live']).any(0) & ~(
        (data['body']|data['net']) & data['live']).any(0)
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
                     profile=profiles[l].to(device), landed=data['landed'][(t-1).clamp_min(0),l].to(device),
                     limits=limits, audit_target=data['final_joint_targets'][t,l].to(device) if audit else None)
        return start,pack_planned_execution(tuple(branches),tuple(sizes))

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
        rows = stratified_starts(pools,args.starts,generator)
        optimizer.zero_grad(set_to_none=True)
        total = 0.
        for group in rows.split(args.microbatch):
            start, memory = context(group, audit=step==0)
            loss = imagine(start,memory,primary,secondary,preview,wm,geometry,observer,tail,step%3,args.horizon)
            weighted = loss*len(group)/len(rows)
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
                             seconds=perf_counter()-started, microbatch=args.microbatch,
                             peak_cuda_mib=torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else None,
                             unique_native_cases=len(rows[:,1].unique()),
                             h8d_starts=int((rows[:,1]%2).sum()),
                             retention=float(retention.detach())))
        print(evidence[-1],flush=True)
    with torch.no_grad():
        final = {k:p.clone() for k,p in params.items()}
        for label, budget in [('conservative',.020),('moderate',.035)]:
            # Use a distinct half proposal even when the full step fits both budgets.
            alpha = .5 if label=='conservative' else 1.
            for _ in range(25):
                for k,p in params.items():
                    p.copy_(torch.lerp(anchor[k],final[k],alpha))
                rms = float((actions(checks)-original).square().mean().sqrt())
                if rms <= budget:
                    break
                alpha *= .5
            if not 0 < rms <= budget:
                raise ValueError('No nonzero proposal within the action budget')
            state = {k:v.cpu().clone() for k,v in primary.state_dict().items()}
            if state.keys() != ckpt['actor'].keys() or any(not torch.equal(val,ckpt['actor'][k])
                    for k,val in state.items() if k not in RECOVERY_NAMES):
                raise ValueError('Frozen weights changed')
            torch.save(dict(ckpt, actor=state, actor_updates=ckpt['actor_updates']+args.updates),
                       args.output/(label+'.pt'))
            dump(args.output/(label+'.json'),dict(alpha=alpha,rms=rms,budget=budget))
    dump(args.output/'training.json',dict(updates=evidence, pools=[len(p) for p in pools], retained_cases=int(success.sum()),
                                         horizon=args.horizon, native=manifest['checkpoint_sha256']))


if __name__ == '__main__':
    main()
