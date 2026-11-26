"""Resident post-release native PPO with a training-only critic and paired evaluation."""
import math
import os
import time

import torch
from torch import nn

from quadsmash.rl.supermimic_physical_loop_loading_v1 import RECOVERY_NAMES
from scripts.distill_native_recovery import focus_columns


def reward_state(env):
    """Called at the native control boundary, before auto-reset clears terminal latches."""
    d = env.robot.data
    g = d.projected_gravity_b
    tilt = torch.maximum(torch.atan2(g[:,1], -g[:,2]).abs(),
                         torch.asin(g[:,0].clamp(-1,1)).abs())
    speed = d.root_link_lin_vel_w[:, :2].norm(dim=-1)
    angular = d.root_ang_vel_w.norm(dim=-1)
    leg = d.joint_vel[:,env.leg_ids].abs().amax(-1)
    upright = torch.exp(-(tilt / math.radians(12)).square())
    support = env.observed_foot_contact.float().mean(-1)
    height = d.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    net_distance = env.cfg.net_x_m - (d.body_link_pos_w[:, env.racket_body_id, 0] - env.scene.env_origins[:, 0])
    air_reward = (.015 * torch.exp(-(tilt / math.radians(18)).square())
                  + .01 * torch.exp(-angular.square()) + .005 * (net_distance / .3).clamp(0, 1))
    landing_reward = (.04 + .06 * upright + .04 * torch.exp(-((height - .34) / .06).square())
                      + .04 * support - .10 * (tilt / math.radians(15) - 1).relu().clamp_max(4).square()
                      - .01 * sum((x / limit - 1).relu().clamp_max(5) for x, limit in
                                  ((speed, .25), (angular, .8), (leg, .8))))
    dense = torch.where(env.landed_once, landing_reward, air_reward)
    r = env.contact_recovery
    return (dense, r.stable_run > 0, r.longest_stable.clamp(max=.8), env.body_ground_seen.clone(),
            env.rally.robot_net_fault.clone(), r.relaxed_success.clone(), r.stable_seen.clone())


def gae(reward, value, active, done, gamma=.995, lam=.95):
    adv = torch.zeros_like(reward)
    tail = torch.zeros_like(reward[0])
    nxt = torch.zeros_like(tail)
    for t in reversed(range(len(reward))):
        live = ~done[t]
        tail = (reward[t] + gamma*live*nxt - value[t] + gamma*lam*live*tail) * active[t]
        adv[t] = tail
        nxt = value[t] * active[t]
    return adv


def train(args, env, wrapper, obs, actor, companion, trained, cases, stop_request):
    from scripts.run_supermimic_recovery_baseline import dump
    from quadsmash.rl.supermimic_native_approach_reset import NativeApproachRestart, verify_resident_snapshot

    device = env.device
    n = env.num_envs
    assert n == 256 and len({case['case_id'] for case in cases}) == n
    assert sum(c['panel']=='A' for c in cases)==128
    for name, param in actor.named_parameters():
        param.requires_grad_(name in RECOVERY_NAMES)
    actor.install_release_arm_reference()
    companion.install_release_arm_reference()
    companion.release_arm_reference.load_state_dict(actor.release_arm_reference.state_dict())
    rec = actor.actor.recovery
    critic = nn.Sequential(nn.Linear(794,256), nn.LayerNorm(256), nn.Tanh(),
                           nn.Linear(256,128), nn.Tanh(), nn.Linear(128,1)).to(device)
    vf = torch.optim.Adam(critic.parameters(), lr=3e-4)
    optimizer = torch.optim.Adam(rec.parameters(), lr=2e-5)
    anchor = {k:p.detach().clone() for k,p in rec.named_parameters()}
    first_iteration = 1
    best_score = None
    if args.recovery_ppo_resume:
        saved = torch.load(args.recovery_ppo_resume, map_location=device, weights_only=False)
        if any(not torch.equal(v.to(device), saved['recovery'][k]) for k,v in rec.state_dict().items()):
            raise ValueError('PPO optimizer checkpoint and deployed Recovery differ')
        critic.load_state_dict(saved['critic'])
        vf.load_state_dict(saved['critic_adam'])
        optimizer.load_state_dict(saved['actor_adam'])
        anchor = saved['anchor']
        best_score = saved['best_score']
        first_iteration = saved['iteration'] + 1
    held = torch.arange(n, device=device) % 16 < 2
    companion.actor.recovery.load_state_dict(rec.state_dict())

    def request(x, gate):
        release = env.rally.release_time
        mean = actor._gated_distribution(x, gate, release_time=release).loc
        mean[1::2] = companion._gated_distribution(x[1::2], gate[1::2], release_time=release[1::2]).loc
        return mean

    @torch.no_grad()
    def advance_prefix(observation):
        for _ in range(197):
            if stop_request['signal']:
                raise InterruptedError(stop_request['signal'])
            gate = env.rally.released & env.rally.had_contact
            if gate.any():
                raise ValueError('Prefix already contains a release')
            observation, _, done, _ = wrapper.step(request(observation['policy'], gate).clamp(-1, 1))
            if done.any():
                raise ValueError('Unexpected prefix termination')
        return observation

    def restart_ground():
        # Explicitly discard unfinished prefixes and auto-reset successor episodes.
        # The terminal recorder only scores real done transitions in wrapper.step.
        env._episode_has_stepped.zero_()
        return advance_prefix(wrapper.reset()[0])

    root = args.output
    obs = advance_prefix(obs)
    snapshot = None
    if args.recovery_ppo_iterations > 1 and not args.recovery_ground_prefix:
        obs, audit = verify_resident_snapshot(
            env, wrapper, obs, lambda o: request(o['policy'], env.rally.released & env.rally.had_contact).clamp(-1, 1))
        dump(root/'snapshot_audit.json', audit)
        if audit['passed']:
            snapshot = NativeApproachRestart.capture(env)
        else:
            obs = restart_ground()
    initial = obs['policy'].detach().clone()
    for iteration in range(first_iteration, first_iteration + args.recovery_ppo_iterations):
        if iteration != first_iteration and snapshot is None:
            obs = restart_ground()
        else:
            if iteration != first_iteration:
                snapshot.restore(env)
            obs = {'policy': initial.clone()}
        companion.actor.recovery.load_state_dict(rec.state_dict())
        torch.manual_seed(args.seed + iteration)
        folder = root / f'iteration_{iteration:02d}'
        folder.mkdir()
        fraction = min(iteration-1, 9)/9
        late = min(max(iteration-10, 0), 10)/10
        sigma = initial.new_tensor([.11-.05*fraction-.01*late]*12 + [.06-.03*fraction-.005*late]*6)
        best_score = update(env, wrapper, obs, actor, trained, cases, stop_request, request, held,
                            critic, vf, optimizer, anchor, sigma, iteration, folder, root, best_score)
        training_state = dict(critic=critic.state_dict(), actor_adam=optimizer.state_dict(),
                              critic_adam=vf.state_dict(), recovery=rec.state_dict(), anchor=anchor,
                              iteration=iteration, best_score=best_score)
        torch.save(training_state, root/'optimizer.tmp')
        os.replace(root/'optimizer.tmp', root/'optimizer.pt')


def update(env, wrapper, obs, actor, trained, cases, stop_request, request, held, critic, vf,
           optimizer, anchor, sigma, iteration, folder, root, best_score):
    from scripts.run_supermimic_recovery_baseline import dump
    start = time.monotonic()
    device, n, rec = env.device, env.num_envs, actor.actor.recovery
    rollout = []
    samples = []
    age = torch.zeros(n,device=device,dtype=torch.long)
    lane_counts = torch.zeros_like(age)
    closed = torch.zeros(n,device=device,dtype=torch.bool)
    prev_run = torch.zeros(n,device=device)
    prev_body = closed.clone()
    prev_net = closed.clone()
    record_offset = len(env.rally_records)
    episode_ids = env.episode_index.clone()
    measured = {key: torch.zeros_like(closed) for key in ('body','net','strict','relaxed','contact')}
    discarded = 0
    for tick in range(197, 800):
        if stop_request['signal']:
            raise InterruptedError(stop_request['signal'])
        with torch.no_grad():
            x = obs['policy']
            gate = env.rally.had_contact & env.rally.released & ~closed
            eligible = gate & (age >= 4) & ~held
            use = eligible & ~prev_body & ~prev_net
            discarded += int((eligible & ~use).sum())
            age = torch.where(gate,age+1,0)
            # Both Approach versions are frozen. Recovery weights are identical during collection.
            mean = request(x, gate)
            z = actor.actor.normalizer(x)[use]
            epsilon = torch.randn_like(mean[use]) * sigma
            raw = mean.clone()
            raw[use] += epsilon
            if len(z):
                samples.append((z.cpu(),epsilon.cpu(),rec(z).cpu()))
            value = torch.zeros_like(prev_run)
            value[use] = critic(z.clamp(-10,10)).flatten()
            obs,_,dones,_ = wrapper.step(raw.clamp(-1,1))
            done = dones.bool().reshape(-1)
            dense,stable_tick,longest,body,net,relaxed,strict = env.recovery_ppo_boundary
            new_relaxed = relaxed & ~measured['relaxed']
            new_strict = strict & ~measured['strict']
            for key, event in [('body',body),('net',net),('strict',strict),('relaxed',relaxed),('contact',gate)]:
                measured[key] |= event & ~closed
            reward = (dense + .06 * stable_tick + 3. * (longest-prev_run)/.8
                      - 15.*(body & ~prev_body) - 20.*(net & ~prev_net)
                      + (~body)*(~net)*(2.*new_relaxed + 6.*new_strict))
            rollout.append((reward*use,value,use,done | body | net))
            lane_counts += use
            closed |= done
            prev_run,prev_body,prev_net = longest,body,net
            if tick % 100 == 99 or closed.all():
                progress = dict(control_ticks=tick+1, completed=int(closed.sum()), samples=int(lane_counts.sum()),
                                elapsed_s=time.monotonic()-start)
                dump(root/'progress.json',dict(progress, iteration=iteration, discarded=discarded))
                print(progress,flush=True)
            if closed.all():
                break
    if not closed.all() or not samples:
        raise ValueError('Incomplete native PPO rollout')
    ids = episode_ids.tolist()
    if any(row['numerical_failures'] for row in env.rally_records[record_offset:]
           if row['episode_index'] == ids[row['env_id']]):
        raise ValueError('Native numerical failure')
    rewards,values,active,done = [torch.stack(col) for col in zip(*rollout)]
    z,eps,old = [torch.cat(col) for col in zip(*samples)]
    validation = {k:int(v[held].sum()) for k,v in measured.items()}
    validation['strict'] = int((held & measured['strict'] & ~measured['body'] & ~measured['net']).sum())
    validation['relaxed'] = int((held & measured['relaxed'] & ~measured['body'] & ~measured['net']).sum())
    score = (-validation['net'], validation['strict'], validation['relaxed'], -validation['body'])
    if best_score is None or score > best_score:
        torch.save(trained, root/'best.tmp')
        os.replace(root/'best.tmp', root/'best.pt')
        best_score = score
    mc = gae(rewards,torch.zeros_like(values),active,done,lam=1.)[active].cpu()
    for _ in range(6):
        for ix in torch.randperm(len(z)).split(1024):
            pred = critic(z[ix].to(device).clamp(-10,10)).flatten()
            vf.zero_grad()
            nn.functional.smooth_l1_loss(pred,mc[ix].to(device)).backward()
            vf.step()
    with torch.no_grad():
        values[active] = torch.cat([critic(chunk.to(device).clamp(-10,10)).flatten() for chunk in z.split(2048)])
    adv = gae(rewards,values,active,done)[active].cpu()
    returns = adv + values[active].cpu()
    adv = (adv-adv.mean())/adv.std().clamp_min(1e-6)
    if not all(torch.isfinite(v).all() for v in (z,eps,old,adv,returns)) or len(z)!=len(adv):
        raise ValueError('Nonfinite or unaligned PPO samples')
    col = focus_columns(z).to(device)
    col = torch.where(col >= 8, col, 0.)
    scale = actor.actor.recovery_scale
    def kl():
        with torch.no_grad():
            total = sum((((rec(chunk.to(device)).cpu()-o)*scale)/sigma.cpu()).square().sum().item()
                        for chunk,o in zip(z.split(2048),old.split(2048)))
        return total/(2*len(z))
    steps = 0
    for epoch in range(4):
        for ix in torch.randperm(len(z)).split(1024):
            batch = z[ix].to(device)
            shift = (rec(batch)-old[ix].to(device))*scale
            log_ratio = ((eps[ix].to(device)*shift-.5*shift.square())/sigma.square()).sum(-1)
            ratio = log_ratio.clamp(-12,12).exp()
            advantage = adv[ix].to(device)
            weight = dict(rec.named_parameters())['0.weight']
            prox = ((weight-anchor['0.weight']).square() * col).mean()
            loss = -torch.minimum(ratio*advantage,ratio.clamp(.85,1.15)*advantage).mean()+.1*prox
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(rec.parameters(),.5)
            optimizer.step()
            vloss = nn.functional.smooth_l1_loss(critic(batch.clamp(-10,10)).flatten(),returns[ix].to(device))
            vf.zero_grad()
            vloss.backward()
            vf.step()
            steps += 1
        if kl() >= .008:
            break
    accepted_kl = kl()
    halvings = 0
    while accepted_kl > .015 and halvings < 12:
        with torch.no_grad():
            for k,p in rec.named_parameters():
                p.lerp_(anchor[k],.5)
        halvings += 1
        accepted_kl = kl()
    if not 0 < accepted_kl <= .015:
        raise ValueError(f'Invalid final policy KL {accepted_kl}')
    state = {k:v.detach().cpu().clone() for k,v in actor.state_dict().items()}
    changed = sorted(k for k,v in trained['actor'].items() if not torch.equal(v,state[k]))
    if set(changed)!=set(RECOVERY_NAMES):
        raise ValueError(f'Unexpected policy changes {changed}')
    report = dict(progress, cases=n, active_cases=int((lane_counts>0).sum()), sigma_leg=float(sigma[0]), sigma_arm=float(sigma[-1]),
                  iteration=iteration, heldout_before_update=validation, discarded_after_fault=discarded,
                  minibatch_updates=steps, epochs=epoch+1, accepted_kl=accepted_kl, halvings=halvings,
                  frozen_verified=True, changed=changed, train_elapsed_s=time.monotonic()-start,
                  approach_frames=dict(final_best=int(lane_counts[::2].sum()),h8d=int(lane_counts[1::2].sum())))
    trained['actor'] = state
    trained['metadata'] = dict(trained['metadata'], recovery_ppo=report)
    torch.save(trained,folder/'policy.tmp')
    os.replace(folder/'policy.tmp',folder/'policy.pt')
    torch.save(trained, root/'policy.tmp')
    os.replace(root/'policy.tmp', root/'policy.pt')
    dump(folder/'training.json',report)
    dump(folder/'cases.json',[dict(case=c['case_id'],steps=int(k)) for c,k in zip(cases,lane_counts.tolist())])
    return best_score
