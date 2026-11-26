"""One on-policy PhysX rollout and clipped Recovery PPO update. Critic is training-only."""
import json
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
    excess = sum((x / limit - 1.).relu().clamp_max(5.) for x,limit in
                 ((tilt,math.radians(12)), (speed,.25), (angular,.8), (leg,.8)))
    support = env.observed_foot_contact.sum(-1) / 4.
    height = d.root_pos_w[:,2] - env.scene.env_origins[:,2]
    dense = env.landed_once * (support - .25*excess - ((.25-height)/.1).relu())
    r = env.contact_recovery
    return (dense, r.stable_run.clamp(max=.8), env.body_ground_seen.clone(),
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
    from quadsmash.rl.supermimic_policy_execution_v1 import policy_forward, POSTHIT
    start = time.monotonic()
    device = env.device
    n = env.num_envs
    assert n == 256 and len({case['case_id'] for case in cases}) == n
    assert sum(c['panel']=='A' for c in cases)==128
    for name,param in actor.named_parameters():
        param.requires_grad_(name in RECOVERY_NAMES)
    actor.install_release_arm_reference()
    critic = nn.Sequential(nn.Linear(794,128),nn.Tanh(),nn.Linear(128,64),nn.Tanh(),nn.Linear(64,1)).to(device)
    nn.init.zeros_(critic[-1].weight)
    nn.init.zeros_(critic[-1].bias)
    rec = actor.actor.recovery
    anchor = {k:p.detach().clone() for k,p in rec.named_parameters()}
    sigma = .05
    rollout = []
    samples = []
    age = torch.zeros(n,device=device,dtype=torch.long)
    lane_counts = torch.zeros_like(age)
    closed = torch.zeros(n,device=device,dtype=torch.bool)
    prev_run = torch.zeros(n,device=device)
    prev_body = closed.clone()
    prev_net = closed.clone()
    for tick in range(800):
        if stop_request['signal']:
            raise InterruptedError(stop_request['signal'])
        with torch.no_grad():
            x = obs['policy']
            gate = env.rally.had_contact & env.rally.released & ~closed
            use = gate & (age >= 4)
            age = torch.where(gate,age+1,0)
            # Both Approach versions are frozen. Recovery weights are identical during collection.
            mean = actor._gated_distribution(x,gate,release_time=env.rally.release_time).loc
            mean[1::2] = companion._gated_distribution(x[1::2],gate[1::2]).loc
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
            dense,stable,body,net,relaxed,strict = env.recovery_ppo_boundary
            reward = (.01*dense*(~body) + 2.5*(stable-prev_run)/.8
                      - 12.*(body & ~prev_body) - 20.*(net & ~prev_net)
                      + done*(~body)*(~net)*(1.5*relaxed + 5.*strict))
            rollout.append((reward*use,value,use,done))
            lane_counts += use
            closed |= done
            prev_run,prev_body,prev_net = stable,body,net
            if tick % 100 == 99 or closed.all():
                progress = dict(control_ticks=tick+1, completed=int(closed.sum()), samples=int(lane_counts.sum()),
                                elapsed_s=time.monotonic()-start)
                dump(args.output/'progress.json',progress)
                print(progress,flush=True)
            if closed.all():
                break
    if not closed.all() or not samples:
        raise ValueError('Incomplete native PPO rollout')
    if any(row['numerical_failures'] for row in env.rally_records if row['episode_index']==0):
        raise ValueError('Native numerical failure')
    rewards,values,active,done = [torch.stack(col) for col in zip(*rollout)]
    z,eps,old = [torch.cat(col) for col in zip(*samples)]
    vf = torch.optim.Adam(critic.parameters(),lr=3e-4)
    # A fresh critic has no prior value estimates. Fit on this rollout's real
    # discounted rewards before using its bootstrapped values for GAE.
    mc = gae(rewards,torch.zeros_like(values),active,done,lam=1.)[active].cpu()
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
    optimizer = torch.optim.Adam(rec.parameters(),lr=2e-5)
    col = focus_columns(z).to(device)
    scale = actor.actor.recovery_scale
    def kl():
        with torch.no_grad():
            total = sum(((rec(chunk.to(device)).cpu()-o)*scale).square().sum().item()
                        for chunk,o in zip(z.split(2048),old.split(2048)))
        return total/(2*sigma**2*len(z))
    steps = 0
    for epoch in range(4):
        for ix in torch.randperm(len(z)).split(1024):
            batch = z[ix].to(device)
            shift = (rec(batch)-old[ix].to(device))*scale
            log_ratio = (eps[ix].to(device)*shift-.5*shift.square()).sum(-1)/sigma**2
            ratio = log_ratio.clamp(-12,12).exp()
            advantage = adv[ix].to(device)
            prox = sum(((p-anchor[k]).square()*(col if k=='0.weight' else 1.)).mean()
                       for k,p in rec.named_parameters())
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
    while accepted_kl > .012 and halvings < 12:
        with torch.no_grad():
            for k,p in rec.named_parameters():
                p.lerp_(anchor[k],.5)
        halvings += 1
        accepted_kl = kl()
    if not 0 < accepted_kl <= .012:
        raise ValueError(f'Invalid final policy KL {accepted_kl}')
    state = {k:v.detach().cpu().clone() for k,v in actor.state_dict().items()}
    changed = sorted(k for k,v in trained['actor'].items() if not torch.equal(v,state[k]))
    if set(changed)!=set(RECOVERY_NAMES):
        raise ValueError(f'Unexpected policy changes {changed}')
    report = dict(progress, cases=n, active_cases=int((lane_counts>0).sum()), sigma=sigma,
                  minibatch_updates=steps, epochs=epoch+1, accepted_kl=accepted_kl, halvings=halvings,
                  frozen_verified=True, changed=changed, train_elapsed_s=time.monotonic()-start,
                  approach_frames=dict(final_best=int(lane_counts[::2].sum()),h8d=int(lane_counts[1::2].sum())))
    trained['actor'] = state
    trained['metadata'] = dict(trained['metadata'], recovery_ppo=report)
    torch.save(trained,args.output/'policy.tmp')
    os.replace(args.output/'policy.tmp',args.output/'policy.pt')
    dump(args.output/'training.json',report)
    dump(args.output/'cases.json',[dict(case=c['case_id'],steps=int(k)) for c,k in zip(cases,lane_counts.tolist())])
