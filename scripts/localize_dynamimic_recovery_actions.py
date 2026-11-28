"""Distill BPTT Recovery corrections near native pre-fall states without adding inference parameters."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from quadsmash.rl.supermimic_physical_loop_loading_v1 import RECOVERY_NAMES
from quadsmash.rl.supermimic_policy_execution_v1 import resolve_contract
from quadsmash.rl.supermimic_v1 import load_ppo_initialized_actor
from scripts.dynamimic_recovery_validation import policies
from scripts.fit_dynamimic_recovery_critic import root_termination
from scripts.fit_dynamimic_recovery_retention import change_metrics
from scripts.fit_dynamimic_recovery_wm import read_data
from scripts.run_dynamimic_recovery_loop import retention_candidates
from scripts.train_dynamimic_recovery_bptt import focus_cost, mixed_request, recovery_start_groups, stratified_starts


def observable_features(obs):
    history = obs[:, :759].reshape(-1, 11, 69)
    body = torch.cat((history[:, :, :45], history[:, :, 57:61]), 2)
    return torch.cat((body[:, 0], body[:, 0]-body[:, 4], body[:, 4]-body[:, 10]), 1)


def sustained_scores(probability, tick, lane, shape):
    dense = probability.new_full(shape, -1.)
    dense[tick, lane] = probability
    # Any missing or ineligible step resets this causal 120 ms window.
    windows = F.pad(dense, (0, 0, 11, 0), value=-1.).unfold(0, 12, 1)
    return windows.amin(-1)[tick, lane]


def split_approaches(pools, lanes):
    paired = [(pool[lanes[pool] % 2 == 0], pool[lanes[pool] % 2 == 1]) for pool in pools]
    if any(len(panel) == 0 for pair in paired for panel in pair):
        raise ValueError('Each sampled stratum needs both Approach policies')
    return paired


def sample(pools, sizes):
    batches = []
    for pair, size in zip(pools, sizes, strict=True):
        if size % 2:
            raise ValueError('Equal Approach allocation requires an even batch')
        for panel in pair:
            batches.append(panel[torch.randint(len(panel), (size // 2,))])
    return torch.cat(batches)


def landing_teacher_masks(tick, lane, release_age, landing, fault, end):
    # Events are sampled at the end of each control tick; observations are
    # sampled at its start. Future failure is only a native teacher label.
    first_ground_state = landing[lane] + 1
    eligible = ((fault[lane] < end) & (landing[lane] < end)
                & (tick <= fault[lane]) & (release_age[tick, lane] >= .05))
    return (eligible & (tick < first_ground_state),
            eligible & (tick >= first_ground_state) & (tick < first_ground_state + 24))


def fit_warning(obs, tick, lane, fault, clean, length):
    train = lane % 16 >= 4
    features = observable_features(obs)
    x = ((features-features[train].mean(0))/features[train].std(0).clamp_min(.05)).clamp(-8,8)
    lead = fault[lane]-tick
    fall = (fault[lane] < length) & (lead >= 0) & (lead <= 48)
    strata = (fall & (lead >= 16), fall & (lead < 16), clean[lane], ~clean[lane] & ~fall)
    pools = [(train & mask).nonzero().flatten() for mask in strata]
    if any(not len(p) for p in pools):
        raise ValueError('Native archive lacks pre-fall or retention coverage')
    pools = split_approaches(pools,lane)
    net = nn.Sequential(nn.Linear(x.shape[1],48), nn.Tanh(), nn.Linear(48,1))
    optimizer = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.01)
    for _ in range(300):
        idx = sample(pools, (96,32,96,32))
        loss = F.binary_cross_entropy_with_logits(net(x[idx]).flatten(), fall[idx].float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        warning = sustained_scores(net(x).flatten().sigmoid(), tick, lane, (length,len(fault)))
    peak = warning.new_full(fault.shape,-1.).scatter_reduce(0,lane,warning,reduce='amax')
    clean_train = clean & (torch.arange(len(fault)) % 16 >= 4)
    if not clean_train.any():
        raise ValueError('No clean training cases for the warning threshold')
    threshold = peak[clean_train].quantile(.9)
    return warning >= threshold, float(threshold)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('native','checkpoint','companion','retention','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--landing-assist',action='store_true')
    parser.add_argument('--anchor-airborne', action='store_true',
                        help='Keep PPO01 airborne actions and distill the landed correction')
    args = parser.parse_args()
    if args.anchor_airborne and not args.landing_assist:
        parser.error('--anchor-airborne requires --landing-assist')
    data, manifest = read_data(args.native, ('active','obs','release_time','done','relaxed','strict','net','request'))
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    behavior = digest(args.checkpoint)
    companion = digest(args.companion)
    if behavior != manifest['checkpoint_sha256'] or companion != manifest['companion_sha256']:
        raise ValueError('Recovery archive was collected by different policies')
    candidates = retention_candidates(args.retention, behavior, companion)
    base = torch.load(args.checkpoint, weights_only=False, mmap=True, map_location='cpu')
    source = Path(base['metadata']['init_policy_checkpoint']['path'])
    if digest(source) != base['metadata']['init_policy_checkpoint']['sha256']:
        raise ValueError('Frozen initialization differs')
    data['root_terminated'] = root_termination(data['done'], data['after'][...,2]+torch.tensor(manifest['origins'])[:,2])
    active = data['active'] & (data['obs'][...,775]-data['release_time']>=.04)
    t,l = active.nonzero().T
    obs = data['obs'][t,l]
    length = len(active)
    first_fault = torch.where(data['live'] & data['body'],torch.arange(length)[:,None],length).amin(0)
    clean = ((data['relaxed']|data['strict']) & data['live']).any(0) & ~(
        (data['body']|data['net']|data['root_terminated']) & data['live']).any(0)
    torch.manual_seed(323)
    warned, threshold = fit_warning(obs,t,l,first_fault,clean,length)
    train = l%16 >= 4
    target = warned
    phases = {}
    if args.landing_assist:
        landing = torch.where(data['live'] & data['landed'],torch.arange(length)[:,None],length).amin(0)
        air, impact = landing_teacher_masks(t,l,data['obs'][...,775]-data['release_time'],landing,first_fault,length)
        target = (warned | air | impact) & ~clean[l]
        phases = dict(air_preparation=air, first_support=impact)
    sizes = (128,64,64)
    if args.anchor_airborne:
        airborne = t <= landing[l]
        target = (warned | impact) & ~clean[l] & ~airborne
        masks = (target, clean[l] & ~airborne, ~clean[l] & ~target & ~airborne, airborne)
        sizes = (96,64,32,64)
        phases['all_airborne'] = airborne
    else:
        masks = (target, clean[l], ~clean[l]&~target)
    pools = [(train & mask).nonzero().flatten() for mask in masks]
    if any(not len(p) for p in pools):
        raise ValueError('Insufficient native states for conditional correction')
    pools = split_approaches(pools,l)
    release_age = data['obs'][...,775] - data['release_time']
    early_tick, early_lane = (data['active'] & (release_age < .04)
                              & (torch.arange(len(clean)) % 16 >= 4)).nonzero().T
    early_pool = split_approaches([torch.arange(len(early_lane))],early_lane)
    ticks = sorted(int(p.stem.split('_')[-1]) for p in (args.native/'trajectories').glob('g07_*.pt'))
    held = torch.arange(len(clean))%16 < 2
    checks = stratified_starts(recovery_start_groups(data,ticks,held),128,torch.Generator().manual_seed(9917))
    args.output.mkdir(parents=True,exist_ok=False)
    report = dict(behavior=behavior, companion=companion, threshold=threshold,
                  action_batch_per_approach=128, early_leg_anchor_batch_per_approach=32,
                  early_leg_anchor_cases=int(early_lane.unique().numel()), focus_cost_weight=.1,
                  train_cases=int(l[train].unique().numel()), warning_train_cases=int(l[train&warned].unique().numel()),
                  landing_assist=args.landing_assist, anchor_airborne=args.anchor_airborne,
                  phase_targets={key:dict(train_cases=len(l[mask&train].unique()),
                                          train_ticks=int((train&mask).sum()),
                                          previously_unwarned_ticks=int((train&mask&~warned).sum()))
                                 for key,mask in phases.items()}, candidates={})
    for label,path,budget in zip(('conservative','moderate'),candidates,(.020,.035),strict=True):
        parent,other,teacher,_ = policies(args.checkpoint,path,args.companion,lambda:load_ppo_initialized_actor(source,'cpu')[0])
        student,_ = load_ppo_initialized_actor(source,'cpu')
        student.load_state_dict(base['actor'], strict=True)
        student.configure_approach_gate(resolve_contract(base)[0].get('arm_preparation_seconds'))
        student.eval()
        for name,param in student.named_parameters():
            param.requires_grad_(name in RECOVERY_NAMES)
        with torch.no_grad():
            z = parent.actor.normalizer(obs)
            nominal = parent.actor.recovery(z)
            delta = teacher.actor.recovery(z)-nominal
            early_z = parent.actor.normalizer(data['obs'][early_tick,early_lane])
            early_target = parent.actor.recovery(early_z)[:,:12]
        first_layer = dict(student.named_parameters())['actor.recovery.0.weight']
        anchor = first_layer.detach().clone()
        optimizer = torch.optim.Adam(student.actor.recovery.parameters(),lr=.00012)
        for _ in range(600):
            idx = sample(pools,sizes)
            prediction = student.actor.recovery(z[idx])-nominal[idx]
            early = sample(early_pool,(64,))
            leg_error = F.mse_loss(student.actor.recovery(early_z[early])[:,:12], early_target[early])
            loss = 100*(F.mse_loss(prediction,target[idx,None]*delta[idx]) + leg_error)
            loss = loss + .1*focus_cost(first_layer,anchor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            metrics = {}
            for key,mask in [('warning',warned),('clean',clean[l]),('no_warning',~warned), *phases.items()]:
                group = (held[l]&mask).nonzero().flatten()
                if len(group):
                    actual = student.actor.recovery(z[group])-nominal[group]
                    metrics[key] = dict(cases=int(l[group].unique().numel()),ticks=len(group),
                                        **change_metrics(delta[group],actual,l[group]))
            ct,cl = checks.T
            args_in = (other,data['obs'][ct,cl],cl%2==1,data['release_time'][ct,cl])
            before = mixed_request(parent,*args_in).clamp(-1,1)
            replay_error = float((before-data['request'][ct,cl]).abs().max())
            after = mixed_request(student,*args_in).clamp(-1,1)
            rms = float((after-before).square().mean().sqrt())
            if not 0 < rms <= budget or replay_error > 2e-6:
                raise ValueError('Native replay or action budget mismatch')
        state = student.state_dict()
        if {k for k,v in state.items() if not torch.equal(v,base['actor'][k])} != set(RECOVERY_NAMES):
            raise ValueError('Localization changed an unexpected tensor')
        dest = args.output/(label+'.pt')
        torch.save(dict(base,actor=state),dest)
        report['candidates'][label] = dict(source=digest(path), after=digest(dest),rms=rms,budget=budget,
                                          replay_max_error=replay_error,held_head_changes=metrics)
    (args.output/'fit.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
