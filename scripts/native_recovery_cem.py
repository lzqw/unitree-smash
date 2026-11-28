"""Resident native search for a post-release Recovery feedback teacher."""
import hashlib
import time

import torch


def panel_layout(bank, generation):
    """Eight feedback candidates share eight Panel A and eight Panel B serves."""
    offset = (generation % 16) * 8
    selected = [c for p in ('A', 'B') for c in [x for x in bank if x['panel'] == p][offset:offset+8]]
    if len({c['case_id'] for c in selected}) != 16:
        raise ValueError('Missing independent training launches')
    return selected * 8


def correction(obs, height, landed, age, coefficients):
    """Six independent task features, four learned gains each, in policy joint order."""
    gains = coefficients.reshape(-1, 6, 4)
    roll = torch.atan2(-obs[:, 7], -obs[:, 8])
    left = ((-roll).relu() / .30).clamp_max(1)
    rolling = ((-obs[:, 3]).relu() / 2).clamp_max(1)
    low = ((.28 - height) / .15).clamp(0, 1)
    pitch = (obs[:, 6] / .5).clamp(-1, 1)
    support = landed.to(obs)
    delta = torch.zeros_like(obs[:, :18])
    def add(indices, feature, group):
        delta[:, indices] += feature[:, None] * gains[:, group]
    add([10, 6, 5, 0], support, 0)
    add([10, 8, 2, 0], support * left, 1)
    add([10, 8, 2, 0], support * rolling, 2)
    add([8, 9, 10, 11], support * low, 3)
    add([4, 5, 6, 7], support * pitch, 4)
    air = (~landed).to(obs) * ((.28-age)/.08).clamp(0, 1)
    add([10, 11, 6, 7], air, 5)
    return delta.clamp(-2, 2) * ((age-.05)/.04).clamp(0, 1)[:, None]


@torch.no_grad()
def search(args, env, wrapper, obs, actor, companion, bank, protocol, stop):
    from quadsmash.rl.supermimic_terminal_scoring import score_terminal_episode
    from scripts.run_supermimic_recovery_baseline import dump

    actor.install_release_arm_reference()
    companion.install_release_arm_reference()
    companion.release_arm_reference.load_state_dict(actor.release_arm_reference.state_dict())
    companion.actor.recovery.load_state_dict(actor.actor.recovery.state_dict())
    root, device = args.output, env.device
    start_sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    mu = torch.zeros(24, device=device)
    sigma = torch.full_like(mu, .28)
    mu[0], sigma[0] = .65, .55
    rng = torch.Generator(device=device).manual_seed(args.seed+8301)
    for generation in range(args.recovery_cem_generations):
        cases = panel_layout(bank, generation)
        if generation:
            env.cases = cases
            env._episode_has_stepped.zero_()
            env.episode_index.zero_()
            env.sweep_misses.zero_()
            env.sweep_unresolved.zero_()
            for buffer in (env.rally_records, env.episode_records, env.racket_plane_records,
                           env.initial_states):
                buffer.clear()
            obs, _ = wrapper.reset()
        group = root / f'generation_{generation+1:02d}'
        group.mkdir()
        directions = torch.randn(3, 24, device=device, generator=rng) * sigma
        params = torch.cat((torch.zeros_like(mu)[None], mu[None], mu+directions, mu-directions)).clamp(-2, 2)
        lane_params = params.repeat_interleave(16, dim=0)
        ended = torch.zeros(128, device=device, dtype=torch.bool)
        episode = env.episode_index.tolist()
        records, frames = {}, []
        cursor = len(env.rally_records)
        onset = torch.full((128,), -1, device=device, dtype=torch.long)
        onset_obs = torch.zeros_like(obs['policy'])
        calf_counts = torch.zeros(8, 3, dtype=torch.long, device=device)
        calf_change = torch.zeros(8, device=device)
        started = time.monotonic()
        for tick in range(800):
            if stop['signal']:
                raise InterruptedError(stop['signal'])
            x = obs['policy']
            gate = env.rally.had_contact & env.rally.released
            released = env.rally.release_time
            age = x[:, 775] - released
            mean = actor._gated_distribution(x, gate, release_time=released).loc
            mean[1::2] = companion._gated_distribution(x[1::2], gate[1::2], release_time=released[1::2]).loc
            height = env.robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
            active = gate & ~ended & (age > .05) & ~env.body_ground_seen & ~env.rally.robot_net_fault
            first = active & (onset < 0)
            onset = torch.where(first, tick, onset)
            onset_obs = torch.where(first[:, None], x, onset_obs)
            delta = correction(x, height, env.landed_once, age, lane_params) * active[:, None]
            request = (mean + delta).clamp(-1, 1)
            calf_before = mean[:,10].clamp(-1, 1)
            clipped = active & (mean[:,10] < -1)
            calf_counts[:,0] += active.reshape(8,16).sum(1)
            calf_counts[:,1] += clipped.reshape(8,16).sum(1)
            calf_counts[:,2] += (clipped & (request[:,10] > -1)).reshape(8,16).sum(1)
            calf_change += ((request[:,10]-calf_before) * active).reshape(8,16).sum(1)
            if tick % 4 == 0 and active.any():
                frames.append((x.cpu().clone(), delta.cpu(), active.cpu()))
            obs, _, done, _ = wrapper.step(request)
            ended |= done.bool().reshape(-1)
            for raw in env.rally_records[cursor:]:
                lane = int(raw['env_id'])
                if raw['episode_index'] != episode[lane] or lane in records:
                    continue
                if raw['numerical_failures'] or raw['case_id'] != cases[lane]['case_id']:
                    raise ValueError('Native terminal identity or numeric failure')
                row = score_terminal_episode(raw, protocol)
                ground = bool(raw['body_ground_seen'])
                net = bool(raw['robot_net_fault'])
                clean = bool(row['qualified_airborne_string_contact'] and not ground and not net and not raw['jump']['terminated'])
                records[lane] = dict(case=raw['case_id'], body=ground, net=net, ball=row['ball_side_complete_except_recovery'],
                                     strict=bool(clean and raw['contact_flight_recovery']['success']),
                                     relaxed=bool(clean and row['physical_recovery_success']),
                                     run=min(.8, raw['contact_flight_recovery']['recovery_s']))
            cursor = len(env.rally_records)
            if tick % 64 == 63:
                dump(root/'progress.json',dict(generation=generation+1,tick=tick+1,episodes=len(records)))
            if ended.all():
                break
        if len(records) != 128 or not frames:
            raise ValueError('Incomplete resident native CEM generation')
        tape, teachers, aggregates = [], [], []
        for j in range(8):
            rows = [records[j*16+i] for i in range(16)]
            aggregates.append({k:sum(row[k] for row in rows) for k in ('body','net','ball','strict','relaxed','run')})
        common = (onset == onset[:16].repeat(8)) & (onset >= 0)
        obs_error = (onset_obs - onset_obs[:16].repeat(8,1)).abs().amax(-1)
        matched = (common & (obs_error <= 1e-4)).tolist()
        for i in range(16):
            base = records[i]
            choices = [j*16+i for j in range(1,8) if matched[j*16+i] and base['body'] and records[j*16+i]['strict']
                       and records[j*16+i]['ball'] >= base['ball']]
            chosen = min(choices, key=lambda lane:float(params[lane//16].norm())) if choices else None
            if chosen is not None or base['strict']:
                lane = i if chosen is None else chosen
                valid = [(x[lane],d[lane]) for x,d,a in frames if a[lane]]
                if valid:
                    x, d = map(torch.stack,zip(*valid))
                    tape.append(dict(case=base['case'],lane=lane,anchor=chosen is None,obs=x,delta_raw=d))
                    if chosen is not None:
                        teachers.append(base['case'])
        def fitness(a):
            return 8*a['strict'] + 3*a['relaxed'] + 5*a['ball'] - 6*a['body'] - 20*a['net'] + a['run']
        elites = sorted(range(8), key=lambda j:fitness(aggregates[j]), reverse=True)[:3]
        mu = .4*mu + .6*params[elites].mean(0)
        sigma = (.5*sigma + .5*params[elites].std(0,unbiased=False)).clamp(.06,.70)
        torch.save(tape, group/'examples.pt')
        dump(group/'results.json',dict(checkpoint=start_sha,generation=generation+1,params=params.tolist(),
             onset_matched=matched,onset_obs_max_abs=obs_error.tolist(),
             rl_calf=dict(active_ticks=calf_counts[:,0].tolist(),lower_saturated_ticks=calf_counts[:,1].tolist(),
                          released_from_clip_ticks=calf_counts[:,2].tolist(),
                          mean_executed_request_change=(calf_change/calf_counts[:,0].clamp_min(1)).tolist()),
             scores=aggregates,elites=elites,teachers=teachers,episodes=list(records[i] for i in range(128)),
             elapsed_s=time.monotonic()-started))


def distill(root, checkpoint):
    """Learn confirmed rescues on teacher-branch observations; anchor nominal successes."""
    import json
    from pathlib import Path
    from quadsmash.rl.supermimic_v1 import load_ppo_initialized_actor
    from quadsmash.rl.supermimic_physical_loop_loading_v1 import RECOVERY_NAMES
    from scripts.run_supermimic_recovery_baseline import dump

    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    examples = []
    for path in sorted(root.glob('generation_*/results.json')):
        info = json.loads(path.read_text())
        if info['checkpoint'] != digest:
            raise ValueError('CEM teacher checkpoint mismatch')
        examples.extend(torch.load(path.parent/'examples.pt', weights_only=True, map_location='cpu'))
    pools = [[e for e in examples if e['anchor']==flag] for flag in (False,True)]
    support = dict(teacher_cases=len({e['case'] for e in pools[0]}), anchor_cases=len({e['case'] for e in pools[1]}))
    if not all(pools):
        dump(root/'distillation.json',dict(support,state='missing_teachers_or_anchors'))
        return
    ckpt = torch.load(checkpoint,map_location='cpu',weights_only=False)
    actor,_ = load_ppo_initialized_actor(Path(ckpt['metadata']['init_policy_checkpoint']['path']),'cpu')
    actor.load_state_dict(ckpt['actor'])
    for name,param in actor.named_parameters():
        param.requires_grad_(name in RECOVERY_NAMES)
    rec = actor.actor.recovery
    with torch.no_grad():
        for e in examples:
            e['z'] = actor.actor.normalizer(e.pop('obs'))
            e['target'] = rec(e['z']) + e.pop('delta_raw')/actor.actor.recovery_scale
    opt = torch.optim.Adam(rec.parameters(),lr=8e-5)
    rng = torch.Generator().manual_seed(7245)
    for _ in range(240):
        losses = []
        for pool in pools:
            errs = []
            for j in torch.randint(len(pool),(8,),generator=rng).tolist():
                e = pool[j]
                idx = torch.randint(len(e['z']),(16,),generator=rng)
                errs.append((rec(e['z'][idx])-e['target'][idx]).square().mean())
            losses.append(torch.stack(errs).mean())
        opt.zero_grad()
        sum(losses).backward()
        torch.nn.utils.clip_grad_norm_(rec.parameters(),.5,error_if_nonfinite=True)
        opt.step()
    state = actor.state_dict()
    changed = {k for k,v in ckpt['actor'].items() if not torch.equal(v,state[k])}
    if changed != set(RECOVERY_NAMES) or state.keys()!=ckpt['actor'].keys():
        raise ValueError('Unexpected Recovery parameter change')
    torch.save(dict(ckpt,actor=state),root/'candidate.pt')
    dump(root/'distillation.json',dict(support,state='candidate_pending_fixed128',changed=sorted(changed)))
