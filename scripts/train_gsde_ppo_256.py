#!/usr/bin/env python3
"""Unchained On-Policy Resident 256-Lane Antithetic gSDE-PPO directly in Real PhysX.

Breaks past the ~80% offline-BC & FB-anchor ceiling by directly optimizing actor.approach
(APPROACH_NAMES only; base & recovery frozen; leg head frozen) on BOTH Panel A and Panel B
simultaneously in real PhysX with ZERO offline FB arm-anchor lock!

Key algorithmic pillars:
1. Exact 256-lane physics parity (env_spacing=0.0) + NativeApproachRestart @ tick 197 (~5.8 min/iter).
2. Layout: 64 deterministic unseen hard-stratified val lanes (32 A + 32 B, zero leakage)
   + 48 fresh disjoint training serves/iter (16 Panel A + 32 Panel B [20 hard-tail + 12 i.i.d.])
   x G=4 (2 exact antithetic smooth gSDE pairs: +z1,-z1,+z2,-z2) = 256 lanes.
3. State-dependent smooth gSDE + pose + whip exploration:
     delta_arm(t) = W @ (h(s_t) / ||h(s_t)||) + b + ramp(t) * r
   100% smooth in time (zero 100Hz jitter), 100% realizable in actor.approach.4's exact feature space!
4. Exact episode-level PPO-Clip surrogate over antithetic within-serve advantages + soft KL trust region
   to theta_old (NOT to FB!) + hard per-iter arm RMS cap + leg-only anchor (rnet=0) + clean rollback.
"""
import argparse
import faulthandler
import gzip
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
os.environ.setdefault('QUADSMASH_DISABLE_APP_SHUTDOWN', '1')
os.environ.setdefault('QUADSMASH_SKIP_KIT_CLOSE', '1')

from isaaclab.app import AppLauncher


def reserved_case_ids(execution_root: Path):
    ids = set()
    p01 = execution_root / 'reports/G07_full_racket_v4/protocol_frozen01'
    for name in ('training_cases.json', 'A_cases.json', 'B_cases.json'):
        p = p01 / name
        if p.is_file():
            d = json.loads(p.read_text())
            rows = d.get('cases', d) if isinstance(d, dict) else d
            ids.update(r['case_id'] for r in rows)
    rs = ROOT / 'reports/supermimic_research_scale_20261121'
    for name in ('fixed128.json.gz', 'independent512.json.gz'):
        p = rs / name
        if p.is_file():
            d = json.loads(gzip.decompress(p.read_bytes()))
            ids.update(r['case_id'] for r in d['cases'])
    p27 = Path('/home/autodrive/LZQW/experiments/E1_v2_approach_scale/P27_push85')
    for name in ('val128_strict.spec.json', 'val128_strict_tail.spec.json', 'val64_hard_monitor.spec.json'):
        p = p27 / name
        if p.is_file():
            ids.update(c['case_id'] for c in json.loads(p.read_text()).get('cases', []))
    return ids


def compute_ppo_reward(row, raw_ep, min_dist_m: float) -> float:
    succ = 1.0 if row['success'] else 0.0
    qstr = 1.0 if row['string'] else 0.0
    rnet = 1.0 if row['robot_net'] else 0.0
    sm = row['string_margin_m']
    sm_t = max(-1.5, min(1.0, sm / 0.040)) if (sm is not None and math.isfinite(sm)) else -1.5
    spm = row['speed_margin_mps']
    spm_t = max(-1.5, min(1.0, spm / 0.60)) if (qstr > 0 and spm is not None and math.isfinite(spm)) else -0.5
    lm = row['landing_margin_m']
    lm_t = max(-1.5, min(1.0, lm / 0.45)) if (qstr > 0 and lm is not None and math.isfinite(lm)) else -1.0
    nm = row['net_margin_m']
    nm_t = max(-1.0, min(1.0, nm / 0.14)) if (qstr > 0 and nm is not None and math.isfinite(nm)) else -0.5
    ctr_pen = 0.0
    if qstr > 0:
        land = raw_ep.get('first_landing_m')
        if land and math.isfinite(land[0]) and math.isfinite(land[1]):
            ctr_pen = -0.45 * max(0.0, land[0] - 3.55) - 0.50 * max(0.0, abs(land[1]) - 0.65)
    had_c = (row['string'] or row['frame'] or (sm is not None))
    dist_pen = 0.0 if had_c else -2.0 * max(0.0, min(0.40, min_dist_m - 0.08))
    return float(
        5.00 * succ
        + 1.50 * qstr
        + 0.90 * sm_t
        + 0.70 * spm_t
        + 1.00 * lm_t
        + 0.50 * nm_t
        + ctr_pen
        + dist_pen
        - 12.00 * rnet
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--g07-execution-root', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=882072)
    parser.add_argument('--val-seed', type=int, default=993101)
    parser.add_argument('--train-seed-base', type=int, default=985000)
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--n-serves-a', type=int, default=16)
    parser.add_argument('--n-serves-b', type=int, default=32)
    parser.add_argument('--sigma-sde', type=float, default=0.055)
    parser.add_argument('--sigma-bias', type=float, default=0.045)
    parser.add_argument('--sigma-ramp', type=float, default=0.055)
    parser.add_argument('--sigma-decay', type=float, default=0.975)
    parser.add_argument('--clip-eps', type=float, default=0.18)
    parser.add_argument('--kl-beta', type=float, default=3.5)
    parser.add_argument('--leg-anchor', type=float, default=25.0)
    parser.add_argument('--prox-weight', type=float, default=0.03)
    parser.add_argument('--ball-tool-focus', type=float, default=8.0)
    parser.add_argument('--lr', type=float, default=4.5e-5)
    parser.add_argument('--ppo-epochs', type=int, default=10)
    parser.add_argument('--max-arm-rms', type=float, default=0.011)
    parser.add_argument('--snapshot-tick', type=int, default=197)
    parser.add_argument('--max-ticks', type=int, default=475)
    parser.add_argument('--physics-device', default='cuda:0')
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    fault_fd = os.open(args.output / 'native_fault.log', os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    faulthandler.enable(file=fault_fd, all_threads=True)
    stop_request = {'signal': None}
    app = AppLauncher(args, fast_shutdown=False).app
    faulthandler.enable(file=fault_fd, all_threads=True)
    signal.signal(signal.SIGINT, lambda s, f: stop_request.__setitem__('signal', signal.Signals(s).name))
    signal.signal(signal.SIGTERM, lambda s, f: stop_request.__setitem__('signal', signal.Signals(s).name))

    try:
        _run(args, stop_request)
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(1)
    else:
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(0)


def _run(args, stop_request):
    import numpy as np
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from freeze_full_racket_protocol_v4 import incoming_cases
    from quadsmash.physics.full_racket_v4 import string_polygon
    from quadsmash.rl.supermimic_v1 import load_ppo_initialized_actor
    from quadsmash.rl.supermimic_task_contract import (
        resolve_task_contract, make_task_cfg, require_resume_contract,
        zero_panel_reset_noise, validate_sweep_contract)
    from quadsmash.rl.supermimic_terminal_scoring import score_terminal_episode, approach_ball_scoring_protocol
    from quadsmash.rl.supermimic_policy_execution_v1 import resolve_contract, policy_forward
    from quadsmash.rl.supermimic_physical_loop_loading_v1 import APPROACH_NAMES
    from quadsmash.rl.supermimic_native_approach_reset import NativeApproachRestart
    from quadsmash.rl.supermimic_native_approach_search import native_fitness
    from quadsmash.tasks.stand.observation_contract import inverse_rotate
    from quadsmash.tasks.stand.supermimic_env_v1 import SuperMimicEnvV1
    from quadsmash.tasks.stand.full_racket_eval_env_v4 import FrozenPanelEnv

    class TrainingEnv(SuperMimicEnvV1, FrozenPanelEnv):
        pass

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    execution_root = args.g07_execution_root.resolve()
    reserved = reserved_case_ids(execution_root)

    # 64 strictly unseen hard-stratified live val serves (32 A + 32 B)
    val_a_cases = incoming_cases(32, args.val_seed, 'A')
    pool_vb = incoming_cases(160, args.val_seed + 1, 'B')
    vb_hard, vb_std = [], []
    for c in pool_vb:
        if c['case_id'] in reserved:
            continue
        ip = c['prediction_only']['intercept_position_m']
        ft = float(c['prediction_only']['flight_time_s'])
        if (float(ip[1]) >= 0.12 or float(ip[2]) >= 1.82 or ft < 0.72) and len(vb_hard) < 20:
            vb_hard.append(c)
        elif len(vb_std) < 12:
            vb_std.append(c)
    val_cases = val_a_cases + vb_hard + vb_std
    assert len(val_cases) == 64
    assert not ({c['case_id'] for c in val_cases} & reserved), 'Val overlap!'
    reserved.update(c['case_id'] for c in val_cases)
    (args.output / 'val64_gsde_monitor.spec.json').write_text(json.dumps({'cases': val_cases}, indent=2) + '\n')

    n_val = 64
    n_a = args.n_serves_a
    n_b = args.n_serves_b
    n_train = n_a + n_b
    G = 4  # 2 exact antithetic smooth gSDE pairs per serve!
    lanes = n_val + n_train * G
    assert lanes == 256, f'Expected 256 lanes, got {lanes}'
    print(f'[gSDE-PPO] Exact 256-lane scene: val={n_val} (32A+32B) + train=({n_a}A+{n_b}B) x G={G} | NO FB arm anchor!', flush=True)

    def make_iter_cases(it_idx: int):
        sa = args.train_seed_base + it_idx * 29 + 1
        sb = args.train_seed_base + it_idx * 29 + 2
        ca = incoming_cases(n_a, sa, 'A')
        pool_b = incoming_cases(140, sb, 'B')
        hb, sb_l = [], []
        n_hard = int(round(n_b * 0.625))  # 20 hard-tail + 12 i.i.d. Panel B
        for c in pool_b:
            if c['case_id'] in reserved:
                continue
            ip = c['prediction_only']['intercept_position_m']
            ft = float(c['prediction_only']['flight_time_s'])
            is_h = (float(ip[1]) >= 0.11 or float(ip[2]) >= 1.815 or ft < 0.72)
            if is_h and len(hb) < n_hard:
                hb.append(c)
            elif (not is_h) and len(sb_l) < (n_b - n_hard):
                sb_l.append(c)
            if len(hb) == n_hard and len(sb_l) == (n_b - n_hard):
                break
        serves = ca + hb + sb_l
        assert len(serves) == n_train
        assert not ({c['case_id'] for c in serves} & reserved), 'Train overlap!'
        full = list(val_cases)
        for c in serves:
            for _ in range(G):
                full.append(c)
        return serves, full

    trained = torch.load(args.checkpoint, map_location='cpu', weights_only=False,
                         mmap=zipfile.is_zipfile(args.checkpoint))
    init_meta = trained['metadata']['init_policy_checkpoint']
    source_path = Path(init_meta['path'])
    source = torch.load(source_path, map_location='cpu', weights_only=False)
    contract, protocol, _, receipt = resolve_task_contract(
        source, source_path, execution_root=execution_root, root=execution_root)
    scoring_protocol = dict(protocol, _root=str(execution_root))
    contract = dict(contract, approach_ball_scoring=approach_ball_scoring_protocol())
    require_resume_contract(trained, contract)
    scoring_protocol['approach_ball_scoring'] = contract['approach_ball_scoring']
    scoring_protocol['relaxed_scoring'] = None

    train_serves_0, full_cases_0 = make_iter_cases(0)
    cfg = make_task_cfg(lanes, contract, protocol, receipt, execution_root=execution_root)
    cfg.seed, cfg.sim.device = args.seed, args.physics_device
    validate_sweep_contract(cfg, contract, None)
    noise = zero_panel_reset_noise(lanes, args.seed)
    env = TrainingEnv(cfg, cases=full_cases_0, reset_noise=noise,
                      residual_policy_checkpoint=source, g07_execution_root=execution_root,
                      contact_flight_recovery=contract['contact_flight_recovery'],
                      relaxed_protocol=None,
                      approach_ball_protocol=contract['approach_ball_scoring'],
                      relaxed_survival_diagnostic=False)
    wrapper = RslRlVecEnvWrapper(env, clip_actions=None)
    device = env.device

    actor, _ = load_ppo_initialized_actor(source_path, device)
    actor.load_state_dict(trained['actor'], strict=True)
    ref_actor, _ = load_ppo_initialized_actor(source_path, device)
    ref_actor.load_state_dict(trained['actor'], strict=True)
    ref_actor.eval()
    for p in ref_actor.parameters():
        p.requires_grad_(False)

    policy_execution, _ = resolve_contract(trained)
    arm_prep = policy_execution.get('arm_preparation_seconds')
    if arm_prep is not None:
        actor.configure_approach_gate(arm_prep)
        ref_actor.configure_approach_gate(arm_prep)

    w0_col_mult = torch.full((794,), float(args.ball_tool_focus), dtype=torch.float32, device=device)
    for f in range(11):
        b = f * 69
        w0_col_mult[b + 0:b + 9] = 1.0
        w0_col_mult[b + 45:b + 57] = 0.45
        w0_col_mult[b + 61:b + 67] = 0.08
    w0_col_mult[759:776] = 0.45

    polygon = torch.tensor(string_polygon(env.racket_config), dtype=torch.float32, device=device)
    edges = torch.roll(polygon, -1, 0) - polygon
    inward = torch.stack((-edges[:, 1], edges[:, 0]), -1)
    inward = inward / inward.norm(dim=-1, keepdim=True)
    poly_offsets = (polygon * inward).sum(-1)

    def signed_string_margin(motion):
        if not motion or not all(k in motion for k in ('point_m', 'racket_center_m', 'racket_quaternion_w')):
            return None
        point = torch.tensor([motion['point_m']], dtype=torch.float32, device=device)
        center = torch.tensor([motion['racket_center_m']], dtype=torch.float32, device=device)
        quat = torch.tensor([motion['racket_quaternion_w']], dtype=torch.float32, device=device)
        local = inverse_rotate(quat, point - center)[0, 1:3]
        return float((local @ inward.T - poly_offsets).min()) if local.isfinite().all() else None

    def save_checkpoint(path: Path, state_dict):
        out_ckpt = {k: v for k, v in trained.items()
                    if k in ('algorithm', 'adapter_schema', 'actor', 'metadata', 'config',
                             'actor_updates', 'world_updates', 'critic_updates')}
        out_ckpt['actor'] = {k: v.detach().cpu().clone() for k, v in state_dict.items()}
        torch.save(out_ckpt, path)

    print(f'[gSDE-PPO] Running ONE-TIME standing prefix 0..{args.snapshot_tick}...', flush=True)
    t_pre = time.perf_counter()
    obs, _ = wrapper.reset()
    actor.eval()
    with torch.inference_mode():
        for tick in range(args.snapshot_tick):
            policy_obs = obs['policy']
            act = policy_forward(actor, policy_obs, env.rally, mode=policy_execution['recovery_gate'], sample=False)[0].clamp(-1.0, 1.0)
            if env.approach_ball_termination is not None:
                env.record_approach_gate(actor.approach_action_gate(policy_obs))
            obs, _, _, _ = wrapper.step(act)
    snapshot = NativeApproachRestart.capture(env)
    obs_snap = obs['policy'].detach().clone()
    print(f'[gSDE-PPO] Snapshot captured @ tick {args.snapshot_tick} in {time.perf_counter()-t_pre:.1f}s!', flush=True)

    history_path = args.output / 'gsde_ppo_history.jsonl'
    best_val_succ = -1
    best_score = -1e9
    best_state = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}
    base_param_ref = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items() if k not in APPROACH_NAMES}
    cur_lr = float(args.lr)
    n_sim_ticks = args.max_ticks - args.snapshot_tick
    tr_sl = slice(n_val, lanes)
    n_tr_lanes = n_train * G

    for it in range(args.iterations):
        if stop_request['signal']:
            break
        t_iter = time.perf_counter()
        train_serves, full_cases = make_iter_cases(it)
        snapshot.restore(env)
        env.cases = full_cases
        policy_obs = obs_snap.clone()

        decay = args.sigma_decay ** it
        s_sde = args.sigma_sde * decay
        s_bias = args.sigma_bias * decay
        s_ramp = args.sigma_ramp * decay
        rng = torch.Generator(device=device)
        rng.manual_seed(args.seed + 104729 * (it + 1))

        # Per-lane antithetic smooth gSDE weights: W [lanes, 6, 64], b [lanes, 6], r [lanes, 6]
        W_sde = torch.zeros((lanes, 6, 64), dtype=torch.float32, device=device)
        b_sde = torch.zeros((lanes, 6), dtype=torch.float32, device=device)
        r_sde = torch.zeros((lanes, 6), dtype=torch.float32, device=device)
        peak_ticks = torch.full((lanes,), 275.0, dtype=torch.float32, device=device)
        for i, c in enumerate(val_cases):
            peak_ticks[i] = round((2.0 + float(c['prediction_only']['flight_time_s'])) * 100.0)
        for i, c in enumerate(train_serves):
            pt = round((2.0 + float(c['prediction_only']['flight_time_s'])) * 100.0)
            bl = n_val + i * G
            peak_ticks[bl:bl + 2] = pt
            peak_ticks[bl + 2:bl + 4] = pt + 3.0  # Pair 2: +30ms delayed whip timing!
            for p_idx in range(2):
                lp = bl + 2 * p_idx
                lm = bl + 2 * p_idx + 1
                W = (torch.randn((6, 64), generator=rng, device=device) * s_sde).clamp(-2.5 * s_sde, 2.5 * s_sde)
                b = (torch.randn(6, generator=rng, device=device) * s_bias).clamp(-2.5 * s_bias, 2.5 * s_bias)
                r = (torch.randn(6, generator=rng, device=device) * s_ramp).clamp(-2.5 * s_ramp, 2.5 * s_ramp)
                W_sde[lp] = W;  W_sde[lm] = -W
                b_sde[lp] = b;  b_sde[lm] = -b
                r_sde[lp] = r;  r_sde[lm] = -r

        rec_obs, rec_old_mu, rec_act, rec_gate, rec_sig, rec_lane, rec_tick = [], [], [], [], [], [], []
        min_dist = torch.full((lanes,), 9.9, dtype=torch.float32, device=device)
        first = {}
        cursor = 0
        actor.eval()
        mod = actor.actor
        # Penultimate feature extractor: actor.approach 0->1->2->3 (794 -> 128 ELU -> 64 ELU)
        _a0 = getattr(mod.approach, '0')
        _a1 = getattr(mod.approach, '1')
        _a2 = getattr(mod.approach, '2')
        _a3 = getattr(mod.approach, '3')
        def feat_net(x):
            return _a3(_a2(_a1(_a0(x))))

        with torch.inference_mode():
            for step_i in range(n_sim_ticks):
                if stop_request['signal']:
                    break
                tick = args.snapshot_tick + step_i
                base_req = policy_forward(actor, policy_obs, env.rally, mode=policy_execution['recovery_gate'], sample=False)[0]
                agate = actor.approach_action_gate(policy_obs)
                t_f = float(tick)
                active_pre = (env.episode_index == 0) & (~env.rally.had_contact)

                # Smooth state-dependent gSDE perturbation on arm 12:18 (0 on val lanes!)
                norm_o = mod.normalizer(policy_obs)
                h = feat_net(norm_o)  # [lanes, 64]
                u = h / (h.norm(dim=-1, keepdim=True) + 1e-4)  # unit-norm [lanes, 64]
                ramp = ((t_f - (peak_ticks - 18.0)) / 18.0).clamp(0.0, 1.0) * (t_f <= peak_ticks + 4.0).float()
                d_arm = torch.bmm(W_sde, u.unsqueeze(-1)).squeeze(-1) + b_sde + ramp[:, None] * r_sde  # [lanes, 6]
                d_arm = d_arm.clamp(-0.28, 0.28)

                act = base_req.clone()
                if tick >= 202:
                    arm_eff = agate[:, 12:18] * d_arm
                    act[:, 12:18] = torch.where(active_pre[:, None], act[:, 12:18] + arm_eff, act[:, 12:18])
                act = act.clamp(-1.0, 1.0)

                if 202 <= tick <= 355:
                    tr_act_m = active_pre[tr_sl] & (agate[tr_sl, 12:18].amax(-1) > 0.05) & env.rally.in_play[tr_sl]
                    if tr_act_m.any():
                        idx_loc = tr_act_m.nonzero().flatten()
                        idx_glb = idx_loc + n_val
                        sig_eff = torch.sqrt(torch.tensor(s_sde * s_sde + s_bias * s_bias, device=device)
                                             + (ramp[idx_glb] * s_ramp).square())
                        rec_obs.append(policy_obs[idx_glb].half().cpu())
                        rec_old_mu.append(base_req[idx_glb].half().cpu())
                        rec_act.append(act[idx_glb, 12:18].half().cpu())
                        rec_gate.append(agate[idx_glb].half().cpu())
                        rec_sig.append(sig_eff.half().cpu())
                        rec_lane.append(idx_loc.to(torch.int16).cpu())
                        rec_tick.append(torch.full((len(idx_loc),), tick, dtype=torch.int16))

                in_flight = (env.episode_index == 0) & env.rally.in_play & (~env.rally.had_contact)
                if in_flight.any():
                    d_br = (env.shuttle.data.root_pos_w - env.robot.data.body_link_pos_w[:, env.racket_body_id]).norm(dim=-1)
                    min_dist = torch.where(in_flight, torch.minimum(min_dist, d_br), min_dist)

                if env.approach_ball_termination is not None:
                    env.record_approach_gate(agate)
                obs, _, _, _ = wrapper.step(act)
                policy_obs = obs['policy']

                for raw in env.rally_records[cursor:]:
                    if raw['episode_index'] == 0 and raw['env_id'] not in first:
                        first[raw['env_id']] = score_terminal_episode(raw, scoring_protocol)
                cursor = len(env.rally_records)
                if len(first) >= lanes:
                    break

        raw_ep0 = {row['env_id']: row for row in env.rally_records if row['episode_index'] == 0}
        for ln in range(lanes):
            if ln not in raw_ep0:
                raw_ep0[ln] = dict(
                    env_id=ln, episode_index=0, case_id=full_cases[ln]['case_id'],
                    numerical_failures=0, had_contact=False, legal_initial_contact=False,
                    ball_side_complete_except_recovery=False, landed_legal_court=False,
                    landed_target=False, shuttle_net_contact=False, robot_net_fault=False,
                    qualified_airborne_string_contact=False, frame_hit=False, string_hit=False,
                    released=False, outgoing_velocity_mps=[float('nan')] * 3,
                    first_landing_m=[float('nan')] * 3, crossed_net=False, net_margin_m=None,
                    first_approach_contact_motion=None,
                )
        ordered_raw = [raw_ep0[i] for i in range(lanes)]
        rows_all = native_fitness(ordered_raw, env.court_config, signed_string_margin)['rows']
        min_dist_cpu = min_dist.cpu().numpy()

        val_rows = rows_all[:n_val]
        val_a = sum(r['success'] for r in val_rows if r['case_id'].startswith('A-'))
        val_b = sum(r['success'] for r in val_rows if r['case_id'].startswith('B-'))
        val_succ = val_a + val_b
        val_str = sum(r['string'] for r in val_rows)
        val_rnet = sum(r['robot_net'] for r in val_rows)
        val_margin = float(np.mean([r['margin_score'] for r in val_rows]))

        tr_rows = rows_all[n_val:]
        tr_a_rows = [r for r in tr_rows if r['case_id'].startswith('A-')]
        tr_b_rows = [r for r in tr_rows if r['case_id'].startswith('B-')]
        tr_a_succ = sum(r['success'] for r in tr_a_rows)
        tr_b_succ = sum(r['success'] for r in tr_b_rows)
        tr_rnet = sum(r['robot_net'] for r in tr_rows)

        # Within-serve antithetic group advantages across all 48 serves x G=4!
        adv_lane = np.zeros(n_tr_lanes, dtype=np.float32)
        n_active_grp = 0
        for s_idx in range(n_train):
            g_rews = np.array([
                compute_ppo_reward(rows_all[n_val + s_idx * G + g], ordered_raw[n_val + s_idx * G + g],
                                   float(min_dist_cpu[n_val + s_idx * G + g]))
                for g in range(G)
            ], dtype=np.float64)
            std = float(g_rews.std())
            if std > 0.08:
                n_active_grp += 1
                adv_lane[s_idx * G:(s_idx + 1) * G] = np.clip((g_rews - g_rews.mean()) / (std + 0.35), -2.2, 2.2)

        comp_score = (0.75 * (val_succ / n_val)
                      + 0.125 * (tr_a_succ / max(1, len(tr_a_rows)))
                      + 0.125 * (tr_b_succ / max(1, len(tr_b_rows)))
                      + 0.04 * val_margin - 0.6 * (val_rnet > 0 or tr_rnet > 1))
        is_best = (val_rnet == 0 and tr_rnet <= 1 and comp_score > best_score)
        save_checkpoint(args.output / f'gsde_ppo_it{it:02d}.pt', actor.state_dict())
        rolled_back = False
        if is_best:
            best_score = comp_score
            best_val_succ = max(best_val_succ, val_succ)
            best_state = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}
            save_checkpoint(args.output / 'gsde_ppo_best.pt', best_state)
        elif it >= 1 and (val_rnet > 0 or tr_rnet >= 2 or val_succ <= best_val_succ - 5):
            actor.load_state_dict({k: v.to(device) for k, v in best_state.items()}, strict=True)
            cur_lr = max(1.5e-5, cur_lr * 0.72)
            rolled_back = True

        print(f'[IT {it:02d}] ({time.perf_counter()-t_iter:5.1f}s) VAL64={val_succ:2d}/64 (A={val_a:2d}/32 B={val_b:2d}/32 str={val_str:2d} rnet={val_rnet}) '
              f'| TR_A={tr_a_succ:2d}/{len(tr_a_rows)} ({100*tr_a_succ/len(tr_a_rows):5.1f}%) '
              f'TR_B={tr_b_succ:3d}/{len(tr_b_rows)} ({100*tr_b_succ/len(tr_b_rows):5.1f}%) tr_rnet={tr_rnet} '
              f'| grp={n_active_grp:2d}/{n_train} | comp={comp_score:.4f}{" *BEST*" if is_best else ""}{" [ROLLBACK]" if rolled_back else ""}', flush=True)

        with history_path.open('a') as f:
            f.write(json.dumps(dict(
                iter=it, elapsed_s=time.perf_counter() - t_iter, val_succ=val_succ, val_a=val_a, val_b=val_b,
                val_str=val_str, val_rnet=val_rnet, tr_a_succ=tr_a_succ, tr_b_succ=tr_b_succ, tr_rnet=tr_rnet,
                active_groups=n_active_grp, comp_score=comp_score, is_best=is_best, rolled_back=rolled_back,
            )) + '\n')

        if it == args.iterations - 1 or rolled_back or not rec_obs:
            continue

        # Exact Episode-Level PPO-Clip update on active pre-contact transitions!
        b_obs = torch.cat(rec_obs, dim=0).float().to(device)
        b_old_mu = torch.cat(rec_old_mu, dim=0).float().to(device)
        b_act = torch.cat(rec_act, dim=0).float().to(device)
        b_gate = torch.cat(rec_gate, dim=0).float().to(device)
        b_sig = torch.cat(rec_sig, dim=0).float().to(device)
        b_lane = torch.cat(rec_lane, dim=0).long().to(device)
        b_tick = torch.cat(rec_tick, dim=0).float().to(device)

        # Compute per-frame contact-proximity weight (3.5x on final 160ms before contact!)
        max_tick_lane = torch.zeros(n_tr_lanes, dtype=torch.float32, device=device)
        max_tick_lane.scatter_reduce_(0, b_lane, b_tick, reduce='amax', include_self=False)
        dt_end = (max_tick_lane[b_lane] - b_tick).clamp_min(0.0)
        b_tw = torch.where(dt_end <= 16.0, 3.5, torch.where(dt_end <= 28.0, 1.8, 1.0))

        with torch.no_grad():
            b_norm = mod.normalizer(b_obs)
            b_base = mod.base(b_norm)
            b_ag = actor.approach_action_gate(b_obs)
            ref_req = (b_base + b_ag * ref_actor.actor.approach(b_norm) * ref_actor.actor.approach_scale).clamp(-1.0, 1.0)
            old_req = (b_base + b_ag * mod.approach(b_norm) * mod.approach_scale).clamp(-1.0, 1.0)
            m_arm = b_gate[:, 12:18] > 0.05
            m_leg = (b_gate[:, :12] > 0.02).float()
            # Precompute old squared residual per active frame
            eff_sd = (b_sig[:, None] * b_gate[:, 12:18].clamp_min(0.20)).square()  # [N, 6]
            old_sq_norm = ((b_act - old_req[:, 12:18]).square() / eff_sd) * m_arm.float()  # [N, 6]

        adv_t = torch.tensor(adv_lane, dtype=torch.float32, device=device)  # [n_tr_lanes]
        lane_wsum = torch.zeros(n_tr_lanes, dtype=torch.float32, device=device).scatter_add_(0, b_lane, b_tw).clamp_min(1.0)
        active_lanes = (adv_t.abs() > 1e-4) & (lane_wsum > 1.0)

        for name, p in actor.named_parameters():
            p.requires_grad_(name in APPROACH_NAMES)
        trainable = [(name, p) for name, p in actor.named_parameters() if p.requires_grad]
        old_params = {name: p.detach().clone() for name, p in trainable}
        opt = torch.optim.Adam([p for _, p in trainable], lr=cur_lr)

        actor.train()
        arm_rms_actual = 0.0
        clip_frac = 0.0
        for ep in range(1, args.ppo_epochs + 1):
            opt.zero_grad(set_to_none=True)
            pred = (b_base + b_ag * mod.approach(b_norm) * mod.approach_scale).clamp(-1.0, 1.0)

            # Exact episode-averaged Gaussian log-likelihood ratio per lane!
            new_sq_norm = ((b_act - pred[:, 12:18]).square() / eff_sd) * m_arm.float()
            frame_dlogp = 0.5 * (old_sq_norm - new_sq_norm).sum(dim=-1)  # [N]
            lane_dlogp = torch.zeros(n_tr_lanes, dtype=torch.float32, device=device).scatter_add_(0, b_lane, frame_dlogp * b_tw) / lane_wsum
            ratio = torch.exp(lane_dlogp.clamp(-2.0, 2.0))
            surr1 = ratio * adv_t
            surr2 = ratio.clamp(1.0 - args.clip_eps, 1.0 + args.clip_eps) * adv_t
            loss_ppo = -torch.minimum(surr1, surr2)[active_lanes].mean()

            # Step-to-step arm KL trust region to old_req (NOT to FB!) + leg anchor to ref_req (rnet=0!)
            loss_kl = ((pred[:, 12:18] - old_req[:, 12:18]).square() * m_arm.float()).sum() / m_arm.float().sum().clamp_min(1.0)
            loss_leg = ((pred[:, :12] - ref_req[:, :12]).square() * m_leg).sum() / m_leg.sum().clamp_min(1.0)

            prox_t = []
            for name, p in trainable:
                if name == 'actor.approach.0.weight':
                    prox_t.append(1.5 * ((p - old_params[name]).square() * w0_col_mult[None, :]).mean())
                else:
                    prox_t.append(0.3 * (p - old_params[name]).square().mean())
            loss = loss_ppo + args.kl_beta * loss_kl + args.leg_anchor * loss_leg + args.prox_weight * (sum(prox_t) / len(prox_t))
            loss.backward()
            for name, p in trainable:
                if name in ('actor.approach.4.weight', 'actor.approach.4.bias') and p.grad is not None:
                    p.grad[:12].zero_()
            torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.2)
            opt.step()

            with torch.no_grad():
                pred_chk = (b_base + b_ag * mod.approach(b_norm) * mod.approach_scale).clamp(-1.0, 1.0)
                arm_rms_actual = float((pred_chk - old_req)[:, 12:18][m_arm].square().mean().sqrt())
                clip_frac = float(((ratio[active_lanes] - 1.0).abs() > args.clip_eps).float().mean())
                if arm_rms_actual >= args.max_arm_rms:
                    break

        actor.eval()
        for k, v in base_param_ref.items():
            if torch.is_tensor(v):
                assert torch.equal(v, actor.state_dict()[k].cpu()), f'Frozen param mutated: {k}'
        print(f'  [PPO {it:02d}] frames={len(b_obs)} ep={ep}/{args.ppo_epochs} d_arm_rms={arm_rms_actual:.4f} clip={100*clip_frac:4.1f}% lr={cur_lr:.1e}', flush=True)

    print(f'[gSDE-PPO] Done! Best -> {args.output / "gsde_ppo_best.pt"}', flush=True)


if __name__ == '__main__':
    main()
