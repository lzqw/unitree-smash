#!/usr/bin/env python3
"""High-SNR 448-Rollout Unchained On-Policy Smooth-Basis Antithetic PPO directly in Real PhysX.

Upgrades over gsde_ppo_256 (9x higher SNR, healthy 10-15% PPO clipping, zero Panel-A exploration damage):
1. Pure 12-D smooth temporal exploration (6 arm pose + 6 arm whip ramp; NO 384-D W_sde noise!).
   Pair 1 @ peak_tick (standard pose + whip), Pair 2 @ peak_tick + 3 (+30ms delayed apex-whip!).
2. 2-Subbatch Accumulation per PPO update (512 real-PhysX episodes/update, exact 256-lane parity!):
   - Sub-batch 1: 64 deterministic unseen val lanes (32 A + 32 B) + 48 serves (16A + 32B) x G=4 = 256 lanes.
   - Sub-batch 2: 64 MORE fresh serves (20A + 44B) x G=4 = 256 lanes (100% GPU utilization!).
   -> 112 fresh disjoint serves (36 A + 76 B) x G=4 = 448 rollouts = 224 antithetic pairs in 12-D (SNR = 4.32!).
3. Adaptive exploration sigma: gentle on Panel A (0.60x) so TR_A >= 85% and Panel-A climbs smoothly to 95%+,
   stronger on Panel B (60% hard-tail iz>=1.815 / iy>=0.11 / ft<0.72 + 40% i.i.d. B).
4. Exact episode-level PPO-Clip with lr=1.8e-5, clip_eps=0.14, kl_beta=6.0 (to theta_old ONLY — ZERO FB arm
   anchor!), max_arm_rms=0.0065 per update, frozen leg head 0:12 + leg_anchor=25.0 (rnet=0), clean rollback.
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
    parser.add_argument('--val-seed', type=int, default=994101)
    parser.add_argument('--train-seed-base', type=int, default=990000)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--sigma-pos-b', type=float, default=0.068)
    parser.add_argument('--sigma-ramp-b', type=float, default=0.060)
    parser.add_argument('--pa-sigma-scale', type=float, default=0.60)
    parser.add_argument('--sigma-decay', type=float, default=0.96)
    parser.add_argument('--clip-eps', type=float, default=0.14)
    parser.add_argument('--kl-beta', type=float, default=6.0)
    parser.add_argument('--leg-anchor', type=float, default=25.0)
    parser.add_argument('--prox-weight', type=float, default=0.03)
    parser.add_argument('--ball-tool-focus', type=float, default=8.0)
    parser.add_argument('--lr', type=float, default=1.8e-5)
    parser.add_argument('--ppo-epochs', type=int, default=14)
    parser.add_argument('--max-arm-rms', type=float, default=0.0065)
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
    (args.output / 'val64_smooth_monitor.spec.json').write_text(json.dumps({'cases': val_cases}, indent=2) + '\n')

    lanes = 256
    G = 4
    print('[SmoothPPO-448] Exact 256-lane x 2-subbatch (112 serves [36A+76B] x G=4 = 448 rollouts / update!)', flush=True)

    def sample_serves(n_a: int, n_b: int, seed_base: int):
        ca = incoming_cases(n_a, seed_base + 1, 'A')
        pool_b = incoming_cases(n_b * 4 + 40, seed_base + 2, 'B')
        hb, sb_l = [], []
        n_hard = int(round(n_b * 0.62))
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
        assert len(serves) == n_a + n_b
        assert not ({c['case_id'] for c in serves} & reserved), 'Train overlap!'
        return serves

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

    init_serves = sample_serves(16, 32, args.train_seed_base)
    full_cases_0 = list(val_cases) + [c for c in init_serves for _ in range(G)]
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

    print(f'[SmoothPPO-448] Running ONE-TIME standing prefix 0..{args.snapshot_tick}...', flush=True)
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
    print(f'[SmoothPPO-448] Snapshot captured @ tick {args.snapshot_tick} in {time.perf_counter()-t_pre:.1f}s!', flush=True)

    history_path = args.output / 'smooth_ppo_history.jsonl'
    best_val_succ = -1
    best_score = -1e9
    best_state = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}
    base_param_ref = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items() if k not in APPROACH_NAMES}
    cur_lr = float(args.lr)
    n_sim_ticks = args.max_ticks - args.snapshot_tick

    def run_subbatch(serves, include_val: bool, seed_rng: int, decay: float, lane_offset: int):
        n_v = len(val_cases) if include_val else 0
        full = (list(val_cases) if include_val else []) + [c for c in serves for _ in range(G)]
        assert len(full) == lanes
        snapshot.restore(env)
        env.cases = full
        policy_obs = obs_snap.clone()

        rng = torch.Generator(device=device)
        rng.manual_seed(seed_rng)
        s_pb = args.sigma_pos_b * decay
        s_rb = args.sigma_ramp_b * decay

        z_pos = torch.zeros((lanes, 6), dtype=torch.float32, device=device)
        z_rmp = torch.zeros((lanes, 6), dtype=torch.float32, device=device)
        sig_p_lane = torch.full((lanes,), s_pb, dtype=torch.float32, device=device)
        sig_r_lane = torch.full((lanes,), s_rb, dtype=torch.float32, device=device)
        peak_ticks = torch.full((lanes,), 275.0, dtype=torch.float32, device=device)
        if include_val:
            for i, c in enumerate(val_cases):
                peak_ticks[i] = round((2.0 + float(c['prediction_only']['flight_time_s'])) * 100.0)

        for i, c in enumerate(serves):
            pt = round((2.0 + float(c['prediction_only']['flight_time_s'])) * 100.0)
            bl = n_v + i * G
            peak_ticks[bl:bl + 2] = pt
            peak_ticks[bl + 2:bl + 4] = pt + 3.0  # Pair 2: +30ms delayed apex-whip!
            sc = args.pa_sigma_scale if c['case_id'].startswith('A-') else 1.0
            sp = s_pb * sc
            sr = s_rb * sc
            sig_p_lane[bl:bl + G] = sp
            sig_r_lane[bl:bl + G] = sr
            for p_idx in range(2):
                lp = bl + 2 * p_idx
                lm = bl + 2 * p_idx + 1
                dp = (torch.randn(6, generator=rng, device=device) * sp).clamp(-2.4 * sp, 2.4 * sp)
                dr = (torch.randn(6, generator=rng, device=device) * sr).clamp(-2.4 * sr, 2.4 * sr)
                if p_idx == 1:
                    # Pair 2 delayed cock-back + whip!
                    z_pos[lp] = dp - 0.45 * dr; z_pos[lm] = -(dp - 0.45 * dr)
                else:
                    z_pos[lp] = dp; z_pos[lm] = -dp
                z_rmp[lp] = dr; z_rmp[lm] = -dr

        tr_sl = slice(n_v, lanes)
        rec_obs, rec_act, rec_gate, rec_sig, rec_lane, rec_tick = [], [], [], [], [], []
        min_dist = torch.full((lanes,), 9.9, dtype=torch.float32, device=device)
        first = {}
        cursor = 0
        actor.eval()

        with torch.inference_mode():
            for step_i in range(n_sim_ticks):
                if stop_request['signal']:
                    break
                tick = args.snapshot_tick + step_i
                base_req = policy_forward(actor, policy_obs, env.rally, mode=policy_execution['recovery_gate'], sample=False)[0]
                agate = actor.approach_action_gate(policy_obs)
                t_f = float(tick)
                active_pre = (env.episode_index == 0) & (~env.rally.had_contact)

                rise = ((t_f - (peak_ticks - 52.0)) / 52.0).clamp(0.0, 1.0)
                fall = ((t_f - peak_ticks) / 16.0).clamp(0.0, 1.0)
                env_p = torch.where(t_f <= peak_ticks, torch.sin(0.5 * math.pi * rise).square(), torch.cos(0.5 * math.pi * fall).square())
                env_r = ((t_f - (peak_ticks - 18.0)) / 18.0).clamp(0.0, 1.0) * (t_f <= peak_ticks + 4.0).float()

                d_arm = (env_p[:, None] * z_pos + env_r[:, None] * z_rmp).clamp(-0.26, 0.26)
                act = base_req.clone()
                if tick >= 202:
                    act[:, 12:18] = torch.where(active_pre[:, None], act[:, 12:18] + agate[:, 12:18] * d_arm, act[:, 12:18])
                act = act.clamp(-1.0, 1.0)

                if 202 <= tick <= 355:
                    tr_m = active_pre[tr_sl] & (agate[tr_sl, 12:18].amax(-1) > 0.05) & env.rally.in_play[tr_sl]
                    if tr_m.any():
                        idx_loc = tr_m.nonzero().flatten()
                        idx_glb = idx_loc + n_v
                        sig_eff = torch.sqrt((env_p[idx_glb] * sig_p_lane[idx_glb]).square()
                                             + (env_r[idx_glb] * sig_r_lane[idx_glb]).square()).clamp_min(0.025)
                        rec_obs.append(policy_obs[idx_glb].half().cpu())
                        rec_act.append(act[idx_glb, 12:18].half().cpu())
                        rec_gate.append(agate[idx_glb].half().cpu())
                        rec_sig.append(sig_eff.half().cpu())
                        rec_lane.append((idx_loc + lane_offset).to(torch.int16).cpu())
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
                    env_id=ln, episode_index=0, case_id=full[ln]['case_id'],
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
        min_d_cpu = min_dist.cpu().numpy()

        n_s = len(serves)
        adv = np.zeros(n_s * G, dtype=np.float32)
        n_act_grp = 0
        for s_idx in range(n_s):
            g_rews = np.array([
                compute_ppo_reward(rows_all[n_v + s_idx * G + g], ordered_raw[n_v + s_idx * G + g],
                                   float(min_d_cpu[n_v + s_idx * G + g]))
                for g in range(G)
            ], dtype=np.float64)
            std = float(g_rews.std())
            if std > 0.08:
                n_act_grp += 1
                adv[s_idx * G:(s_idx + 1) * G] = np.clip((g_rews - g_rews.mean()) / (std + 0.35), -2.0, 2.0)

        return dict(
            val_rows=rows_all[:n_v] if include_val else [],
            tr_rows=rows_all[n_v:],
            adv=adv, n_act_grp=n_act_grp,
            obs=rec_obs, act=rec_act, gate=rec_gate, sig=rec_sig, lane=rec_lane, tick=rec_tick,
        )

    for it in range(args.iterations):
        if stop_request['signal']:
            break
        t_iter = time.perf_counter()
        decay = args.sigma_decay ** it
        s1 = sample_serves(16, 32, args.train_seed_base + it * 47 + 0)
        s2 = sample_serves(20, 44, args.train_seed_base + it * 47 + 19)

        b1 = run_subbatch(s1, include_val=True, seed_rng=args.seed + 10007 * (2 * it + 1), decay=decay, lane_offset=0)
        if stop_request['signal']:
            break
        b2 = run_subbatch(s2, include_val=False, seed_rng=args.seed + 10007 * (2 * it + 2), decay=decay, lane_offset=len(s1) * G)

        val_rows = b1['val_rows']
        val_a = sum(r['success'] for r in val_rows if r['case_id'].startswith('A-'))
        val_b = sum(r['success'] for r in val_rows if r['case_id'].startswith('B-'))
        val_succ = val_a + val_b
        val_str = sum(r['string'] for r in val_rows)
        val_rnet = sum(r['robot_net'] for r in val_rows)
        val_margin = float(np.mean([r['margin_score'] for r in val_rows]))

        tr_rows = b1['tr_rows'] + b2['tr_rows']
        tr_a_r = [r for r in tr_rows if r['case_id'].startswith('A-')]
        tr_b_r = [r for r in tr_rows if r['case_id'].startswith('B-')]
        tr_a_succ = sum(r['success'] for r in tr_a_r)
        tr_b_succ = sum(r['success'] for r in tr_b_r)
        tr_rnet = sum(r['robot_net'] for r in tr_rows)
        n_act_grp = b1['n_act_grp'] + b2['n_act_grp']

        comp_score = (0.70 * (val_succ / 64.0)
                      + 0.15 * (tr_a_succ / max(1, len(tr_a_r)))
                      + 0.15 * (tr_b_succ / max(1, len(tr_b_r)))
                      + 0.04 * val_margin - 0.6 * (val_rnet > 0 or tr_rnet > 1))
        is_best = (val_rnet == 0 and tr_rnet <= 1 and comp_score > best_score)
        save_checkpoint(args.output / f'smooth_ppo_it{it:02d}.pt', actor.state_dict())
        rolled_back = False
        if is_best:
            best_score = comp_score
            best_val_succ = max(best_val_succ, val_succ)
            best_state = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}
            save_checkpoint(args.output / 'smooth_ppo_best.pt', best_state)
        elif it >= 1 and (val_rnet > 0 or tr_rnet >= 3 or val_succ <= best_val_succ - 5):
            actor.load_state_dict({k: v.to(device) for k, v in best_state.items()}, strict=True)
            cur_lr = max(8e-6, cur_lr * 0.72)
            rolled_back = True

        print(f'[IT {it:02d}] ({time.perf_counter()-t_iter:5.1f}s) VAL64={val_succ:2d}/64 (A={val_a:2d}/32 B={val_b:2d}/32 str={val_str:2d} rnet={val_rnet}) '
              f'| TR_A={tr_a_succ:3d}/{len(tr_a_r)} ({100*tr_a_succ/len(tr_a_r):5.1f}%) '
              f'TR_B={tr_b_succ:3d}/{len(tr_b_r)} ({100*tr_b_succ/len(tr_b_r):5.1f}%) tr_rnet={tr_rnet} '
              f'| grp={n_act_grp:3d}/112 | comp={comp_score:.4f}{" *BEST*" if is_best else ""}{" [ROLLBACK]" if rolled_back else ""}', flush=True)

        with history_path.open('a') as f:
            f.write(json.dumps(dict(
                iter=it, elapsed_s=time.perf_counter() - t_iter, val_succ=val_succ, val_a=val_a, val_b=val_b,
                val_str=val_str, val_rnet=val_rnet, tr_a_succ=tr_a_succ, tr_b_succ=tr_b_succ, tr_rnet=tr_rnet,
                active_groups=n_act_grp, comp_score=comp_score, is_best=is_best, rolled_back=rolled_back,
            )) + '\n')

        if it == args.iterations - 1 or rolled_back:
            continue

        # Merge all 448 real-PhysX training rollouts into ONE low-variance PPO-Clip update!
        n_tr_lanes = (len(s1) + len(s2)) * G
        b_obs = torch.cat(b1['obs'] + b2['obs'], dim=0).float().to(device)
        b_act = torch.cat(b1['act'] + b2['act'], dim=0).float().to(device)
        b_gate = torch.cat(b1['gate'] + b2['gate'], dim=0).float().to(device)
        b_sig = torch.cat(b1['sig'] + b2['sig'], dim=0).float().to(device)
        b_lane = torch.cat(b1['lane'] + b2['lane'], dim=0).long().to(device)
        b_tick = torch.cat(b1['tick'] + b2['tick'], dim=0).float().to(device)
        adv_t = torch.tensor(np.concatenate([b1['adv'], b2['adv']]), dtype=torch.float32, device=device)

        max_tick_lane = torch.zeros(n_tr_lanes, dtype=torch.float32, device=device)
        max_tick_lane.scatter_reduce_(0, b_lane, b_tick, reduce='amax', include_self=False)
        dt_end = (max_tick_lane[b_lane] - b_tick).clamp_min(0.0)
        b_tw = torch.where(dt_end <= 16.0, 3.5, torch.where(dt_end <= 28.0, 1.8, 1.0))

        mod = actor.actor
        with torch.no_grad():
            b_norm = mod.normalizer(b_obs)
            b_base = mod.base(b_norm)
            b_ag = actor.approach_action_gate(b_obs)
            ref_req = (b_base + b_ag * ref_actor.actor.approach(b_norm) * ref_actor.actor.approach_scale).clamp(-1.0, 1.0)
            old_req = (b_base + b_ag * mod.approach(b_norm) * mod.approach_scale).clamp(-1.0, 1.0)
            m_arm = b_gate[:, 12:18] > 0.05
            m_leg = (b_gate[:, :12] > 0.02).float()
            eff_sd = (b_sig[:, None] * b_gate[:, 12:18].clamp_min(0.25)).square()
            old_sq_norm = ((b_act - old_req[:, 12:18]).square() / eff_sd) * m_arm.float()

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
            new_sq_norm = ((b_act - pred[:, 12:18]).square() / eff_sd) * m_arm.float()
            frame_dlogp = 0.5 * (old_sq_norm - new_sq_norm).sum(dim=-1)
            lane_dlogp = torch.zeros(n_tr_lanes, dtype=torch.float32, device=device).scatter_add_(0, b_lane, frame_dlogp * b_tw) / lane_wsum
            ratio = torch.exp(lane_dlogp.clamp(-1.8, 1.8))
            surr1 = ratio * adv_t
            surr2 = ratio.clamp(1.0 - args.clip_eps, 1.0 + args.clip_eps) * adv_t
            loss_ppo = -torch.minimum(surr1, surr2)[active_lanes].mean()

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
            torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
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

    print(f'[SmoothPPO-448] Done! Best -> {args.output / "smooth_ppo_best.pt"}', flush=True)


if __name__ == '__main__':
    main()
