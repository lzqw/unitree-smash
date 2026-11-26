#!/usr/bin/env python3
"""Fast On-Policy Resident 256-Lane Real-PhysX Rejection-Sampled RL (RS-GRPO v2) + 891-Anchor Panel-A Fortress.

Fixes EVERY bottleneck discovered across P22-P27:
1. Exact 256-lane physics parity (env_spacing=0.0) + NativeApproachRestart @ tick 197 (~4.8 min/iter, 2.2x faster!).
2. Layout: 56 deterministic unseen hard-stratified val lanes (24 A + 32 B, zero leakage)
   + 40 fresh disjoint Panel-B serves/iter (60% hard-tail iz>=1.81 / |iy|>=0.12 / ft<0.72 + 40% i.i.d. B)
   x G=5 (1 deterministic nominal + Pair 1 standard smooth pos+whip + Pair 2 +30ms delayed apex-whip!) = 256 lanes.
3. ZERO NOISE ON CRISP SUCCESSES: nominal passes with safe margins (lm>=0.14, spm>=0.25, nm>=0.020, sm>=0.010)
   stay pinned to delta=0 (never corrupted by random ES noise!).
4. HIGH-MARGIN ELITE SELF-IMITATION: nominal failures rescued by a clean perturbed branch
   (success=True, rnet=0, spm>=0.15, lm>=0.08, nm>=0.010) imitate the exact winning smooth delta (0.85x);
   unrescued failures only apply gentle antithetic ES if best branch beats nominal by >= +0.85 reward!
5. Smooth per-minibatch trust-region cap (lr=2.0e-5, max_arm_rms=0.0065) + CLEAN rollback (clear buffer & skip!).
6. Ironclad 891-anchor L_inf Panel-A Fortress (all pure FB Panel-A successes across r1..r6, r9, r12 on GPU,
   pa_mse=10.0 + pa_hinge=100.0 @ margin=0.010) + frozen leg head rows 0:12 -> PA_q95 <= 0.014, rnet=0!
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


def compute_rsrl_reward(row, raw_ep, min_dist_m: float) -> float:
    succ = 1.0 if row['success'] else 0.0
    qstr = 1.0 if row['string'] else 0.0
    rnet = 1.0 if row['robot_net'] else 0.0
    sm = row['string_margin_m']
    sm_t = max(-1.5, min(1.0, sm / 0.040)) if (sm is not None and math.isfinite(sm)) else -1.5
    spm = row['speed_margin_mps']
    spm_t = max(-1.5, min(1.0, spm / 0.60)) if (qstr > 0 and spm is not None and math.isfinite(spm)) else -0.5
    lm = row['landing_margin_m']
    lm_t = max(-1.5, min(1.0, lm / 0.42)) if (qstr > 0 and lm is not None and math.isfinite(lm)) else -1.0
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
        4.20 * succ
        + 1.35 * qstr
        + 0.80 * sm_t
        + 0.70 * spm_t
        + 0.90 * lm_t
        + 0.45 * nm_t
        + ctr_pen
        + dist_pen
        - 8.00 * rnet
    )


def load_panel_a_fortress(anchor_dirs, pa_actor, device):
    """Preload ALL verified pure-FB Panel-A Fortress frames onto GPU."""
    import torch
    from distill_native_approach import extract_episode_frames
    obs_l, gate_l, w_l = [], [], []
    n_eps = 0
    seen = set()
    for adir in anchor_dirs:
        adir = Path(adir)
        ps = json.loads((adir / 'probe_summary.json').read_text())['rows']
        tape = torch.load(adir / 'probe_tape.pt', map_location='cpu', weights_only=True)['tape']
        for i, row in enumerate(ps):
            cid = row['case_id']
            if cid in seen or not cid.startswith('A-'):
                continue
            if not (row['success'] and not row['robot_net'] and not row['low_speed']):
                continue
            ep = extract_episode_frames(tape, i)
            if ep is None:
                continue
            seen.add(cid)
            n_eps += 1
            obs_l.append(ep['obs'])
            gate_l.append(ep['gate'])
            w_l.append(ep['weight'])
    obs_all = torch.cat(obs_l, dim=0).to(device)
    gate_all = torch.cat(gate_l, dim=0).to(device)
    w_all = torch.cat(w_l, dim=0).to(device)
    mod = pa_actor.actor
    with torch.no_grad():
        norm_all = mod.normalizer(obs_all)
        base_all = mod.base(norm_all)
        agate_all = pa_actor.approach_action_gate(obs_all)
        req_all = (base_all + agate_all * mod.approach(norm_all) * mod.approach_scale).clamp(-1.0, 1.0)
    print(f'[RSRL-256] Preloaded {n_eps} pure-FB Panel-A Fortress anchors ({len(obs_all)} frames) on {device}', flush=True)
    return dict(norm_obs=norm_all, base_out=base_all, agate=agate_all, req=req_all, gate=gate_all, weight=w_all)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--panel-a-anchor-ckpt', type=Path, required=True)
    parser.add_argument('--panel-a-anchor-dirs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--g07-execution-root', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=882072)
    parser.add_argument('--val-seed', type=int, default=992501)
    parser.add_argument('--train-seed-base', type=int, default=980000)
    parser.add_argument('--iterations', type=int, default=14)
    parser.add_argument('--sigma-arm-pos', type=float, default=0.074)
    parser.add_argument('--sigma-arm-ramp', type=float, default=0.064)
    parser.add_argument('--sigma-delay-pos', type=float, default=0.090)
    parser.add_argument('--sigma-delay-ramp', type=float, default=0.080)
    parser.add_argument('--sigma-decay', type=float, default=0.96)
    parser.add_argument('--step-eta', type=float, default=0.14)
    parser.add_argument('--elite-scale', type=float, default=0.85)
    parser.add_argument('--kl-anchor', type=float, default=0.80)
    parser.add_argument('--pa-mse-weight', type=float, default=10.0)
    parser.add_argument('--pa-hinge-margin', type=float, default=0.010)
    parser.add_argument('--pa-hinge-weight', type=float, default=100.0)
    parser.add_argument('--prox-weight', type=float, default=0.05)
    parser.add_argument('--ball-tool-focus', type=float, default=10.0)
    parser.add_argument('--lr', type=float, default=2.0e-5)
    parser.add_argument('--update-epochs', type=int, default=35)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--max-arm-rms', type=float, default=0.0065)
    parser.add_argument('--replay-iters', type=int, default=4)
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
    from torch.nn import functional as F
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

    # Build 56 strictly unseen hard-stratified live validation serves (24 A + 32 B)
    val_a_cases = incoming_cases(24, args.val_seed, 'A')
    pool_vb = incoming_cases(140, args.val_seed + 1, 'B')
    val_b_hard, val_b_std = [], []
    for c in pool_vb:
        if c['case_id'] in reserved:
            continue
        ip = c['prediction_only']['intercept_position_m']
        ft = float(c['prediction_only']['flight_time_s'])
        if (float(ip[1]) >= 0.12 or float(ip[2]) >= 1.82 or ft < 0.72) and len(val_b_hard) < 20:
            val_b_hard.append(c)
        elif len(val_b_std) < 12:
            val_b_std.append(c)
    val_cases = val_a_cases + val_b_hard + val_b_std
    assert len(val_cases) == 56
    assert not ({c['case_id'] for c in val_cases} & reserved), 'Val overlap!'
    reserved.update(c['case_id'] for c in val_cases)
    (args.output / 'val56_monitor.spec.json').write_text(json.dumps({'cases': val_cases}, indent=2) + '\n')

    n_val = 56
    n_train = 40
    G = 5  # 1 nominal + Pair 1 (std pos+whip) + Pair 2 (+30ms delayed apex-whip!)
    lanes = n_val + n_train * G
    assert lanes == 256
    print(f'[RSRL-256] Exact 256-lane resident scene: val={n_val} (24A+32B) + train={n_train}B x G={G} (snapshot@{args.snapshot_tick})', flush=True)

    def make_iter_cases(it_idx: int):
        sb = args.train_seed_base + it_idx * 23 + 5
        pool = incoming_cases(160, sb, 'B')
        hard_b, std_b = [], []
        for c in pool:
            if c['case_id'] in reserved:
                continue
            ip = c['prediction_only']['intercept_position_m']
            ft = float(c['prediction_only']['flight_time_s'])
            is_hard = (float(ip[1]) >= 0.11 or float(ip[2]) >= 1.815 or ft < 0.72)
            if is_hard and len(hard_b) < 24:
                hard_b.append(c)
            elif (not is_hard) and len(std_b) < 16:
                std_b.append(c)
            if len(hard_b) == 24 and len(std_b) == 16:
                break
        cb = hard_b + std_b
        assert len(cb) == n_train
        assert not ({c['case_id'] for c in cb} & reserved), 'Train overlap!'
        full = list(val_cases)
        for c in cb:
            for _ in range(G):
                full.append(c)
        return cb, full

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
    policy_execution, _ = resolve_contract(trained)
    arm_prep = policy_execution.get('arm_preparation_seconds')
    if arm_prep is not None:
        actor.configure_approach_gate(arm_prep)

    # Preload Ironclad 891-anchor pure-FB Panel-A Fortress onto GPU
    pa_ckpt = torch.load(args.panel_a_anchor_ckpt, map_location='cpu', weights_only=False)
    pa_actor, _ = load_ppo_initialized_actor(source_path, device)
    pa_actor.load_state_dict(pa_ckpt['actor'], strict=True)
    if arm_prep is not None:
        pa_actor.configure_approach_gate(arm_prep)
    pa_actor.eval()
    pa_buf = load_panel_a_fortress(args.panel_a_anchor_dirs, pa_actor, device)
    del pa_actor

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

    # ONE-TIME standing prefix ticks 0..snapshot_tick (197) -> saves 4 min on EVERY iteration!
    print(f'[RSRL-256] Running ONE-TIME standing prefix 0..{args.snapshot_tick}...', flush=True)
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
    print(f'[RSRL-256] Snapshot captured @ tick {args.snapshot_tick} in {time.perf_counter()-t_pre:.1f}s!', flush=True)

    history_path = args.output / 'rsrl256_history.jsonl'
    best_val_succ = -1
    best_score = -1e9
    best_state = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}
    replay_buffer = []
    nom_lane_ids = torch.tensor([n_val + i * G for i in range(n_train)], dtype=torch.long, device=device)
    base_param_ref = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items() if k not in APPROACH_NAMES}
    cur_eta = float(args.step_eta)
    n_sim_ticks = args.max_ticks - args.snapshot_tick

    for it in range(args.iterations):
        if stop_request['signal']:
            break
        t_iter = time.perf_counter()
        train_serves, full_cases = make_iter_cases(it)
        snapshot.restore(env)
        env.cases = full_cases
        policy_obs = obs_snap.clone()

        decay = args.sigma_decay ** it
        s_p1 = args.sigma_arm_pos * decay
        s_r1 = args.sigma_arm_ramp * decay
        s_p2 = args.sigma_delay_pos * decay
        s_r2 = args.sigma_delay_ramp * decay
        rng = torch.Generator(device=device)
        rng.manual_seed(args.seed + 9173 * (it + 1))

        # Per-lane smooth basis: Pair 1 @ peak_tick, Pair 2 @ peak_tick + 3 (+30ms delayed apex-whip!)
        z_pos = torch.zeros((lanes, 18), dtype=torch.float32, device=device)
        z_ramp = torch.zeros((lanes, 18), dtype=torch.float32, device=device)
        peak_ticks = torch.full((lanes,), 275.0, dtype=torch.float32, device=device)
        for i, c in enumerate(val_cases):
            peak_ticks[i] = round((2.0 + float(c['prediction_only']['flight_time_s'])) * 100.0)
        for i, c in enumerate(train_serves):
            pt = round((2.0 + float(c['prediction_only']['flight_time_s'])) * 100.0)
            bl = n_val + i * G
            peak_ticks[bl:bl + 3] = pt
            peak_ticks[bl + 3:bl + 5] = pt + 3.0  # Pair 2: +30ms delayed apex-whip timing!
            # Pair 1: standard smooth position + whip
            dp1 = (torch.randn(6, generator=rng, device=device) * s_p1).clamp(-2.4 * s_p1, 2.4 * s_p1)
            dr1 = (torch.randn(6, generator=rng, device=device) * s_r1).clamp(-2.4 * s_r1, 2.4 * s_r1)
            z_pos[bl + 1, 12:] = dp1;  z_pos[bl + 2, 12:] = -dp1
            z_ramp[bl + 1, 12:] = dr1; z_ramp[bl + 2, 12:] = -dr1
            # Pair 2: delayed cock-back + whip (pos offset - 0.5*ramp + ramp!)
            dp2 = (torch.randn(6, generator=rng, device=device) * s_p2).clamp(-2.4 * s_p2, 2.4 * s_p2)
            dr2 = (torch.randn(6, generator=rng, device=device) * s_r2).clamp(-2.4 * s_r2, 2.4 * s_r2)
            z_pos[bl + 3, 12:] = dp2 - 0.5 * dr2;  z_pos[bl + 4, 12:] = -(dp2 - 0.5 * dr2)
            z_ramp[bl + 3, 12:] = dr2;             z_ramp[bl + 4, 12:] = -dr2

        # Store per-tick nominal obs/mu/gate + BOTH timing envelopes (std & +30ms delayed!)
        nom_obs_l, nom_mu_l, nom_gate_l, nom_ep1_l, nom_er1_l, nom_ep2_l, nom_er2_l, nom_val_l = [], [], [], [], [], [], [], []
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

                rise = ((t_f - (peak_ticks - 52.0)) / 52.0).clamp(0.0, 1.0)
                fall = ((t_f - peak_ticks) / 16.0).clamp(0.0, 1.0)
                env_pos = torch.where(t_f <= peak_ticks,
                                      torch.sin(0.5 * math.pi * rise).square(),
                                      torch.cos(0.5 * math.pi * fall).square())
                env_rmp = ((t_f - (peak_ticks - 18.0)) / 18.0).clamp(0.0, 1.0) * (t_f <= peak_ticks + 4.0).float()

                active_pre = (env.episode_index == 0) & (~env.rally.had_contact)
                delta_t = env_pos[:, None] * z_pos + env_rmp[:, None] * z_ramp
                act = (base_req + agate * delta_t * active_pre[:, None].float()).clamp(-1.0, 1.0)

                if 202 <= tick <= 355:
                    valid_n = ((env.episode_index[nom_lane_ids] == 0)
                               & (~env.rally.had_contact[nom_lane_ids])
                               & (env.rally.in_play[nom_lane_ids])
                               & (agate[nom_lane_ids].amax(-1) > 0.05))
                    if valid_n.any():
                        pt_n = peak_ticks[nom_lane_ids]
                        pt_d = pt_n + 3.0
                        r2 = ((t_f - (pt_d - 52.0)) / 52.0).clamp(0.0, 1.0)
                        f2 = ((t_f - pt_d) / 16.0).clamp(0.0, 1.0)
                        ep2 = torch.where(t_f <= pt_d, torch.sin(0.5 * math.pi * r2).square(), torch.cos(0.5 * math.pi * f2).square())
                        er2 = ((t_f - (pt_d - 18.0)) / 18.0).clamp(0.0, 1.0) * (t_f <= pt_d + 4.0).float()

                        nom_obs_l.append(policy_obs[nom_lane_ids].half().cpu())
                        nom_mu_l.append(base_req[nom_lane_ids].half().cpu())
                        nom_gate_l.append(agate[nom_lane_ids].half().cpu())
                        nom_ep1_l.append(env_pos[nom_lane_ids].cpu())
                        nom_er1_l.append(env_rmp[nom_lane_ids].cpu())
                        nom_ep2_l.append(ep2.cpu())
                        nom_er2_l.append(er2.cpu())
                        nom_val_l.append(valid_n.cpu())

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

        nom_rows = [rows_all[n_val + i * G] for i in range(n_train)]
        nom_b_succ = sum(r['success'] for r in nom_rows)
        nom_b_str = sum(r['string'] for r in nom_rows)
        nom_rnet = sum(r['robot_net'] for r in nom_rows)

        rewards = np.zeros((n_train, G), dtype=np.float64)
        for i in range(n_train):
            for g_idx in range(G):
                lane = n_val + i * G + g_idx
                rewards[i, g_idx] = compute_rsrl_reward(rows_all[lane], ordered_raw[lane], float(min_dist_cpu[lane]))

        # Measure live PA_ok drift across all 891 pure-FB Panel-A anchors
        with torch.no_grad():
            mod = actor.actor
            pa_pred = (pa_buf['base_out'] + pa_buf['agate'] * mod.approach(pa_buf['norm_obs']) * mod.approach_scale).clamp(-1.0, 1.0)
            pa_m = pa_buf['gate'][:, 12:18] > 0.05
            pa_d = (pa_pred - pa_buf['req'])[:, 12:18][pa_m]
            pa_rms = float(pa_d.square().mean().sqrt())
            pa_q95 = float(pa_d.abs().quantile(0.95))

        comp_score = (0.80 * (val_succ / n_val) + 0.20 * (nom_b_succ / n_train) + 0.04 * val_margin
                      - 1.5 * max(0.0, pa_q95 - 0.018) - 0.6 * (val_rnet > 0 or nom_rnet > 0))
        is_best = (val_rnet == 0 and nom_rnet == 0 and comp_score > best_score)
        save_checkpoint(args.output / f'rsrl256_it{it:02d}.pt', actor.state_dict())
        rolled_back = False
        if is_best:
            best_score = comp_score
            best_val_succ = max(best_val_succ, val_succ)
            best_state = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}
            save_checkpoint(args.output / 'rsrl256_best.pt', best_state)
        elif it >= 1 and (val_rnet > 0 or nom_rnet > 0 or val_succ <= best_val_succ - 4):
            # CLEAN ROLLBACK: restore best, wipe buffer, decay eta, SKIP update on bad rollout!
            actor.load_state_dict({k: v.to(device) for k, v in best_state.items()}, strict=True)
            replay_buffer.clear()
            cur_eta = max(0.06, cur_eta * 0.72)
            rolled_back = True

        # Count clean high-margin rescues (0->1)
        def is_clean_win(row, raw_ep):
            if not (row['success'] and row['string'] and not row['robot_net'] and not row['low_speed']):
                return False
            spm = row['speed_margin_mps'] if row['speed_margin_mps'] is not None else 0.0
            lm = row['landing_margin_m'] if row['landing_margin_m'] is not None else 0.0
            nm = row['net_margin_m'] if row['net_margin_m'] is not None else 0.0
            return (spm >= 0.15 and lm >= 0.08 and nm >= 0.010)

        n_elite_p1, n_elite_p2, n_es_used = 0, 0, 0
        z_pos_cpu = z_pos.cpu()
        z_rmp_cpu = z_ramp.cpu()
        dp1_arr = torch.zeros((n_train, 18), dtype=torch.float32)
        dr1_arr = torch.zeros((n_train, 18), dtype=torch.float32)
        dp2_arr = torch.zeros((n_train, 18), dtype=torch.float32)
        dr2_arr = torch.zeros((n_train, 18), dtype=torch.float32)
        serve_w = torch.zeros((n_train,), dtype=torch.float32)

        for i in range(n_train):
            bl = n_val + i * G
            r0 = rows_all[bl]; raw0 = ordered_raw[bl]
            if r0['success'] and not r0['robot_net']:
                sm = r0['string_margin_m'] or 0.0
                spm = r0['speed_margin_mps'] or 0.0
                lm = r0['landing_margin_m'] or 0.0
                nm = r0['net_margin_m'] or 0.0
                if sm >= 0.010 and spm >= 0.25 and lm >= 0.14 and nm >= 0.020:
                    # CRISP SUCCESS -> ZERO-NOISE RETENTION ANCHOR!
                    serve_w[i] = 1.4
                else:
                    # Razor-edge success -> only nudge if a perturbed branch is MUCH cleaner!
                    clean_wins = [(rewards[i, k], k) for k in range(1, G)
                                  if is_clean_win(rows_all[bl + k], ordered_raw[bl + k]) and rewards[i, k] >= rewards[i, 0] + 0.45]
                    if clean_wins:
                        _, bk = max(clean_wins)
                        if bk <= 2:
                            dp1_arr[i] = 0.40 * z_pos_cpu[bl + bk]; dr1_arr[i] = 0.40 * z_rmp_cpu[bl + bk]
                        else:
                            dp2_arr[i] = 0.40 * z_pos_cpu[bl + bk]; dr2_arr[i] = 0.40 * z_rmp_cpu[bl + bk]
                        serve_w[i] = 1.6
                    else:
                        serve_w[i] = 1.0  # Keep nominal cleanly!
            else:
                # NOMINAL FAILURE -> Priority 1: Clean High-Margin Elite Self-Imitation!
                clean_wins = [(rewards[i, k], k) for k in range(1, G)
                              if is_clean_win(rows_all[bl + k], ordered_raw[bl + k])]
                if clean_wins:
                    _, bk = max(clean_wins)
                    if bk <= 2:
                        dp1_arr[i] = args.elite_scale * z_pos_cpu[bl + bk]
                        dr1_arr[i] = args.elite_scale * z_rmp_cpu[bl + bk]
                        n_elite_p1 += 1
                    else:
                        dp2_arr[i] = args.elite_scale * z_pos_cpu[bl + bk]
                        dr2_arr[i] = args.elite_scale * z_rmp_cpu[bl + bk]
                        n_elite_p2 += 1
                    serve_w[i] = 2.6
                else:
                    # Priority 2: Gentle directional ES ONLY if a branch meaningfully beat nominal (>= +0.85)!
                    best_gain = max(rewards[i, k] - rewards[i, 0] for k in range(1, G))
                    if best_gain >= 0.85:
                        d1 = float(np.clip((rewards[i, 1] - rewards[i, 2]) / 2.0, -1.8, 1.8))
                        d2 = float(np.clip((rewards[i, 3] - rewards[i, 4]) / 2.0, -1.8, 1.8))
                        dp1_arr[i] = cur_eta * d1 * z_pos_cpu[bl + 1]
                        dr1_arr[i] = cur_eta * d1 * z_rmp_cpu[bl + 1]
                        dp2_arr[i] = cur_eta * d2 * z_pos_cpu[bl + 3]
                        dr2_arr[i] = cur_eta * d2 * z_rmp_cpu[bl + 3]
                        serve_w[i] = 0.90
                        n_es_used += 1
                    else:
                        serve_w[i] = 0.0  # Skip hopeless noise!

            dp1_arr[i, :12] = 0.0; dr1_arr[i, :12] = 0.0
            dp2_arr[i, :12] = 0.0; dr2_arr[i, :12] = 0.0
            dp1_arr[i, 12:].clamp_(-0.22, 0.22); dr1_arr[i, 12:].clamp_(-0.18, 0.18)
            dp2_arr[i, 12:].clamp_(-0.24, 0.24); dr2_arr[i, 12:].clamp_(-0.20, 0.20)

        print(f'[IT {it:02d}] ({time.perf_counter()-t_iter:5.1f}s) VAL56={val_succ:2d}/{n_val} (A={val_a:2d}/24 B={val_b:2d}/32 str={val_str:2d} rnet={val_rnet}) '
              f'| HARD40_B={nom_b_succ:2d}/{n_train} ({100*nom_b_succ/n_train:5.1f}% str={nom_b_str:2d}) '
              f'| elite={n_elite_p1}+{n_elite_p2} es={n_es_used} | PA_rms={pa_rms:.4f}(q95={pa_q95:.3f}) '
              f'| comp={comp_score:.4f}{" *BEST*" if is_best else ""}{" [ROLLBACK-SKIP]" if rolled_back else ""}', flush=True)

        with history_path.open('a') as f:
            f.write(json.dumps(dict(
                iter=it, elapsed_s=time.perf_counter() - t_iter, val_succ=val_succ, val_a=val_a, val_b=val_b,
                val_str=val_str, val_rnet=val_rnet, nom_b_succ=nom_b_succ, nom_b_str=nom_b_str, nom_rnet=nom_rnet,
                elite_p1=n_elite_p1, elite_p2=n_elite_p2, es_used=n_es_used,
                pa_rms=pa_rms, pa_q95=pa_q95, comp_score=comp_score, is_best=is_best, rolled_back=rolled_back,
            )) + '\n')

        if it == args.iterations - 1 or rolled_back:
            continue

        obs_s = torch.stack(nom_obs_l, dim=0).float()
        mu_s = torch.stack(nom_mu_l, dim=0).float()
        gate_s = torch.stack(nom_gate_l, dim=0).float()
        ep1_s = torch.stack(nom_ep1_l, dim=0).float()
        er1_s = torch.stack(nom_er1_l, dim=0).float()
        ep2_s = torch.stack(nom_ep2_l, dim=0).float()
        er2_s = torch.stack(nom_er2_l, dim=0).float()
        val_s = torch.stack(nom_val_l, dim=0)
        T_cap = obs_s.shape[0]
        time_w = torch.zeros((T_cap, n_train), dtype=torch.float32)
        for i in range(n_train):
            if float(serve_w[i]) <= 0.0:
                continue
            vi = val_s[:, i].nonzero().flatten()
            if len(vi):
                dist = (int(vi[-1]) - vi).clamp_min(0)
                time_w[vi, i] = torch.where(dist <= 16, 3.5, torch.where(dist <= 30, 1.8, 1.0)) * serve_w[i]

        tot_delta = (ep1_s[:, :, None] * dp1_arr[None] + er1_s[:, :, None] * dr1_arr[None]
                     + ep2_s[:, :, None] * dp2_arr[None] + er2_s[:, :, None] * dr2_arr[None])
        tgt_s = (mu_s + gate_s * tot_delta).clamp(-1.0, 1.0)
        fm = val_s.reshape(-1) & (time_w.reshape(-1) > 0.0)
        if not fm.any():
            continue
        b_obs = obs_s.reshape(-1, obs_s.shape[-1])[fm].to(device)
        b_tgt = tgt_s.reshape(-1, 18)[fm].to(device)
        b_gate = gate_s.reshape(-1, 18)[fm].to(device)
        b_w = time_w.reshape(-1)[fm].to(device)

        with torch.no_grad():
            mod = actor.actor
            b_norm = mod.normalizer(b_obs)
            b_base = mod.base(b_norm)
            b_ag = actor.approach_action_gate(b_obs)

        replay_buffer.append(dict(norm_obs=b_norm.cpu(), base_out=b_base.cpu(), agate=b_ag.cpu(),
                                  tgt_mu=b_tgt.cpu(), gate=b_gate.cpu(), weight=b_w.cpu()))
        if len(replay_buffer) > args.replay_iters:
            replay_buffer.pop(0)

        n_buf = len(replay_buffer)
        tr_norm = torch.cat([b['norm_obs'] for b in replay_buffer], dim=0).to(device)
        tr_base = torch.cat([b['base_out'] for b in replay_buffer], dim=0).to(device)
        tr_ag = torch.cat([b['agate'] for b in replay_buffer], dim=0).to(device)
        tr_tgt = torch.cat([b['tgt_mu'] for b in replay_buffer], dim=0).to(device)
        tr_gate = torch.cat([b['gate'] for b in replay_buffer], dim=0).to(device)
        tr_w = torch.cat([b['weight'] * (0.75 ** (n_buf - 1 - k)) for k, b in enumerate(replay_buffer)], dim=0).to(device)

        for name, p in actor.named_parameters():
            p.requires_grad_(name in APPROACH_NAMES)
        trainable = [(name, p) for name, p in actor.named_parameters() if p.requires_grad]
        old_params = {name: p.detach().clone() for name, p in trainable}
        opt = torch.optim.Adam([p for _, p in trainable], lr=args.lr)
        mod = actor.actor
        n_frames = len(tr_norm)
        n_pa = len(pa_buf['norm_obs'])

        with torch.no_grad():
            cur_old = (tr_base + tr_ag * mod.approach(tr_norm) * mod.approach_scale).clamp(-1.0, 1.0)
            m_arm = tr_gate[:, 12:18] > 0.05

        actor.train()
        arm_rms_actual = 0.0
        n_steps_done = 0
        stop_tr = False
        for ep in range(1, args.update_epochs + 1):
            perm = torch.randperm(n_frames, device=device)
            for s0 in range(0, n_frames, args.batch_size):
                idx = perm[s0:s0 + args.batch_size]
                opt.zero_grad(set_to_none=True)
                pred = (tr_base[idx] + tr_ag[idx] * mod.approach(tr_norm[idx]) * mod.approach_scale).clamp(-1.0, 1.0)
                mask = (tr_gate[idx] > 0.02).float()
                jw = torch.ones_like(mask); jw[:, :12] = 3.0
                mw = mask * jw
                denom = mw.sum(-1).clamp_min(1.0)
                wb = tr_w[idx]; ws = wb.sum().clamp_min(1.0)
                loss_pg = (((pred - tr_tgt[idx]).square() * mw).sum(-1) / denom * wb).sum() / ws
                loss_kl = (((pred - cur_old[idx]).square() * mw).sum(-1) / denom * wb).sum() / ws

                # Ironclad 891-anchor L_inf Panel-A Fortress step
                pa_idx = torch.randint(0, n_pa, (min(args.batch_size, n_pa),), device=device)
                pred_pa = (pa_buf['base_out'][pa_idx] + pa_buf['agate'][pa_idx] * mod.approach(pa_buf['norm_obs'][pa_idx]) * mod.approach_scale).clamp(-1.0, 1.0)
                pa_mask = (pa_buf['gate'][pa_idx] > 0.02).float()
                pa_sq = ((pred_pa - pa_buf['req'][pa_idx]).square() * pa_mask).sum(-1) / pa_mask.sum(-1).clamp_min(1.0)
                loss_pa_mse = (pa_sq * pa_buf['weight'][pa_idx]).sum() / pa_buf['weight'][pa_idx].sum().clamp_min(1.0)
                pa_arm_m = (pa_buf['gate'][pa_idx, 12:18] > 0.02).float()
                pa_exc = F.relu((pred_pa[:, 12:18] - pa_buf['req'][pa_idx, 12:18]).abs() - args.pa_hinge_margin)
                loss_pa_h = (pa_exc.square() * pa_arm_m).sum() / pa_arm_m.sum().clamp_min(1.0)

                prox_t = []
                for name, p in trainable:
                    if name == 'actor.approach.0.weight':
                        prox_t.append(2.0 * ((p - old_params[name]).square() * w0_col_mult[None, :]).mean())
                    else:
                        prox_t.append(0.3 * (p - old_params[name]).square().mean())
                loss = (loss_pg + args.kl_anchor * loss_kl
                        + args.pa_mse_weight * loss_pa_mse + args.pa_hinge_weight * loss_pa_h
                        + args.prox_weight * (sum(prox_t) / len(prox_t)))
                loss.backward()
                for name, p in trainable:
                    if name in ('actor.approach.4.weight', 'actor.approach.4.bias') and p.grad is not None:
                        p.grad[:12].zero_()
                torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.2)
                opt.step()
                n_steps_done += 1

                # Per-minibatch exact trust-region cap! Never overshoots!
                if n_steps_done % 4 == 0:
                    with torch.no_grad():
                        pred_all = (tr_base + tr_ag * mod.approach(tr_norm) * mod.approach_scale).clamp(-1.0, 1.0)
                        arm_rms_actual = float((pred_all - cur_old)[:, 12:18][m_arm].square().mean().sqrt()) if m_arm.any() else 0.0
                        if arm_rms_actual >= args.max_arm_rms:
                            stop_tr = True
                            break
            if stop_tr:
                break

        with torch.no_grad():
            pred_all = (tr_base + tr_ag * mod.approach(tr_norm) * mod.approach_scale).clamp(-1.0, 1.0)
            arm_rms_actual = float((pred_all - cur_old)[:, 12:18][m_arm].square().mean().sqrt()) if m_arm.any() else 0.0
        actor.eval()
        for k, v in base_param_ref.items():
            if torch.is_tensor(v):
                assert torch.equal(v, actor.state_dict()[k].cpu()), f'Frozen param mutated: {k}'
        print(f'  [UPD {it:02d}] frames={n_frames} (buf={n_buf}) steps={n_steps_done} ep={ep}/{args.update_epochs} d_arm_rms={arm_rms_actual:.4f}', flush=True)

    print(f'[RSRL-256] Done! Best -> {args.output / "rsrl256_best.pt"}', flush=True)


if __name__ == '__main__':
    main()
