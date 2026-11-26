#!/usr/bin/env python3
"""Resident 1024-env Paired-GRPO (Antithetic Smooth-Basis RL) directly in real PhysX.

Preserves 100% of the native PhysX physics, G07 execution contract, causal recovery gate,
and scoring definitions while running 1,024 parallel environments continuously across
iterations without restarting Isaac Sim:
  - Lanes 0..63: Live deterministic evaluation on strictly disjoint val64_select (never trained on).
  - Lanes 64..1023: 192 strictly disjoint fresh serves per iteration x G=5 paired rollouts
    (1 deterministic nominal + 2 antithetic smooth-basis pairs (+z_a, -z_a, +z_b, -z_b)).

Updates ONLY actor.approach.* on unperturbed nominal observations (eliminating the
action-history causal shortcut) via KL-regularized mirror-descent / antithetic policy
gradient + elite rescue self-imitation, directly optimizing real PhysX complete return
and multi-dimensional safety margins (string, speed, landing, net clearance).
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
    return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--val-spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--g07-execution-root', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=882072)
    parser.add_argument('--train-seed-base', type=int, default=940000)
    parser.add_argument('--iterations', type=int, default=6)
    parser.add_argument('--n-serves-a', type=int, default=64)
    parser.add_argument('--n-serves-b', type=int, default=128)
    parser.add_argument('--group-size', type=int, default=5, help='1 nominal + (G-1)/2 antithetic pairs (must be odd >= 3)')
    parser.add_argument('--sigma-arm-pos', type=float, default=0.085)
    parser.add_argument('--sigma-arm-ramp', type=float, default=0.075)
    parser.add_argument('--sigma-leg-pos', type=float, default=0.030)
    parser.add_argument('--sigma-decay', type=float, default=0.94)
    parser.add_argument('--step-eta', type=float, default=0.32)
    parser.add_argument('--elite-blend', type=float, default=0.40, help='Extra pull toward 0->1 rescued branch on failing serves')
    parser.add_argument('--kl-anchor', type=float, default=0.35, help='Action-space trust-region anchor weight')
    parser.add_argument('--prox-weight', type=float, default=0.05, help='Parameter proximal anchor weight')
    parser.add_argument('--lr', type=float, default=1.2e-4)
    parser.add_argument('--update-epochs', type=int, default=45)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--max-arm-rms', type=float, default=0.024, help='Hard per-iteration trust-region cap on arm action RMS')
    parser.add_argument('--replay-iters', type=int, default=2, help='Keep last K iterations of nominal GRPO targets in buffer')
    parser.add_argument('--max-ticks', type=int, default=530)
    parser.add_argument('--physics-device', default='cuda:0')
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    if args.group_size < 3 or args.group_size % 2 != 1:
        raise ValueError('--group-size must be odd >= 3 (1 nominal + antithetic pairs)')

    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    fault_fd = os.open(args.output / 'native_fault.log', os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    faulthandler.enable(file=fault_fd, all_threads=True)
    stop_request = {'signal': None}
    app = AppLauncher(args, fast_shutdown=False).app
    faulthandler.enable(file=fault_fd, all_threads=True)
    signal.signal(signal.SIGINT, lambda s, f: stop_request.__setitem__('signal', signal.Signals(s).name))
    signal.signal(signal.SIGTERM, lambda s, f: stop_request.__setitem__('signal', signal.Signals(s).name))

    import traceback
    try:
        _run_grpo(args, stop_request)
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    else:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


def compute_grpo_reward(row, min_dist_m: float) -> float:
    succ = 1.0 if row['success'] else 0.0
    qstr = 1.0 if row['string'] else 0.0
    rnet = 1.0 if row['robot_net'] else 0.0
    sm = row['string_margin_m']
    sm_t = max(-1.5, min(1.0, sm / 0.040)) if (sm is not None and math.isfinite(sm)) else -1.5
    spm = row['speed_margin_mps']
    spm_t = max(-1.5, min(1.0, spm / 0.55)) if (qstr > 0 and spm is not None and math.isfinite(spm)) else -0.5
    lm = row['landing_margin_m']
    lm_t = max(-1.5, min(1.0, lm / 0.35)) if (qstr > 0 and lm is not None and math.isfinite(lm)) else -1.0
    nm = row['net_margin_m']
    nm_t = max(-1.0, min(1.0, nm / 0.12)) if (qstr > 0 and nm is not None and math.isfinite(nm)) else -0.5
    had_c = (row['string'] or row['frame'] or (sm is not None))
    dist_pen = 0.0 if had_c else -2.0 * max(0.0, min(0.40, min_dist_m - 0.10))
    return float(
        3.50 * succ
        + 1.20 * qstr
        + 0.70 * sm_t
        + 0.55 * spm_t
        + 0.70 * lm_t
        + 0.35 * nm_t
        + dist_pen
        - 6.00 * rnet
    )


def _run_grpo(args, stop_request):
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
    from quadsmash.rl.supermimic_native_approach_search import native_fitness
    from quadsmash.tasks.stand.observation_contract import inverse_rotate
    from quadsmash.tasks.stand.supermimic_env_v1 import SuperMimicEnvV1
    from quadsmash.tasks.stand.full_racket_eval_env_v4 import FrozenPanelEnv

    class EvaluationEnv(SuperMimicEnvV1, FrozenPanelEnv):
        pass

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    val_spec = json.loads(args.val_spec.read_text())
    val_cases = val_spec['cases']
    n_val = len(val_cases)
    n_train_serves = args.n_serves_a + args.n_serves_b
    G = args.group_size
    n_pairs = (G - 1) // 2
    lanes = n_val + n_train_serves * G
    print(f'[GRPO] Starting resident scene: lanes={lanes} (val={n_val} + train={n_train_serves}x{G})', flush=True)

    execution_root = args.g07_execution_root
    reserved = reserved_case_ids(execution_root)
    reserved.update(c['case_id'] for c in val_cases)

    def make_iter_cases(it_idx: int):
        sa = args.train_seed_base + it_idx * 10 + 1
        sb = args.train_seed_base + it_idx * 10 + 2
        ca = incoming_cases(args.n_serves_a, sa, 'A')
        cb = incoming_cases(args.n_serves_b, sb, 'B')
        train_serves = ca + cb
        overlap = {c['case_id'] for c in train_serves} & reserved
        if overlap:
            raise ValueError(f'Training batch overlaps reserved eval panel: {sorted(overlap)[:5]}')
        # Layout: [val_cases (0..n_val-1), serve0 x G, serve1 x G, ...]
        full = list(val_cases)
        for c in train_serves:
            for _ in range(G):
                full.append(c)
        return train_serves, full

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
    env = EvaluationEnv(cfg, cases=full_cases_0, reset_noise=noise,
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

    # Precompute string polygon geometry for native_fitness scoring
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

    history_path = args.output / 'grpo_history.jsonl'
    best_score = -1e9
    best_state = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}
    replay_buffer = []  # list of dicts with obs, base_out, norm_obs, agate, old_mu, target_mu, gate, weight

    # Index helpers for nominal lanes of training serves: n_val + i * G
    nom_lane_ids = torch.tensor([n_val + i * G for i in range(n_train_serves)], dtype=torch.long, device=device)
    base_param_ref = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items() if k not in APPROACH_NAMES}

    for it in range(args.iterations):
        if stop_request['signal']:
            print(f'[GRPO] Stopped by signal {stop_request["signal"]}', flush=True)
            break

        t_iter = time.perf_counter()
        if it == 0:
            train_serves, full_cases = train_serves_0, full_cases_0
            obs, _ = wrapper.reset()
        else:
            train_serves, full_cases = make_iter_cases(it)
            env._episode_has_stepped.zero_()
            env.episode_index.zero_()
            env.sweep_misses.zero_()
            env.sweep_unresolved.zero_()
            env.rally_records.clear()
            env.episode_records.clear()
            env.racket_plane_records.clear()
            env.first_approach_contact_motion.clear()
            env.initial_states.clear()
            env.cases = full_cases
            obs, _ = wrapper.reset()

        # Generate antithetic smooth-basis exploration perturbations for training lanes
        decay = args.sigma_decay ** it
        s_arm_p = args.sigma_arm_pos * decay
        s_arm_r = args.sigma_arm_ramp * decay
        s_leg_p = args.sigma_leg_pos * decay

        rng = torch.Generator(device=device)
        rng.manual_seed(args.seed + 7919 * (it + 1))

        # Shape [lanes, 18] for pos & ramp perturbations; 0 for val lanes and nominal branch 0
        z_pos = torch.zeros((lanes, 18), dtype=torch.float32, device=device)
        z_ramp = torch.zeros((lanes, 18), dtype=torch.float32, device=device)
        peak_ticks = torch.full((lanes,), 275.0, dtype=torch.float32, device=device)
        for i, c in enumerate(val_cases):
            peak_ticks[i] = round((2.0 + float(c['prediction_only']['flight_time_s'])) * 100.0)
        for i, c in enumerate(train_serves):
            pt = round((2.0 + float(c['prediction_only']['flight_time_s'])) * 100.0)
            base_lane = n_val + i * G
            peak_ticks[base_lane:base_lane + G] = pt
            for p_idx in range(n_pairs):
                lp = base_lane + 1 + 2 * p_idx
                lm = base_lane + 2 + 2 * p_idx
                dp_arm = torch.randn(6, generator=rng, device=device) * s_arm_p
                dr_arm = torch.randn(6, generator=rng, device=device) * s_arm_r
                dp_leg = torch.randn(12, generator=rng, device=device) * s_leg_p if s_leg_p > 0 else torch.zeros(12, device=device)
                # Clamp individual draws at 2.5 sigma to prevent extreme outliers
                dp_arm.clamp_(-2.5 * s_arm_p, 2.5 * s_arm_p)
                dr_arm.clamp_(-2.5 * s_arm_r, 2.5 * s_arm_r)
                if s_leg_p > 0:
                    dp_leg.clamp_(-2.5 * s_leg_p, 2.5 * s_leg_p)
                z_pos[lp, :12] = dp_leg;  z_pos[lm, :12] = -dp_leg
                z_pos[lp, 12:] = dp_arm;  z_pos[lm, 12:] = -dp_arm
                z_ramp[lp, 12:] = dr_arm; z_ramp[lm, 12:] = -dr_arm

        # Track nominal observations + active masks ONLY for ticks 200..360 (pre-contact window)
        nom_obs_list, nom_mu_list, nom_gate_list, nom_env_pos_list, nom_env_rmp_list, nom_valid_list = [], [], [], [], [], []
        nom_tick_list = []
        min_dist = torch.full((lanes,), 9.9, dtype=torch.float32, device=device)
        first = {}
        cursor = 0
        actor.eval()

        with torch.inference_mode():
            for tick in range(args.max_ticks):
                if stop_request['signal']:
                    break
                policy_obs = obs['policy']
                base_req = policy_forward(actor, policy_obs, env.rally, mode=policy_execution['recovery_gate'], sample=False)[0]
                agate = actor.approach_action_gate(policy_obs)  # [lanes, 18]

                # Smooth C1 temporal envelopes aligned to each serve's predicted intercept tick
                t_f = float(tick)
                rise = ((t_f - (peak_ticks - 50.0)) / 50.0).clamp(0.0, 1.0)
                fall = ((t_f - peak_ticks) / 16.0).clamp(0.0, 1.0)
                env_pos = torch.where(t_f <= peak_ticks,
                                      torch.sin(0.5 * math.pi * rise).square(),
                                      torch.cos(0.5 * math.pi * fall).square())
                env_rmp = ((t_f - (peak_ticks - 18.0)) / 18.0).clamp(0.0, 1.0) * (t_f <= peak_ticks + 4.0).float()

                # Apply perturbation strictly on pre-contact active approach gate for non-nominal training lanes
                active_pre = (env.episode_index == 0) & (~env.rally.had_contact)
                delta_t = env_pos[:, None] * z_pos + env_rmp[:, None] * z_ramp
                act = (base_req + agate * delta_t * active_pre[:, None].float()).clamp(-1.0, 1.0)

                # Capture unperturbed nominal frames (only for the 192 nominal training lanes!)
                if 200 <= tick <= 360:
                    valid_n = ((env.episode_index[nom_lane_ids] == 0)
                               & (~env.rally.had_contact[nom_lane_ids])
                               & (env.rally.in_play[nom_lane_ids])
                               & (agate[nom_lane_ids].amax(-1) > 0.05))
                    if valid_n.any():
                        nom_obs_list.append(policy_obs[nom_lane_ids].half().cpu())
                        nom_mu_list.append(base_req[nom_lane_ids].half().cpu())
                        nom_gate_list.append(agate[nom_lane_ids].half().cpu())
                        nom_env_pos_list.append(env_pos[nom_lane_ids].cpu())
                        nom_env_rmp_list.append(env_rmp[nom_lane_ids].cpu())
                        nom_valid_list.append(valid_n.cpu())
                        nom_tick_list.append(tick)

                # Track minimum ball-to-racket distance for NO_CONTACT shaping
                in_flight = (env.episode_index == 0) & env.rally.in_play & (~env.rally.had_contact)
                if in_flight.any():
                    d_br = (env.shuttle.data.root_pos_w - env.robot.data.body_link_pos_w[:, env.racket_body_id]).norm(dim=-1)
                    min_dist = torch.where(in_flight, torch.minimum(min_dist, d_br), min_dist)

                if env.approach_ball_termination is not None:
                    env.record_approach_gate(agate)
                obs, _, _, _ = wrapper.step(act)

                for raw in env.rally_records[cursor:]:
                    if raw['episode_index'] == 0 and raw['env_id'] not in first:
                        first[raw['env_id']] = score_terminal_episode(raw, scoring_protocol)
                cursor = len(env.rally_records)

                if (tick + 1) % 100 == 0 or len(first) >= lanes:
                    print(f'  [it {it:02d}] tick={tick+1:3d} settled={len(first):4d}/{lanes} elapsed={time.perf_counter()-t_iter:5.1f}s', flush=True)
                if len(first) >= lanes:
                    break

        if len(first) < lanes:
            print(f'[GRPO] Warning: only {len(first)}/{lanes} settled by max_ticks={args.max_ticks}', flush=True)
            break

        raw_ep0 = {row['env_id']: row for row in env.rally_records if row['episode_index'] == 0}
        ordered_raw = [raw_ep0[i] for i in range(lanes)]
        ordered = [first[i] for i in range(lanes)]
        fit_all = native_fitness(ordered_raw, env.court_config, signed_string_margin)
        rows_all = fit_all['rows']
        min_dist_cpu = min_dist.cpu().numpy()

        # 1. Live disjoint val64_select metrics (NEVER trained on!)
        val_rows = rows_all[:n_val]
        val_succ = sum(r['success'] for r in val_rows)
        val_str = sum(r['string'] for r in val_rows)
        val_frm = sum(r['frame'] for r in val_rows)
        val_noc = sum(not r['string'] and not r['frame'] for r in val_rows)
        val_low = sum(r['low_speed'] for r in val_rows)
        val_bnet = sum(r['ball_net'] for r in val_rows)
        val_rnet = sum(r['robot_net'] for r in val_rows)
        val_ret = float(np.mean([ordered[i]['native_episode_return'] for i in range(n_val)]))
        val_margin = float(np.mean([r['margin_score'] for r in val_rows]))

        # 2. Live unseen nominal training serves (Branch 0 of each of the 192 fresh serves!)
        nom_rows = [rows_all[n_val + i * G] for i in range(n_train_serves)]
        nom_succ_a = sum(r['success'] for r in nom_rows[:args.n_serves_a])
        nom_succ_b = sum(r['success'] for r in nom_rows[args.n_serves_a:])
        nom_str_a = sum(r['string'] for r in nom_rows[:args.n_serves_a])
        nom_str_b = sum(r['string'] for r in nom_rows[args.n_serves_a:])
        nom_rnet = sum(r['robot_net'] for r in nom_rows)
        eq_succ_rate = 0.5 * (nom_succ_a / args.n_serves_a) + 0.5 * (nom_succ_b / args.n_serves_b)
        eq_str_rate = 0.5 * (nom_str_a / args.n_serves_a) + 0.5 * (nom_str_b / args.n_serves_b)

        # Compute exact GRPO rewards & antithetic pair differences for all 192 training serves
        rewards = np.zeros((n_train_serves, G), dtype=np.float64)
        for i in range(n_train_serves):
            for g_idx in range(G):
                lane = n_val + i * G + g_idx
                rewards[i, g_idx] = compute_grpo_reward(rows_all[lane], float(min_dist_cpu[lane]))

        pair_diffs = np.zeros((n_train_serves, n_pairs), dtype=np.float64)
        for p_idx in range(n_pairs):
            pair_diffs[:, p_idx] = rewards[:, 1 + 2 * p_idx] - rewards[:, 2 + 2 * p_idx]
        diff_std = float(np.std(pair_diffs)) + 1e-6
        adv_pairs = np.clip(pair_diffs / diff_std, -2.5, 2.5)  # [n_train_serves, n_pairs]

        # Count stochastic rescues (0 -> 1) and regressions (1 -> 0) for diagnostics
        rescued_serves = 0
        improved_margin_serves = 0
        total_rnet_all = sum(r['robot_net'] for r in rows_all)
        for i in range(n_train_serves):
            nom_ok = rows_all[n_val + i * G]['success']
            branch_oks = [rows_all[n_val + i * G + k]['success'] and not rows_all[n_val + i * G + k]['robot_net']
                          for k in range(1, G)]
            if not nom_ok and any(branch_oks):
                rescued_serves += 1
            if max(rewards[i, 1:]) > rewards[i, 0] + 0.15:
                improved_margin_serves += 1

        print(f'\n[ITER {it:02d}] VAL64: succ={val_succ:2d}/{n_val} ({100*val_succ/n_val:5.1f}%) str={val_str:2d} '
              f'frm={val_frm:2d} noc={val_noc} low={val_low:2d} bnet={val_bnet:2d} rnet={val_rnet} ret={val_ret:.2f} m={val_margin:.3f}', flush=True)
        print(f'          UNSEEN192 (nom): eq_succ={100*eq_succ_rate:5.1f}% (A={nom_succ_a}/{args.n_serves_a}={100*nom_succ_a/args.n_serves_a:.1f}%, '
              f'B={nom_succ_b}/{args.n_serves_b}={100*nom_succ_b/args.n_serves_b:.1f}%) | eq_str={100*eq_str_rate:5.1f}% | '
              f'rescued_0->1={rescued_serves}/{n_train_serves - (nom_succ_a + nom_succ_b)} | imp_R={improved_margin_serves}/{n_train_serves} | all_rnet={total_rnet_all}', flush=True)

        # Save current checkpoint & track best by unbiased composite score (val64 + unseen192 equal-panel)
        ckpt_it = args.output / f'grpo_it{it:02d}.pt'
        save_checkpoint(ckpt_it, actor.state_dict())
        comp_score = 0.55 * eq_succ_rate + 0.45 * (val_succ / n_val) + 0.05 * val_margin - 0.5 * (val_rnet > 0 or nom_rnet > 0)
        is_best = comp_score > best_score
        if is_best:
            best_score = comp_score
            best_state = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}
            save_checkpoint(args.output / 'grpo_best.pt', best_state)
            print(f'  >>> NEW BEST grpo_best.pt @ it{it:02d} (comp={comp_score:.4f}, val64={val_succ}/{n_val}, eq192={100*eq_succ_rate:.1f}%)', flush=True)

        rec = dict(
            iter=it, elapsed_s=time.perf_counter() - t_iter,
            sigma_arm_pos=s_arm_p, sigma_arm_ramp=s_arm_r, sigma_leg_pos=s_leg_p,
            val_succ=val_succ, val_str=val_str, val_frm=val_frm, val_noc=val_noc,
            val_low=val_low, val_bnet=val_bnet, val_rnet=val_rnet, val_ret=val_ret, val_margin=val_margin,
            nom_succ_a=nom_succ_a, nom_succ_b=nom_succ_b, eq_succ_rate=eq_succ_rate, eq_str_rate=eq_str_rate,
            nom_rnet=nom_rnet, all_rnet=total_rnet_all, rescued_0_to_1=rescued_serves,
            improved_R=improved_margin_serves, diff_std=diff_std, comp_score=comp_score, is_best=is_best,
        )
        with history_path.open('a') as f:
            f.write(json.dumps(rec) + '\n')

        # Don't run an extra update after the final evaluation iteration if desired;
        # actually DO update after every iteration AND run a final post-update iteration if it == iterations-1?
        # Here we run `iterations` rollouts and update after it=0..iterations-2, OR update on all and evaluate it=iterations.
        if it == args.iterations - 1:
            break

        # 3. Build per-serve smooth GRPO targets on clean unperturbed nominal observations
        z_pos_cpu = z_pos.cpu()
        z_rmp_cpu = z_ramp.cpu()
        # Per-serve combined delta coefficients [n_train_serves, 18] for pos & ramp
        delta_pos_serve = torch.zeros((n_train_serves, 18), dtype=torch.float32)
        delta_rmp_serve = torch.zeros((n_train_serves, 18), dtype=torch.float32)
        serve_weight = torch.ones((n_train_serves,), dtype=torch.float32)

        for i in range(n_train_serves):
            base_lane = n_val + i * G
            # Antithetic ES / GRPO direction: average over the n_pairs antithetic pairs
            dp = torch.zeros(18, dtype=torch.float32)
            dr = torch.zeros(18, dtype=torch.float32)
            for p_idx in range(n_pairs):
                lp = base_lane + 1 + 2 * p_idx
                a_p = float(adv_pairs[i, p_idx])
                dp = dp + a_p * z_pos_cpu[lp]
                dr = dr + a_p * z_rmp_cpu[lp]
            dp = args.step_eta * (dp / n_pairs)
            dr = args.step_eta * (dr / n_pairs)

            # Elite Self-Imitation boost when nominal failed (0) and a safe stochastic branch succeeded (1)!
            nom_ok = rows_all[base_lane]['success']
            if not nom_ok and args.elite_blend > 0:
                winners = [(rewards[i, k], k) for k in range(1, G)
                           if rows_all[base_lane + k]['success'] and not rows_all[base_lane + k]['robot_net']]
                if winners:
                    _, best_k = max(winners)
                    lk = base_lane + best_k
                    dp = (1.0 - args.elite_blend) * dp + args.elite_blend * z_pos_cpu[lk]
                    dr = (1.0 - args.elite_blend) * dr + args.elite_blend * z_rmp_cpu[lk]
                    serve_weight[i] = 1.6  # higher weight on confirmed 0->1 rescues!
            elif not nom_ok:
                serve_weight[i] = 1.25  # mild extra weight on remaining failures

            # Clamp per-serve target shift safely
            dp[:12].clamp_(-0.08, 0.08)
            dp[12:].clamp_(-0.22, 0.22)
            dr[12:].clamp_(-0.20, 0.20)
            delta_pos_serve[i] = dp
            delta_rmp_serve[i] = dr

        # Stack nominal pre-contact frames [T_cap, n_train_serves, ...]
        obs_stack = torch.stack(nom_obs_list, dim=0).float()      # [T, S, 794]
        mu_stack = torch.stack(nom_mu_list, dim=0).float()        # [T, S, 18]
        gate_stack = torch.stack(nom_gate_list, dim=0).float()    # [T, S, 18]
        ep_stack = torch.stack(nom_env_pos_list, dim=0).float()   # [T, S]
        er_stack = torch.stack(nom_env_rmp_list, dim=0).float()   # [T, S]
        val_stack = torch.stack(nom_valid_list, dim=0)            # [T, S] bool

        # Temporal emphasis near impact (last 15 ticks before each serve's last valid tick get 3.5x)
        T_cap = obs_stack.shape[0]
        time_w = torch.ones((T_cap, n_train_serves), dtype=torch.float32)
        for i in range(n_train_serves):
            v_idx = val_stack[:, i].nonzero().flatten()
            if len(v_idx):
                last_t = int(v_idx[-1])
                dist = (last_t - v_idx).clamp_min(0)
                time_w[v_idx, i] = torch.where(dist <= 15, 3.5, torch.where(dist <= 28, 1.8, 1.0)) * serve_weight[i]

        delta_frame = ep_stack[:, :, None] * delta_pos_serve[None, :, :] + er_stack[:, :, None] * delta_rmp_serve[None, :, :]
        target_stack = (mu_stack + gate_stack * delta_frame).clamp(-1.0, 1.0)

        flat_mask = val_stack.reshape(-1)
        batch_obs = obs_stack.reshape(-1, obs_stack.shape[-1])[flat_mask].to(device)
        batch_old_mu = mu_stack.reshape(-1, 18)[flat_mask].to(device)
        batch_tgt_mu = target_stack.reshape(-1, 18)[flat_mask].to(device)
        batch_gate = gate_stack.reshape(-1, 18)[flat_mask].to(device)
        batch_w = time_w.reshape(-1)[flat_mask].to(device)

        with torch.no_grad():
            mod = actor.actor
            batch_norm = mod.normalizer(batch_obs)
            batch_base = mod.base(batch_norm)
            batch_agate = actor.approach_action_gate(batch_obs)

        replay_buffer.append(dict(
            norm_obs=batch_norm.cpu(), base_out=batch_base.cpu(), agate=batch_agate.cpu(),
            old_mu=batch_old_mu.cpu(), tgt_mu=batch_tgt_mu.cpu(), gate=batch_gate.cpu(), weight=batch_w.cpu(),
        ))
        if len(replay_buffer) > args.replay_iters:
            replay_buffer.pop(0)

        # Concatenate recent on-policy replay buffer (exponential recency weight 0.65 for previous iter)
        n_buf = len(replay_buffer)
        tr_norm = torch.cat([b['norm_obs'] for b in replay_buffer], dim=0).to(device)
        tr_base = torch.cat([b['base_out'] for b in replay_buffer], dim=0).to(device)
        tr_agate = torch.cat([b['agate'] for b in replay_buffer], dim=0).to(device)
        tr_old = torch.cat([b['old_mu'] for b in replay_buffer], dim=0).to(device)
        tr_tgt = torch.cat([b['tgt_mu'] for b in replay_buffer], dim=0).to(device)
        tr_gate = torch.cat([b['gate'] for b in replay_buffer], dim=0).to(device)
        tr_w = torch.cat([b['weight'] * (0.65 ** (n_buf - 1 - k)) for k, b in enumerate(replay_buffer)], dim=0).to(device)

        # 4. KL-Trust-Region update on actor.approach.* ONLY
        for name, p in actor.named_parameters():
            p.requires_grad_(name in APPROACH_NAMES)
        trainable = [(name, p) for name, p in actor.named_parameters() if p.requires_grad]
        old_params = {name: p.detach().clone() for name, p in trainable}
        opt = torch.optim.Adam([p for _, p in trainable], lr=args.lr)
        n_frames = len(tr_norm)
        steps_ep = max(1, math.ceil(n_frames / args.batch_size))
        mod = actor.actor

        # Compute current-iter old predictions for exact KL anchor relative to current policy
        with torch.no_grad():
            cur_old_all = (tr_base + tr_agate * mod.approach(tr_norm) * mod.approach_scale).clamp(-1.0, 1.0)
            m_arm_all = tr_gate[:, 12:18] > 0.05
            m_leg_all = tr_gate[:, 0:12] > 0.05

        actor.train()
        final_ep = 0
        arm_rms_actual = 0.0
        leg_rms_actual = 0.0
        for ep in range(1, args.update_epochs + 1):
            final_ep = ep
            perm = torch.randperm(n_frames, device=device)
            for s0 in range(0, n_frames, args.batch_size):
                idx = perm[s0:s0 + args.batch_size]
                opt.zero_grad(set_to_none=True)
                pred = (tr_base[idx] + tr_agate[idx] * mod.approach(tr_norm[idx]) * mod.approach_scale).clamp(-1.0, 1.0)
                mask = (tr_gate[idx] > 0.02).float()
                # Joint weights: 1.0 on arm 12:18, 1.5 on legs 0:12 (extra stability for legs)
                jw = torch.ones_like(mask)
                jw[:, :12] = 1.5
                mw = mask * jw
                denom = mw.sum(-1).clamp_min(1.0)
                w_b = tr_w[idx]
                w_sum = w_b.sum().clamp_min(1.0)

                loss_pg = (((pred - tr_tgt[idx]).square() * mw).sum(-1) / denom * w_b).sum() / w_sum
                loss_kl = (((pred - cur_old_all[idx]).square() * mw).sum(-1) / denom * w_b).sum() / w_sum
                prox = sum((p - old_params[n]).square().mean() for n, p in trainable) / len(trainable)
                loss = loss_pg + args.kl_anchor * loss_kl + args.prox_weight * prox
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.5)
                opt.step()

            # Check hard trust-region cap on arm action RMS
            with torch.no_grad():
                pred_all = (tr_base + tr_agate * mod.approach(tr_norm) * mod.approach_scale).clamp(-1.0, 1.0)
                diff = pred_all - cur_old_all
                arm_rms_actual = float(diff[:, 12:18][m_arm_all].square().mean().sqrt()) if m_arm_all.any() else 0.0
                leg_rms_actual = float(diff[:, 0:12][m_leg_all].square().mean().sqrt()) if m_leg_all.any() else 0.0
                if arm_rms_actual >= args.max_arm_rms:
                    break

        actor.eval()
        # Verify frozen parameters (base, normalizer, recovery) are 100% untouched!
        for k, v in base_param_ref.items():
            if torch.is_tensor(v):
                assert torch.equal(v, actor.state_dict()[k].cpu()), f'Frozen param mutated: {k}'

        dp_rms = float(torch.cat([(p.detach() - old_params[n]).flatten() for n, p in trainable]).square().mean().sqrt())
        print(f'  [UPDATE it {it:02d}] frames={n_frames} (buf={n_buf}) ep={final_ep}/{args.update_epochs} '
              f'd_arm_rms={arm_rms_actual:.4f} d_leg_rms={leg_rms_actual:.4f} dp_rms={dp_rms:.5f}', flush=True)

    print(f'[GRPO] Completed. Best checkpoint -> {args.output / "grpo_best.pt"}', flush=True)


if __name__ == '__main__':
    main()
