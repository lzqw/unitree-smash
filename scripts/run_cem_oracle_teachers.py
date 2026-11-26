#!/usr/bin/env python3
"""256-Lane Real-PhysX Per-Case Multi-Generation CEM Oracle & Robustness-Verified Teacher Harvester.

Implements the exact Step 0.2 (arm-only oracle ceiling measurement) and Step 2 (per-case CEM +
perturbation robustness verification + smooth-manifold teacher selection) directly in exact-parity
256-lane GPU PhysX using NativeApproachRestart @ tick 197 (~4.8 min/generation):

- Runs `N_cases = 16` failed serves in parallel x `K = 16` CEM candidates per serve = 256 lanes.
- 13-D smooth arm parameter space per serve:
    z = [offset_arm(6), ramp_arm(6), dt_ticks(1)]
  where peak_tick = nominal_contact_tick + dt_ticks (dt_ticks in [-2, +5] = -20ms..+50ms!),
  allowing CEM to discover exact closed-loop PhysX sweet-spot hits, inward yaw, AND delayed
  apex-whip strokes automatically with ZERO open-loop IK or surrogate error!
- Generations 0..G-1: Per-serve Cross-Entropy Method (top-4 elites by margin-rich real-PhysX fitness,
  with optional smooth-manifold prior regularization toward f(iy, iz, ft)).
- Final Robustness Generation: Takes the top-4 distinct winning candidates per serve and evaluates
  each across 4 micro-perturbed copies (0, +eps*u1, -eps*u1, +eps*u2; eps=0.018). Keeps ONLY
  robust winners passing >= 3/4 copies with rnet=0, ranked by robustness -> continuous margin ->
  smoothness!
- Records full compact trajectories (identical probe_tape.pt / probe_summary.json / episodes.jsonl
  schema) for the winning robust teacher of every solved serve so distillation plugs straight in!
"""
import argparse
import faulthandler
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


def compute_cem_fitness(row, raw_ep, min_dist_m: float, z_vec, prior_vec=None, prior_reg: float = 0.0) -> float:
    succ = 1.0 if row['success'] else 0.0
    qstr = 1.0 if row['string'] else 0.0
    rnet = 1.0 if row['robot_net'] else 0.0
    sm = row['string_margin_m']
    sm_t = max(-1.5, min(1.0, sm / 0.040)) if (sm is not None and math.isfinite(sm)) else -1.5
    spm = row['speed_margin_mps']
    spm_t = max(-1.5, min(1.2, spm / 0.65)) if (qstr > 0 and spm is not None and math.isfinite(spm)) else -0.5
    lm = row['landing_margin_m']
    lm_t = max(-1.5, min(1.0, lm / 0.45)) if (qstr > 0 and lm is not None and math.isfinite(lm)) else -1.0
    nm = row['net_margin_m']
    nm_t = max(-1.0, min(1.0, nm / 0.14)) if (qstr > 0 and nm is not None and math.isfinite(nm)) else -0.5
    ctr_pen = 0.0
    if qstr > 0:
        land = raw_ep.get('first_landing_m')
        if land and math.isfinite(land[0]) and math.isfinite(land[1]):
            ctr_pen = -0.65 * max(0.0, land[0] - 3.40) - 0.85 * max(0.0, abs(land[1]) - 0.50)
    had_c = (row['string'] or row['frame'] or (sm is not None))
    dist_pen = 0.0 if had_c else -2.5 * max(0.0, min(0.50, min_dist_m - 0.06))
    # Gentle action-effort & smooth-prior regularization (prevents wild multimodal jumps)
    eff_pen = -0.20 * float((z_vec[:12] ** 2).mean())
    if prior_vec is not None and prior_reg > 0:
        eff_pen -= prior_reg * float(((z_vec[:12] - prior_vec[:12]) ** 2).mean())
    return float(
        5.50 * succ
        + 1.60 * qstr
        + 0.95 * sm_t
        + 0.90 * spm_t
        + 1.15 * lm_t
        + 0.60 * nm_t
        + ctr_pen
        + dist_pen
        + eff_pen
        - 12.00 * rnet
    )


def classify_failure(row, raw_ep) -> str:
    if row['success'] and row['string'] and not row['robot_net'] and not row['low_speed']:
        return 'ok'
    if row['robot_net']:
        return 'rnet'
    if row['string']:
        if row['low_speed']:
            return 'str_low'
        if raw_ep.get('shuttle_net_contact'):
            return 'str_bnet'
        if not raw_ep.get('crossed_net'):
            return 'str_short'
        land = raw_ep.get('first_landing_m') or [0.0, 0.0]
        if abs(land[1]) > 1.0:
            return 'str_woob'
        if land[0] > 4.0:
            return 'str_loob'
        return 'str_other'
    if row['frame']:
        return 'frame'
    return 'no_contact'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--cases-spec', type=Path, required=True,
                        help='JSON spec with {"cases": [...]} of target serves (batches of 16)')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--g07-execution-root', type=Path, required=True)
    parser.add_argument('--arm-prep-seconds', type=float, default=None,
                        help='Optional causal arm preparation window override (e.g. 0.25 or 0.38)')
    parser.add_argument('--prior-json', type=Path, default=None,
                        help='Optional smooth ridge/KNN prior mapping (iy,iz,ft,iv) -> 13-D z_init')
    parser.add_argument('--prior-reg', type=float, default=0.35)
    parser.add_argument('--batch-offset', type=int, default=0)
    parser.add_argument('--num-batches', type=int, default=1, help='Number of 16-serve batches to solve sequentially')
    parser.add_argument('--cem-gens', type=int, default=4)
    parser.add_argument('--init-sigma-pos', type=float, default=0.22)
    parser.add_argument('--init-sigma-ramp', type=float, default=0.17)
    parser.add_argument('--init-sigma-dt', type=float, default=1.8)
    parser.add_argument('--robust-eps', type=float, default=0.018, help='Action perturbation std for robustness audit')
    parser.add_argument('--min-save-rob', type=int, default=3, help='Min robustness count out of 4 to save teacher')
    parser.add_argument('--seed', type=int, default=882072)
    parser.add_argument('--snapshot-tick', type=int, default=197)
    parser.add_argument('--search-max-ticks', type=int, default=436, help='Max ticks during CEM search gens (100% wins finish <=432)')
    parser.add_argument('--max-ticks', type=int, default=456, help='Max ticks during final robustness + tape gen')
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
    from quadsmash.physics.full_racket_v4 import string_polygon
    from quadsmash.rl.supermimic_v1 import load_ppo_initialized_actor
    from quadsmash.rl.supermimic_task_contract import (
        resolve_task_contract, make_task_cfg, require_resume_contract,
        zero_panel_reset_noise, validate_sweep_contract)
    from quadsmash.rl.supermimic_terminal_scoring import score_terminal_episode, approach_ball_scoring_protocol
    from quadsmash.rl.supermimic_policy_execution_v1 import resolve_contract, policy_forward
    from quadsmash.rl.supermimic_native_approach_reset import NativeApproachRestart
    from quadsmash.rl.supermimic_native_approach_search import native_fitness
    from quadsmash.rl.supermimic_physical_dataset_v1 import encode_physical, STATE_FIELDS
    from quadsmash.rl.supermimic_physical_capture_v1 import capture_native_inputs
    from quadsmash.tasks.stand.observation_contract import inverse_rotate
    from quadsmash.tasks.stand.supermimic_env_v1 import SuperMimicEnvV1
    from quadsmash.tasks.stand.full_racket_eval_env_v4 import FrozenPanelEnv

    class CemEnv(SuperMimicEnvV1, FrozenPanelEnv):
        pass

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    all_cases = json.loads(args.cases_spec.read_text())['cases']
    B_SIZE = 16
    K = 16
    lanes = B_SIZE * K
    assert lanes == 256

    start_idx = args.batch_offset * B_SIZE
    end_idx = min(len(all_cases), start_idx + args.num_batches * B_SIZE)
    target_cases = all_cases[start_idx:end_idx]
    if not target_cases:
        raise ValueError(f'No cases in slice [{start_idx}:{end_idx}]')

    # Optional smooth linear/RBF prior predictor (self-updates online with new robust winners!)
    prior_X_list, prior_Z_list = [], []
    if args.prior_json and args.prior_json.exists():
        pr = json.loads(args.prior_json.read_text())
        prior_X_list = [list(x) for x in pr['X']]
        prior_Z_list = [list(z) for z in pr['Z']]
        print(f'[CEM-256] Loaded smooth KNN prior from {args.prior_json.name} ({len(prior_X_list)} anchors)', flush=True)

    def predict_prior(c):
        if not prior_X_list:
            return np.zeros(13, dtype=np.float64)
        X_ref = np.array(prior_X_list, dtype=np.float64)
        Z_ref = np.array(prior_Z_list, dtype=np.float64)
        mu_x = X_ref.mean(0); sd_x = X_ref.std(0) + 1e-6
        Xn = (X_ref - mu_x) / sd_x
        ip = c['prediction_only']['intercept_position_m']
        iv = c['prediction_only']['incoming_velocity_mps']
        ft = float(c['prediction_only']['flight_time_s'])
        x = np.array([ip[1], ip[2], ft, iv[0], iv[1], iv[2]], dtype=np.float64)
        xn = (x - mu_x) / sd_x
        d2 = ((Xn - xn[None]) ** 2).sum(axis=1)
        w = np.exp(-0.5 * d2 / 1.2)
        if w.sum() < 1e-6:
            return np.zeros(13, dtype=np.float64)
        return (w[:, None] * Z_ref).sum(0) / w.sum()

    execution_root = args.g07_execution_root.resolve()
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

    # Pad first batch to 16 cases if needed
    first_batch = list(target_cases[:B_SIZE])
    while len(first_batch) < B_SIZE:
        first_batch.append(first_batch[-1])
    full_cases_0 = [c for c in first_batch for _ in range(K)]

    cfg = make_task_cfg(lanes, contract, protocol, receipt, execution_root=execution_root)
    cfg.seed, cfg.sim.device = args.seed, args.physics_device
    validate_sweep_contract(cfg, contract, None)
    noise = zero_panel_reset_noise(lanes, args.seed)
    env = CemEnv(cfg, cases=full_cases_0, reset_noise=noise,
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
    arm_prep = args.arm_prep_seconds if args.arm_prep_seconds is not None else policy_execution.get('arm_preparation_seconds', 0.25)
    actor.configure_approach_gate(arm_prep)
    actor.eval()
    print(f'[CEM-256] Checkpoint={args.checkpoint.name} | arm_prep={arm_prep:.2f}s | cases={len(target_cases)} ({args.num_batches} batches x {args.cem_gens} gens + robustness)', flush=True)

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

    # ONE-TIME standing prefix 0..snapshot_tick (197)
    t_pre = time.perf_counter()
    obs, _ = wrapper.reset()
    with torch.inference_mode():
        for tick in range(args.snapshot_tick):
            p_obs = obs['policy']
            act = policy_forward(actor, p_obs, env.rally, mode=policy_execution['recovery_gate'], sample=False)[0].clamp(-1.0, 1.0)
            if env.approach_ball_termination is not None:
                env.record_approach_gate(actor.approach_action_gate(p_obs))
            obs, _, _, _ = wrapper.step(act)
    snapshot = NativeApproachRestart.capture(env)
    obs_snap = obs['policy'].detach().clone()
    print(f'[CEM-256] Snapshot @ tick {args.snapshot_tick} captured in {time.perf_counter()-t_pre:.1f}s!', flush=True)

    results_path = args.output / 'cem_results.jsonl'

    def eval_256(batch_cases, z_mat, record_lanes=None, max_ticks=None):
        """Evaluate z_mat [256, 13] = [off6, rmp6, dt_ticks] on batch_cases (16 x 16 = 256 lanes)."""
        n_sim_ticks = (max_ticks if max_ticks is not None else args.max_ticks) - args.snapshot_tick
        full = [c for c in batch_cases for _ in range(K)]
        snapshot.restore(env)
        env.cases = full
        policy_obs = obs_snap.clone()

        z_t = torch.tensor(z_mat, dtype=torch.float32, device=device)
        z_pos = torch.zeros((lanes, 18), dtype=torch.float32, device=device)
        z_rmp = torch.zeros((lanes, 18), dtype=torch.float32, device=device)
        z_pos[:, 12:18] = z_t[:, 0:6]
        z_rmp[:, 12:18] = z_t[:, 6:12]
        base_peaks = torch.tensor([round((2.0 + float(c['prediction_only']['flight_time_s'])) * 100.0) for c in full],
                                  dtype=torch.float32, device=device)
        peak_ticks = base_peaks + z_t[:, 12]
        width_ticks = 62.0 if arm_prep >= 0.32 else 54.0

        min_dist = torch.full((lanes,), 9.9, dtype=torch.float32, device=device)
        first = {}
        cursor = 0
        rec_frames = [] if record_lanes is not None else None
        rec_idx = torch.tensor(record_lanes, dtype=torch.long, device=device) if record_lanes is not None else None

        with torch.inference_mode():
            for step_i in range(n_sim_ticks):
                if stop_request['signal']:
                    break
                tick = args.snapshot_tick + step_i
                base_req = policy_forward(actor, policy_obs, env.rally, mode=policy_execution['recovery_gate'], sample=False)[0]
                agate = actor.approach_action_gate(policy_obs)
                t_f = float(tick)
                rise = ((t_f - (peak_ticks - width_ticks)) / width_ticks).clamp(0.0, 1.0)
                fall = ((t_f - peak_ticks) / 16.0).clamp(0.0, 1.0)
                env_p = torch.where(t_f <= peak_ticks, torch.sin(0.5 * math.pi * rise).square(), torch.cos(0.5 * math.pi * fall).square())
                env_r = ((t_f - (peak_ticks - 18.0)) / 18.0).clamp(0.0, 1.0) * (t_f <= peak_ticks + 4.0).float()

                pre = ((env.episode_index == 0) & (~env.rally.had_contact)).float()
                delta = (env_p[:, None] * z_pos + env_r[:, None] * z_rmp) * pre[:, None]
                act = (base_req + agate * delta * actor.actor.approach_scale).clamp(-1.0, 1.0)

                if env.approach_ball_termination is not None:
                    env.record_approach_gate(agate)

                if rec_frames is not None and tick >= 200:
                    physical = capture_native_inputs(env)
                    rec_frames.append(dict(
                        state=encode_physical({k: getattr(physical, k)[rec_idx] for k, _ in STATE_FIELDS}).half().cpu(),
                        observation=policy_obs[rec_idx].half().cpu(),
                        request=act[rec_idx].float().cpu(),
                        gate=agate[rec_idx].half().cpu(),
                        elapsed=physical.elapsed_s[rec_idx].float().cpu(),
                        present=physical.ball_present[rec_idx].bool().cpu(),
                        episode=env.episode_index[rec_idx].to(torch.int16).cpu(),
                        had_contact=env.rally.had_contact[rec_idx].bool().cpu(),
                        released=env.rally.released[rec_idx].bool().cpu(),
                    ))

                in_flight = (env.episode_index == 0) & env.rally.in_play & (~env.rally.had_contact)
                if in_flight.any():
                    d_br = (env.shuttle.data.root_pos_w - env.robot.data.body_link_pos_w[:, env.racket_body_id]).norm(dim=-1)
                    min_dist = torch.where(in_flight, torch.minimum(min_dist, d_br), min_dist)

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
        rows = native_fitness(ordered_raw, env.court_config, signed_string_margin)['rows']
        return rows, ordered_raw, min_dist.cpu().numpy(), rec_frames

    # Process batches of 16 serves sequentially
    n_total_cases = 0
    n_any_solved = 0
    n_robust_3of4 = 0
    n_robust_4of4 = 0
    saved_teachers = []
    rng = np.random.default_rng(args.seed + 31337)

    n_b_actual = math.ceil(len(target_cases) / B_SIZE)
    for b_idx in range(n_b_actual):
        if stop_request['signal']:
            break
        raw_batch = target_cases[b_idx * B_SIZE:(b_idx + 1) * B_SIZE]
        n_real = len(raw_batch)
        batch_cases = list(raw_batch)
        while len(batch_cases) < B_SIZE:
            batch_cases.append(batch_cases[-1])

        # Initialize CEM distribution per case: mu [16, 13], std [16, 13]
        priors = np.array([predict_prior(c) for c in batch_cases], dtype=np.float64)
        mu = priors.copy()
        std = np.zeros((B_SIZE, 13), dtype=np.float64)
        std[:, 0:6] = args.init_sigma_pos
        std[:, 6:12] = args.init_sigma_ramp
        std[:, 12] = args.init_sigma_dt

        # Track top distinct candidates per serve across all CEM generations
        archive = [[] for _ in range(B_SIZE)]  # list of (fitness, is_succ, z13, row, raw_ep)

        for g in range(args.cem_gens):
            t_g = time.perf_counter()
            z_mat = np.zeros((lanes, 13), dtype=np.float64)
            for ci in range(B_SIZE):
                if g == 0 and prior_X_list:
                    z_mat[ci * K + 0] = np.zeros(13)
                    z_mat[ci * K + 1] = priors[ci]
                    z_mat[ci * K + 2] = 0.5 * priors[ci]
                    z_mat[ci * K + 3] = np.clip(1.25 * priors[ci], -0.68, 0.68)
                    z_mat[ci * K + 3, 6:12] = np.clip(z_mat[ci * K + 3, 6:12], -0.45, 0.45)
                    z_mat[ci * K + 3, 12] = np.clip(z_mat[ci * K + 3, 12], -2.0, 5.0)
                    n_pairs, off_idx = 6, 4
                else:
                    z_mat[ci * K + 0] = np.zeros(13) if g == 0 else mu[ci]
                    z_mat[ci * K + 1] = priors[ci] if g == 0 else (archive[ci][0][2] if archive[ci] else mu[ci])
                    n_pairs, off_idx = 7, 2
                for p in range(n_pairs):
                    eps = rng.standard_normal(13) * std[ci]
                    zp = mu[ci] + eps
                    zm = mu[ci] - eps
                    for vec in (zp, zm):
                        vec[0:6] = np.clip(vec[0:6], -0.68, 0.68)
                        vec[6:12] = np.clip(vec[6:12], -0.45, 0.45)
                        vec[12] = np.clip(vec[12], -2.0, 5.0)
                    z_mat[ci * K + off_idx + 2 * p] = zp
                    z_mat[ci * K + off_idx + 2 * p + 1] = zm

            rows, raw_eps, min_d, _ = eval_256(batch_cases, z_mat, max_ticks=args.search_max_ticks)
            gen_solved = 0
            gen_str = 0
            gen_rnet = 0
            for ci in range(B_SIZE):
                scored_c = []
                any_ok_ci = False
                any_str_ci = False
                for k in range(K):
                    ln = ci * K + k
                    r = rows[ln]; raw = raw_eps[ln]
                    fit = compute_cem_fitness(r, raw, float(min_d[ln]), z_mat[ln], priors[ci], args.prior_reg)
                    ok = bool(r['success'] and r['string'] and not r['robot_net'] and not r['low_speed'])
                    if ok: any_ok_ci = True
                    if r['string'] and not r['robot_net']: any_str_ci = True
                    if r['robot_net']: gen_rnet += 1
                    scored_c.append((fit, ok, z_mat[ln].copy(), r, raw))
                    archive[ci].append((fit, ok, z_mat[ln].copy(), r, raw))
                if ci < n_real:
                    if any_ok_ci: gen_solved += 1
                    if any_str_ci: gen_str += 1
                # Keep archive sorted (successes first, then fitness)
                archive[ci].sort(key=lambda x: (1 if x[1] else 0, x[0]), reverse=True)
                # CEM update using top-4 elites of current generation + best-2 from archive
                scored_c.sort(key=lambda x: x[0], reverse=True)
                elites = np.array([x[2] for x in scored_c[:4]] + [x[2] for x in archive[ci][:2]], dtype=np.float64)
                mu[ci] = 0.65 * elites.mean(axis=0) + 0.35 * mu[ci]
                emp_std = elites.std(axis=0)
                std[ci, 0:6] = np.clip(0.55 * emp_std[0:6] + 0.45 * (std[ci, 0:6] * 0.68), 0.030, 0.22)
                std[ci, 6:12] = np.clip(0.55 * emp_std[6:12] + 0.45 * (std[ci, 6:12] * 0.68), 0.025, 0.17)
                std[ci, 12] = np.clip(0.55 * emp_std[12] + 0.45 * (std[ci, 12] * 0.68), 0.35, 1.8)

            cum_solved = sum(1 for ci in range(n_real) if archive[ci] and archive[ci][0][1])
            print(f'[CEM B{b_idx:02d} Gen {g}/{args.cem_gens-1}] ({time.perf_counter()-t_g:5.1f}s) '
                  f'gen_solved={gen_solved:2d}/{n_real} gen_str={gen_str:2d}/{n_real} | cum_solved={cum_solved:2d}/{n_real} | rnet={gen_rnet}', flush=True)

        # FINAL ROBUSTNESS AUDIT GENERATION (+ tape capture of unperturbed candidate 0..3!)
        # For each of the 16 serves, pick up to 4 top distinct candidates and test each on 4 micro-perturbations (0, +e1, -e1, +e2)!
        t_rob = time.perf_counter()
        z_rob = np.zeros((lanes, 13), dtype=np.float64)
        cand_z = [[None] * 4 for _ in range(B_SIZE)]
        for ci in range(B_SIZE):
            picked = []
            for item in archive[ci]:
                z_c = item[2]
                if all(np.linalg.norm(z_c[:12] - p[:12]) > 0.05 for p in picked):
                    picked.append(z_c)
                if len(picked) == 4:
                    break
            while len(picked) < 4:
                picked.append(archive[ci][0][2] if archive[ci] else np.zeros(13))
            for m in range(4):
                cand_z[ci][m] = picked[m].copy()
                base_ln = ci * K + m * 4
                e1 = rng.standard_normal(12) * args.robust_eps
                e2 = rng.standard_normal(12) * args.robust_eps
                z_rob[base_ln + 0] = picked[m]
                z_rob[base_ln + 1] = picked[m].copy(); z_rob[base_ln + 1, :12] += e1
                z_rob[base_ln + 2] = picked[m].copy(); z_rob[base_ln + 2, :12] -= e1
                z_rob[base_ln + 3] = picked[m].copy(); z_rob[base_ln + 3, :12] += e2

        # Record unperturbed lanes (m*4 + 0 for m=0..3 across all 16 cases = 64 lanes!)
        rec_lanes = [ci * K + m * 4 for ci in range(B_SIZE) for m in range(4)]
        rows_r, raw_r, min_d_r, rec_frames = eval_256(batch_cases, z_rob, record_lanes=rec_lanes, max_ticks=args.max_ticks)

        # Pick best robust winner per serve
        batch_3of4, batch_4of4, batch_any = 0, 0, 0
        for ci in range(n_real):
            n_total_cases += 1
            cid = batch_cases[ci]['case_id']
            ip = batch_cases[ci]['prediction_only']['intercept_position_m']
            iv = batch_cases[ci]['prediction_only']['incoming_velocity_mps']
            ft = float(batch_cases[ci]['prediction_only']['flight_time_s'])
            nom_ok = bool(archive[ci] and any(x[1] for x in archive[ci]))

            best_rob = None  # (rob_count, fit0, m, z13, row0, raw0)
            for m in range(4):
                bl = ci * K + m * 4
                r0, raw0 = rows_r[bl], raw_r[bl]
                ok0 = bool(r0['success'] and r0['string'] and not r0['robot_net'] and not r0['low_speed'])
                rob_cnt = sum(1 for j in range(4)
                              if rows_r[bl + j]['success'] and rows_r[bl + j]['string']
                              and not rows_r[bl + j]['robot_net'] and not rows_r[bl + j]['low_speed'])
                fit0 = compute_cem_fitness(r0, raw0, float(min_d_r[bl]), cand_z[ci][m], priors[ci], args.prior_reg)
                if ok0 and (best_rob is None or (rob_cnt, fit0) > (best_rob[0], best_rob[1])):
                    best_rob = (rob_cnt, fit0, m, cand_z[ci][m], r0, raw0)

            any_ok = nom_ok or (best_rob is not None)
            if any_ok:
                batch_any += 1
                n_any_solved += 1

            rob_cnt = best_rob[0] if best_rob else 0
            if rob_cnt >= 3:
                batch_3of4 += 1; n_robust_3of4 += 1
            if rob_cnt == 4:
                batch_4of4 += 1; n_robust_4of4 += 1

            if best_rob is not None:
                best_row, best_raw, best_z, best_st = best_rob[4], best_rob[5], best_rob[3], 'ok'
            else:
                top_a = archive[ci][0]
                best_row, best_raw, best_z = top_a[3], top_a[4], top_a[2]
                best_st = classify_failure(best_row, best_raw)
            land = best_raw.get('first_landing_m') or [None, None]
            rec_entry = dict(
                case_id=cid, iy=round(float(ip[1]), 4), iz=round(float(ip[2]), 4), ft=round(ft, 4),
                any_solved=any_ok, rob_cnt=rob_cnt, best_status=best_st,
                z13=best_z.tolist() if best_rob else None,
                sm=best_row['string_margin_m'],
                spm=best_row['speed_margin_mps'],
                lm=best_row['landing_margin_m'],
                nm=best_row['net_margin_m'],
                lx=land[0], ly=land[1],
            )
            with results_path.open('a') as f:
                f.write(json.dumps(rec_entry) + '\n')

            if best_rob is not None and rob_cnt >= args.min_save_rob:
                # Extract tape slice for lane index (ci * 4 + m) in rec_frames!
                local_rec_col = ci * 4 + best_rob[2]
                saved_teachers.append((batch_cases[ci], best_rob[3], best_rob[4], best_rob[5], rob_cnt, local_rec_col, rec_frames))
                # Online smooth prior self-update with high-margin robust wins!
                lm_v = best_rob[4].get('landing_margin_m') or 0.0
                spm_v = best_rob[4].get('speed_margin_mps') or 0.0
                nm_v = best_rob[4].get('net_margin_m') or 0.0
                lx_v = land[0] if (land[0] is not None and math.isfinite(land[0])) else 9.0
                if spm_v >= 0.20 and lm_v >= 0.12 and nm_v >= 0.015 and lx_v <= 3.72 and np.linalg.norm(best_rob[3][:12]) <= 0.80:
                    prior_X_list.append([float(ip[1]), float(ip[2]), ft, float(iv[0]), float(iv[1]), float(iv[2])])
                    prior_Z_list.append(best_rob[3].tolist())

        print(f'[CEM B{b_idx:02d} ROBUST] ({time.perf_counter()-t_rob:5.1f}s) any_solved={batch_any:2d}/{n_real} '
              f'| robust>=3/4={batch_3of4:2d}/{n_real} | robust=4/4={batch_4of4:2d}/{n_real} | total_saved={len(saved_teachers)} | prior_anchors={len(prior_X_list)}', flush=True)

        # Save incrementally after EVERY batch so progress is never lost!
        if prior_X_list:
            (args.output / 'updated_prior.json').write_text(json.dumps(dict(X=prior_X_list, Z=prior_Z_list), indent=2) + '\n')
        if saved_teachers:
            T_min = min(len(x[6]) for x in saved_teachers)
            tape_out = {}
            for key in saved_teachers[0][6][0].keys():
                tape_out[key] = torch.stack([
                    torch.stack([item[6][t][key][item[5]] for item in saved_teachers], dim=0)
                    for t in range(T_min)
                ], dim=0)
            torch.save(dict(tape=tape_out, start_capture_tick=200), args.output / 'probe_tape.pt')
            out_cases, out_offs, out_rmps, out_peaks, out_widths, bmeta, out_rows = [], [], [], [], [], [], []
            width_ticks = 62 if arm_prep >= 0.32 else 54
            with (args.output / 'episodes.jsonl').open('w') as ef:
                for idx, (c, z13, row, raw, rcnt, _, _) in enumerate(saved_teachers):
                    off18 = [0.0] * 12 + z13[0:6].tolist()
                    rmp18 = [0.0] * 12 + z13[6:12].tolist()
                    pt = int(round((2.0 + float(c['prediction_only']['flight_time_s'])) * 100.0 + float(z13[12])))
                    out_cases.append(c); out_offs.append(off18); out_rmps.append(rmp18)
                    out_peaks.append(pt); out_widths.append(width_ticks)
                    bmeta.append(dict(case_id=c['case_id'], tag='cem_robust', rob_cnt=int(rcnt), dt_ticks=round(float(z13[12]), 2)))
                    r_copy = dict(row); r_copy['env_id'] = idx
                    out_rows.append(r_copy)
                    raw_copy = dict(raw); raw_copy['env_id'] = idx
                    ef.write(json.dumps(raw_copy) + '\n')
            (args.output / 'probe_summary.json').write_text(json.dumps(dict(rows=out_rows, branch_meta=bmeta), indent=2) + '\n')
            (args.output / 'cem_teachers.spec.json').write_text(json.dumps(dict(
                mode='cem_robust_teachers', arm_prep_seconds=arm_prep,
                cases=out_cases, offsets=out_offs, ramp_offsets=out_rmps,
                peak_ticks=out_peaks, width_ticks=out_widths, branch_meta=bmeta,
                gate_mode='approach_gate', record_tape=True,
            ), indent=2) + '\n')

    summary = dict(
        checkpoint=str(args.checkpoint), arm_prep_seconds=arm_prep,
        total_cases=n_total_cases, any_solved=n_any_solved,
        any_solved_rate=n_any_solved / max(1, n_total_cases),
        robust_3of4=n_robust_3of4, robust_3of4_rate=n_robust_3of4 / max(1, n_total_cases),
        robust_4of4=n_robust_4of4, robust_4of4_rate=n_robust_4of4 / max(1, n_total_cases),
        saved_teachers=len(saved_teachers),
    )
    (args.output / 'cem_oracle_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('[CEM-256] FINAL ORACLE SUMMARY:', json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
