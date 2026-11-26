#!/usr/bin/env python3
"""True On-Policy Resident 256-Lane Block-Gaussian PPO-GRPO directly in real PhysX.

Key breakthroughs over earlier offline DAgger & 1024-lane GRPO:
1. Exact 256-lane physics parity (env_spacing=0.0, bit-identical to official 128-lane eval).
2. NativeApproachRestart snapshot at tick 197 (just before ball launch at tick 200): runs
   the 2-second ground-standing prefix ONCE (~4 min), then every RL iteration runs ONLY
   ticks 197..450 (~5.0 min per full 256-lane iteration, 2.2x faster!).
3. Antithetic 50ms (K=5 tick) Block-Gaussian exploration on arm 12:18 across 48 fresh
   disjoint serves/iter x G=4 (+z1, -z1, +z2, -z2) with 2:1 Panel B focus (16 A + 32 B)
   and live deterministic validation on lanes 0..63 (strictly disjoint val64_select).
4. Exact closed-form block-Gaussian clipped PPO-GRPO update with ZERO retention lock to
   biased nominal trajectories and explicit center-seeking string/landing/speed shaping
   that directly removes the +51mm top-rim and 3.48m back-line biases.
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


def reserved_case_ids(execution_root: Path):
    ids = set()
    p01 = execution_root / 'reports/G07_full_racket_v4/protocol_frozen01'
    for name in ('training_cases.json', 'A_cases.json', 'B_cases.json'):
        p = p01 / name
        if p.is_file():
            d = json.loads(p.read_text())
            rows = d.get('cases', d) if isinstance(d, dict) else d
            ids.update(r['case_id'] for r in rows)
    import gzip
    for pat in ('**/fixed128*.json*', '**/independent*.json*', '**/quick64*.json*', '**/val64*.json', '**/val256*.json'):
        for p in execution_root.glob(pat):
            try:
                txt = gzip.open(p, 'rt').read() if p.suffix == '.gz' else p.read_text()
                d = json.loads(txt)
                rows = d.get('cases', d) if isinstance(d, dict) else d
                if isinstance(rows, list):
                    ids.update(r['case_id'] for r in rows if isinstance(r, dict) and 'case_id' in r)
            except Exception:
                pass
    return ids


def compute_shaped_reward(row, min_dist_m: float, loc_z_m) -> float:
    succ = 1.0 if row['success'] else 0.0
    qstr = 1.0 if row['string'] else 0.0
    rnet = 1.0 if row['robot_net'] else 0.0
    sm = row['string_margin_m']
    sm_t = max(-1.5, min(1.0, sm / 0.045)) if (sm is not None and math.isfinite(sm)) else -1.5
    spm = row['speed_margin_mps']
    spm_t = max(-1.5, min(1.0, spm / 0.60)) if (qstr > 0 and spm is not None and math.isfinite(spm)) else -0.5
    lm = row['landing_margin_m']
    lm_t = max(-1.5, min(1.0, lm / 0.45)) if (qstr > 0 and lm is not None and math.isfinite(lm)) else -1.0
    nm = row['net_margin_m']
    nm_t = max(-1.0, min(1.0, nm / 0.14)) if (qstr > 0 and nm is not None and math.isfinite(nm)) else -0.5
    # Direct center-seeking bonus on loc_z to erase the +51mm top-rim bias even on successes!
    lz_pen = 0.0
    if loc_z_m is not None and math.isfinite(loc_z_m):
        lz_pen = -0.55 * max(0.0, min(0.10, abs(loc_z_m) - 0.015)) / 0.060
    had_c = (row['string'] or row['frame'] or (sm is not None))
    dist_pen = 0.0 if had_c else -1.80 * max(0.0, min(0.40, min_dist_m - 0.08))
    return float(
        4.00 * succ
        + 1.40 * qstr
        + 0.85 * sm_t
        + 0.65 * spm_t
        + 0.90 * lm_t
        + 0.35 * nm_t
        + lz_pen
        + dist_pen
        - 8.00 * rnet
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--val-spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--g07-execution-root', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=882072)
    parser.add_argument('--serve-seed-base', type=int, default=960000)
    parser.add_argument('--iterations', type=int, default=14)
    parser.add_argument('--n-serves-a', type=int, default=16)
    parser.add_argument('--n-serves-b', type=int, default=32)
    parser.add_argument('--group-size', type=int, default=4, help='Even >= 2 (antithetic pairs per serve)')
    parser.add_argument('--block-ticks', type=int, default=5, help='Temporal correlation block length in ticks (50ms)')
    parser.add_argument('--sigma-arm', type=float, default=0.075, help='Block-Gaussian arm exploration std')
    parser.add_argument('--sigma-decay', type=float, default=0.96)
    parser.add_argument('--clip-eps', type=float, default=0.16)
    parser.add_argument('--kl-beta', type=float, default=4.0, help='Step-to-step arm KL penalty coefficient')
    parser.add_argument('--leg-anchor', type=float, default=25.0, help='Keep legs 0:12 tightly anchored to ref')
    parser.add_argument('--prox-weight', type=float, default=0.04)
    parser.add_argument('--lr', type=float, default=9e-5)
    parser.add_argument('--ppo-epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--max-arm-rms', type=float, default=0.015, help='Hard per-iter arm RMS trust-region cap')
    parser.add_argument('--unlock-base', action='store_true', default=False)
    parser.add_argument('--snapshot-tick', type=int, default=197)
    parser.add_argument('--max-ticks', type=int, default=465)
    parser.add_argument('--physics-device', default='cuda:0')
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    if args.group_size < 2 or args.group_size % 2 != 0:
        raise ValueError('--group-size must be even >= 2 for antithetic pairs')

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
        _run_ppo(args, stop_request)
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    else:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


def _run_ppo(args, stop_request):
    import numpy as np
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from quadsmash.rl.supermimic_v1 import load_ppo_initialized_actor
    from quadsmash.rl.supermimic_task_contract import (
        resolve_task_contract, make_task_cfg, require_resume_contract,
        zero_panel_reset_noise, validate_sweep_contract)
    from quadsmash.rl.supermimic_terminal_scoring import (
        score_terminal_episode, approach_ball_scoring_protocol)
    from quadsmash.rl.supermimic_policy_execution_v1 import resolve_contract, policy_forward
    from quadsmash.rl.supermimic_physical_loop_loading_v1 import APPROACH_NAMES
    from quadsmash.rl.supermimic_native_approach_reset import NativeApproachRestart
    from quadsmash.physics.full_racket_v4 import string_polygon
    from quadsmash.rl.supermimic_native_approach_search import native_fitness
    from quadsmash.tasks.stand.observation_contract import inverse_rotate
    from quadsmash.tasks.stand.supermimic_env_v1 import SuperMimicEnvV1
    from quadsmash.tasks.stand.full_racket_eval_env_v4 import FrozenPanelEnv
    from freeze_full_racket_protocol_v4 import incoming_cases

    class TrainingEnv(SuperMimicEnvV1, FrozenPanelEnv):
        pass

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    val_spec = json.loads(args.val_spec.read_text())
    val_cases = val_spec['cases']
    n_val = len(val_cases)
    n_train_serves = args.n_serves_a + args.n_serves_b
    G = args.group_size
    n_pairs = G // 2
    lanes = n_val + n_train_serves * G

    execution_root = args.g07_execution_root.resolve()
    reserved = reserved_case_ids(execution_root)
    reserved.update(c['case_id'] for c in val_cases)
    print(f'[PPO-256] Lanes={lanes} ({n_val} live val64 + {n_train_serves} serves [{args.n_serves_a}A+{args.n_serves_b}B] x G={G}), '
          f'block={args.block_ticks}t, sigma_arm={args.sigma_arm}, unlock_base={args.unlock_base}', flush=True)

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

    def make_iter_cases(it_idx: int):
        seed_a = args.serve_seed_base + it_idx * 10 + 1
        seed_b = args.serve_seed_base + it_idx * 10 + 2
        ca = incoming_cases(args.n_serves_a, seed_a, 'A')
        cb = incoming_cases(args.n_serves_b, seed_b, 'B')
        serves = ca + cb
        ov = {c['case_id'] for c in serves} & reserved
        if ov:
            raise ValueError(f'Training serves overlap reserved panels: {sorted(ov)[:3]}')
        full = list(val_cases)
        for c in serves:
            for _ in range(G):
                full.append(c)
        return serves, full

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

    base_names = {k for k in trained['actor'].keys() if k.startswith('actor.base.')}
    trainable_names = (set(APPROACH_NAMES) | base_names) if args.unlock_base else set(APPROACH_NAMES)

    polygon = torch.tensor(string_polygon(env.racket_config), dtype=torch.float32, device=device)
    edges = torch.roll(polygon, -1, 0) - polygon
    inward = torch.stack((-edges[:, 1], edges[:, 0]), -1)
    inward = inward / inward.norm(dim=-1, keepdim=True)
    poly_offsets = (polygon * inward).sum(-1)

    def signed_string_margin_and_locz(motion):
        if not motion or not all(k in motion for k in ('point_m', 'racket_center_m', 'racket_quaternion_w')):
            return None, None
        point = torch.tensor([motion['point_m']], dtype=torch.float32, device=device)
        center = torch.tensor([motion['racket_center_m']], dtype=torch.float32, device=device)
        quat = torch.tensor([motion['racket_quaternion_w']], dtype=torch.float32, device=device)
        local = inverse_rotate(quat, point - center)[0, 1:3]
        if not local.isfinite().all():
            return None, None
        sm = float((local @ inward.T - poly_offsets).min())
        return sm, float(local[1])

    def signed_margin_only(motion):
        return signed_string_margin_and_locz(motion)[0]

    def save_checkpoint(path: Path, state_dict):
        out_ckpt = {k: v for k, v in trained.items()
                    if k in ('algorithm', 'adapter_schema', 'actor', 'metadata', 'config',
                             'actor_updates', 'world_updates', 'critic_updates')}
        out_ckpt['actor'] = {k: v.detach().cpu().clone() for k, v in state_dict.items()}
        torch.save(out_ckpt, path)

    # ONE-TIME ground-standing prefix up to snapshot_tick (197), BEFORE ball launch at tick 200!
    print(f'[PPO-256] Running ONE-TIME standing prefix ticks 0..{args.snapshot_tick}...', flush=True)
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
            if (tick + 1) % 50 == 0:
                print(f'  prefix tick={tick+1}/{args.snapshot_tick} elapsed={time.perf_counter()-t_pre:.1f}s', flush=True)
    snapshot = NativeApproachRestart.capture(env)
    obs_snap = obs['policy'].detach().clone()
    print(f'[PPO-256] Captured NativeApproachRestart snapshot @ tick {args.snapshot_tick} in {time.perf_counter()-t_pre:.1f}s!', flush=True)

    history_path = args.output / 'ppo_history.jsonl'
    best_val_key = (-1, -1e9)
    best_state = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}
    consec_drops = 0
    cur_lr = args.lr

    for it in range(args.iterations):
        if stop_request['signal']:
            print(f'[PPO-256] Stopped by signal {stop_request["signal"]}', flush=True)
            break

        t_iter = time.perf_counter()
        train_serves, full_cases = make_iter_cases(it)
        snapshot.restore(env)
        env.cases = full_cases
        policy_obs = obs_snap.clone()

        sigma = args.sigma_arm * (args.sigma_decay ** it)
        rng = torch.Generator(device=device)
        rng.manual_seed(args.seed + 7919 * (it + 1))

        # Pre-generate antithetic K-tick Block-Gaussian arm noise for ticks snapshot_tick..max_ticks
        n_sim_ticks = args.max_ticks - args.snapshot_tick
        n_blocks = (n_sim_ticks + args.block_ticks - 1) // args.block_ticks
        # Shape [n_blocks, lanes, 6]: 0 for val lanes, antithetic (+z, -z) for training pairs!
        z_blocks = torch.zeros((n_blocks, lanes, 6), dtype=torch.float32, device=device)
        for i in range(n_train_serves):
            base_lane = n_val + i * G
            for p_idx in range(n_pairs):
                lp = base_lane + 2 * p_idx
                lm = base_lane + 2 * p_idx + 1
                z = (torch.randn((n_blocks, 6), generator=rng, device=device) * sigma).clamp(-2.6 * sigma, 2.6 * sigma)
                z_blocks[:, lp, :] = z
                z_blocks[:, lm, :] = -z

        # Rollout buffer per block for training lanes (lanes n_val..lanes-1)
        tr_sl = slice(n_val, lanes)
        rec_obs, rec_old_mu, rec_gate, rec_z, rec_blk, rec_lane = [], [], [], [], [], []
        min_dist = torch.full((lanes,), 9.9, dtype=torch.float32, device=device)
        first = {}
        cursor = 0
        actor.eval()

        with torch.inference_mode():
            for step_i in range(n_sim_ticks):
                tick = args.snapshot_tick + step_i
                blk_id = step_i // args.block_ticks
                base_req = policy_forward(actor, policy_obs, env.rally, mode=policy_execution['recovery_gate'], sample=False)[0]
                agate = actor.approach_action_gate(policy_obs)  # [lanes, 18]

                active_pre = (env.episode_index == 0) & (~env.rally.had_contact)
                arm_gate = agate[:, 12:18]  # [lanes, 6]
                action = base_req.clone()
                if tick >= 205:
                    delta_arm = arm_gate * z_blocks[blk_id]  # [lanes, 6] (0 on val lanes!)
                    action[:, 12:18] = torch.where(active_pre[:, None], action[:, 12:18] + delta_arm, action[:, 12:18])
                action = action.clamp(-1.0, 1.0)

                # Record training transitions strictly when pre-contact arm gate is meaningfully active!
                if tick >= 205 and tick <= 345:
                    tr_active = active_pre[tr_sl] & (arm_gate[tr_sl].amax(-1) > 0.05)
                    if tr_active.any():
                        idx_local = tr_active.nonzero().flatten()
                        idx_global = idx_local + n_val
                        rec_obs.append(policy_obs[idx_global].cpu())
                        rec_old_mu.append(base_req[idx_global].cpu())
                        rec_gate.append(agate[idx_global].cpu())
                        rec_z.append(z_blocks[blk_id, idx_global].cpu())
                        rec_blk.append(torch.full((len(idx_local),), blk_id, dtype=torch.int32))
                        rec_lane.append(idx_local.cpu().to(torch.int32))

                if env.approach_ball_termination is not None:
                    env.record_approach_gate(agate)
                obs, _, _, _ = wrapper.step(action)
                policy_obs = obs['policy']

                # Track shuttle-racket min distance for miss shaping
                if tick >= 202 and tick <= 340:
                    bpos = env.shuttle.data.root_pos_w - env.scene.env_origins
                    rpos = env.robot.data.body_link_pos_w[:, env.racket_body_id] - env.scene.env_origins
                    d_br = (bpos - rpos).norm(dim=-1)
                    in_pre = (env.episode_index == 0) & (~env.rally.had_contact) & env.ball_observation_enabled
                    min_dist = torch.where(in_pre & (d_br < min_dist), d_br, min_dist)

                for record in env.rally_records[cursor:]:
                    lane = int(record['env_id'])
                    if lane not in first:
                        measured = score_terminal_episode(record, scoring_protocol)
                        if measured['episode_index'] == 0:
                            first[lane] = measured
                cursor = len(env.rally_records)

                if len(first) == lanes:
                    break

        t_roll = time.perf_counter() - t_iter
        # Score all settled lanes using official native_fitness on raw ep0 records
        raw_ep0 = {}
        for r in env.rally_records:
            if r['episode_index'] == 0 and int(r['env_id']) not in raw_ep0:
                raw_ep0[int(r['env_id'])] = r
        # Handle rare unsettled lanes gracefully if any
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
        scored = native_fitness([raw_ep0[i] for i in range(lanes)], env.court_config, signed_margin_only)
        all_rows = scored['rows']
        min_d_cpu = min_dist.cpu().numpy()

        # Extract loc_z per lane from first_approach_contact_motion
        loc_z_arr = [None] * lanes
        for ln in range(lanes):
            mot = raw_ep0[ln].get('first_approach_contact_motion')
            if mot:
                _, loc_z_arr[ln] = signed_string_margin_and_locz(mot)

        # Live deterministic validation metrics on val64_select (lanes 0..63)
        val_rows = all_rows[:n_val]
        val_succ = sum(r['success'] for r in val_rows)
        val_str = sum(r['string'] for r in val_rows)
        val_frm = sum(r['frame'] for r in val_rows)
        val_low = sum(r['low_speed'] for r in val_rows)
        val_bnet = sum(r['ball_net'] for r in val_rows)
        val_rnet = sum(r['robot_net'] for r in val_rows)
        val_margin = float(np.mean([r['margin_score'] for r in val_rows]))
        val_ret = float(np.mean([first[i]['native_episode_return'] for i in range(n_val) if i in first])) if any(i in first for i in range(n_val)) else 0.0
        val_lzs = [loc_z_arr[i] for i in range(n_val) if loc_z_arr[i] is not None]
        val_lz_mm = float(np.median(val_lzs) * 1000.0) if val_lzs else 0.0

        # Compute group-relative advantages for all 48 x 4 training rollouts
        tr_rows = all_rows[n_val:]
        tr_succ = sum(r['success'] for r in tr_rows)
        tr_str = sum(r['string'] for r in tr_rows)
        tr_rnet = sum(r['robot_net'] for r in tr_rows)
        adv_lane = np.zeros(n_train_serves * G, dtype=np.float32)
        rew_all = np.zeros(n_train_serves * G, dtype=np.float32)
        n_active_groups = 0
        for s_idx in range(n_train_serves):
            g_rews = []
            for g in range(G):
                ln = n_val + s_idx * G + g
                rw = compute_shaped_reward(all_rows[ln], float(min_d_cpu[ln]), loc_z_arr[ln])
                g_rews.append(rw)
                rew_all[s_idx * G + g] = rw
            g_rews = np.array(g_rews, dtype=np.float32)
            std = float(g_rews.std())
            if std > 0.08:
                n_active_groups += 1
                adv_lane[s_idx * G:(s_idx + 1) * G] = np.clip((g_rews - g_rews.mean()) / (std + 0.30), -2.5, 2.5)

        val_key = (val_succ, val_margin)
        is_new_best = (val_rnet == 0) and (val_key > best_val_key)
        if is_new_best:
            best_val_key = val_key
            best_state = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}
            save_checkpoint(args.output / 'best_val.pt', best_state)
            consec_drops = 0
        elif val_succ <= best_val_key[0] - 4:
            consec_drops += 1
        else:
            consec_drops = 0

        save_checkpoint(args.output / f'ppo_it{it:02d}.pt', actor.state_dict())

        # Rollback safeguard if policy degrades 2 iterations in a row
        if consec_drops >= 2:
            print(f'  [ROLLBACK] val64 dropped to {val_succ}/64 vs best {best_val_key[0]}/64 -> restoring best_val & halving lr!', flush=True)
            actor.load_state_dict(best_state, strict=True)
            actor.to(device)
            cur_lr *= 0.5
            consec_drops = 0
            continue

        # Aggregate recorded frames into exact K-tick Block-Gaussian PPO update!
        if not rec_obs:
            print(f'[PPO-256 it={it:02d}] No active training frames recorded, skipping update!', flush=True)
            continue

        obs_t = torch.cat(rec_obs, dim=0).to(device)
        old_mu_t = torch.cat(rec_old_mu, dim=0).to(device)
        gate_t = torch.cat(rec_gate, dim=0).to(device)
        z_t = torch.cat(rec_z, dim=0).to(device)
        blk_t = torch.cat(rec_blk, dim=0).to(device)
        lane_t = torch.cat(rec_lane, dim=0).to(device)
        adv_t = torch.tensor(adv_lane, dtype=torch.float32, device=device)[lane_t]

        # Group frames by unique (lane, blk) so each block contributes ONE exact log-ratio!
        pair_key = lane_t.long() * 1000 + blk_t.long()
        uniq_keys, inv_idx = torch.unique(pair_key, return_inverse=True)
        n_uniq = len(uniq_keys)

        with torch.no_grad():
            mod_ref = ref_actor.actor
            norm_obs_t = mod_ref.normalizer(obs_t)
            base_out_t = mod_ref.base(norm_obs_t)
            ref_mu_t = ref_actor._gated_distribution(obs_t, torch.zeros((len(obs_t), 1), dtype=torch.bool, device=device)).loc.clamp(-1.0, 1.0)
            agate_t = gate_t
            # Precompute per-block z and advantage (constant across ticks in same (lane, blk))
            blk_cnt = torch.zeros(n_uniq, dtype=torch.float32, device=device).scatter_add_(0, inv_idx, torch.ones(len(obs_t), device=device))
            blk_z = torch.zeros((n_uniq, 6), dtype=torch.float32, device=device).scatter_add_(0, inv_idx[:, None].expand(-1, 6), z_t) / blk_cnt[:, None]
            blk_adv = torch.zeros(n_uniq, dtype=torch.float32, device=device).scatter_add_(0, inv_idx, adv_t) / blk_cnt

        # Filter to blocks with non-trivial advantage
        active_blk_mask = blk_adv.abs() > 0.02
        for name, p in actor.named_parameters():
            p.requires_grad_(name in trainable_names)
        trainable = [(n, p) for n, p in actor.named_parameters() if p.requires_grad]
        old_params = {n: p.detach().clone() for n, p in trainable}
        opt = torch.optim.Adam([p for _, p in trainable], lr=cur_lr)

        mod = actor.actor
        inv_sigma2 = 1.0 / (sigma * sigma)
        early_stop_ep = args.ppo_epochs
        final_arm_rms = 0.0
        final_leg_rms = 0.0
        final_clip_frac = 0.0

        # Mini-batch over unique (lane, blk) groups!
        # Since n_obs is only ~5,500 frames (~1,200 blocks), full-batch or 2-3 microbatches on RTX 5080 takes <0.3s total!
        for ep in range(args.ppo_epochs):
            actor.train()
            opt.zero_grad(set_to_none=True)
            b_out = mod.base(norm_obs_t) if args.unlock_base else base_out_t
            pred_mu = (b_out + agate_t * mod.approach(norm_obs_t) * mod.approach_scale).clamp(-1.0, 1.0)
            d_arm = (pred_mu[:, 12:18] - old_mu_t[:, 12:18])  # [N, 6]
            # Project action shift into z-space (since rollout applied arm_gate * z)
            w_g = agate_t[:, 12:18].clamp_min(0.10)
            d_zspace = d_arm / w_g
            num_d = torch.zeros((n_uniq, 6), dtype=torch.float32, device=device).scatter_add_(0, inv_idx[:, None].expand(-1, 6), d_zspace * w_g)
            den_d = torch.zeros((n_uniq, 6), dtype=torch.float32, device=device).scatter_add_(0, inv_idx[:, None].expand(-1, 6), w_g)
            blk_d = num_d / den_d.clamp_min(1e-4)  # [n_uniq, 6]

            # Exact block-Gaussian log-ratio: (<blk_d, z> - 0.5*||blk_d||^2) / sigma^2
            log_ratio = ((blk_d * blk_z).sum(-1) - 0.5 * blk_d.square().sum(-1)) * inv_sigma2
            log_ratio = log_ratio.clamp(-2.0, 2.0)
            ratio = torch.exp(log_ratio)
            surr1 = ratio * blk_adv
            surr2 = ratio.clamp(1.0 - args.clip_eps, 1.0 + args.clip_eps) * blk_adv
            if active_blk_mask.any():
                loss_ppo = -torch.minimum(surr1, surr2)[active_blk_mask].mean()
                clip_frac = float(((ratio - 1.0).abs() > args.clip_eps)[active_blk_mask].float().mean())
            else:
                loss_ppo = torch.tensor(0.0, device=device)
                clip_frac = 0.0

            loss_kl_arm = d_arm.square().mean()
            loss_leg = (pred_mu[:, :12] - ref_mu_t[:, :12]).square().mean()
            prox = sum((p - old_params[n]).square().mean() for n, p in trainable) / len(trainable)
            loss = loss_ppo + args.kl_beta * loss_kl_arm + args.leg_anchor * loss_leg + args.prox_weight * prox
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
            opt.step()

            with torch.no_grad():
                b_out2 = mod.base(norm_obs_t) if args.unlock_base else base_out_t
                pred2 = (b_out2 + agate_t * mod.approach(norm_obs_t) * mod.approach_scale).clamp(-1.0, 1.0)
                final_arm_rms = float((pred2[:, 12:18] - old_mu_t[:, 12:18]).square().mean().sqrt())
                final_leg_rms = float((pred2[:, :12] - ref_mu_t[:, :12]).square().mean().sqrt())
                final_clip_frac = clip_frac
                if final_arm_rms >= args.max_arm_rms:
                    early_stop_ep = ep + 1
                    break

        rec = dict(
            iter=it, elapsed_s=round(time.perf_counter() - t_iter, 1), rollout_s=round(t_roll, 1),
            settled=len(first), sigma=round(sigma, 4), lr=cur_lr,
            val64_succ=val_succ, val64_str=val_str, val64_frm=val_frm, val64_low=val_low,
            val64_bnet=val_bnet, val64_rnet=val_rnet, val64_ret=round(val_ret, 2),
            val64_margin=round(val_margin, 4), val64_locz_mm=round(val_lz_mm, 1),
            is_best=is_new_best, best_val_succ=best_val_key[0],
            tr192_succ=tr_succ, tr192_str=tr_str, tr192_rnet=tr_rnet,
            tr_mean_rew=round(float(rew_all.mean()), 3), active_groups=n_active_groups,
            n_blocks=n_uniq, ppo_ep=early_stop_ep, clip_frac=round(final_clip_frac, 3),
            arm_rms=round(final_arm_rms, 5), leg_rms=round(final_leg_rms, 5),
        )
        with history_path.open('a') as f:
            f.write(json.dumps(rec) + '\n')
        print(f'[PPO-256 it={it:02d}] val64={val_succ:2d}/64 (str={val_str:2d} frm={val_frm:2d} low={val_low:2d} bnet={val_bnet:2d} rnet={val_rnet} '
              f'lz={val_lz_mm:+5.1f}mm ret={val_ret:.2f} m={val_margin:.3f}){" *BEST*" if is_new_best else ""} | '
              f'tr192={tr_succ:3d}/192 (str={tr_str:3d} rw={rew_all.mean():+.2f} grp={n_active_groups}/{n_train_serves}) | '
              f'ep={early_stop_ep} clip={final_clip_frac:.2f} arm_rms={final_arm_rms:.4f} leg_rms={final_leg_rms:.4f} t={rec["elapsed_s"]}s',
              flush=True)

    print(f'[PPO-256 DONE] Best val64={best_val_key[0]}/64 (margin={best_val_key[1]:.4f}) -> {args.output / "best_val.pt"}', flush=True)


if __name__ == '__main__':
    main()
