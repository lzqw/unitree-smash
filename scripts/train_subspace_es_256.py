#!/usr/bin/env python3
"""256-Lane Real-PhysX Subspace Antithetic Evolution Strategies (ES) for actor.approach.

Directly optimizes the exact real-PhysX complete return + centered continuous margin of the
standard 794->128->64->18 actor.approach MLP inside a 36-D smooth policy-basin + spatial-steering
subspace:
  - 100% deterministic closed-loop real-PhysX rollouts (ZERO per-tick action noise!)
  - ZERO distillation error, ZERO surrogate error, ZERO architecture change!
  - Exact NativeApproachRestart @ tick 197 (~5.3 min per 256-lane eval)
  - Common Random Numbers (CRN): exact antithetic pairs (theta + sigma*u, theta - sigma*u) run
    on the IDENTICAL stratified 32-serve panel inside the same 256-lane batch!
  - Hard robot_net == 0 veto + Panel-A 1.35x priority + centered lx/ly margin shaping!
  - Periodic 128-serve paired verification (theta_curr vs theta_best in 128x2=256 lanes) saves
    es_best.pt ONLY on strict verified real-PhysX improvement with robot_net == 0!
"""
import argparse
import copy
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


def score_episode_es(row, raw_ep, is_pa: bool) -> float:
    """Dense, margin-rich, court-centered real-PhysX fitness per serve."""
    succ = 1.0 if (row['success'] and row['string'] and not row['robot_net'] and not row['low_speed']) else 0.0
    qstr = 1.0 if (row['string'] and not row['robot_net']) else 0.0
    rnet = 1.0 if row['robot_net'] else 0.0
    sm = row['string_margin_m']
    sm_t = max(-1.5, min(1.0, sm / 0.040)) if (sm is not None and math.isfinite(sm)) else -1.5
    spm = row['speed_margin_mps']
    spm_t = max(-1.5, min(0.8, spm / 0.65)) if (qstr > 0 and spm is not None and math.isfinite(spm)) else -0.5
    lm = row['landing_margin_m']
    lm_t = max(-1.5, min(1.0, lm / 0.45)) if (qstr > 0 and lm is not None and math.isfinite(lm)) else -1.0
    nm = row['net_margin_m']
    nm_t = max(-1.0, min(1.0, nm / 0.14)) if (qstr > 0 and nm is not None and math.isfinite(nm)) else -0.5
    ctr_pen = 0.0
    if qstr > 0:
        land = raw_ep.get('first_landing_m')
        if land and math.isfinite(land[0]) and math.isfinite(land[1]):
            # Strong backline centering to keep lx safely <= 3.45m and |ly| <= 0.55m!
            ctr_pen = -0.95 * max(0.0, land[0] - 3.45) - 0.85 * max(0.0, abs(land[1]) - 0.55)
    base_f = (
        6.00 * succ
        + 1.60 * qstr
        + 0.85 * sm_t
        + 0.65 * spm_t
        + 1.05 * lm_t
        + 0.55 * nm_t
        + ctr_pen
        - 20.00 * rnet
    )
    return float(base_f * (1.35 if is_pa else 1.0))


def summarize_chunk(rows_c, raw_c):
    n = len(rows_c)
    ok = sum(1 for r in rows_c if r['success'] and r['string'] and not r['robot_net'] and not r['low_speed'])
    okA = sum(1 for r in rows_c if r['case_id'].startswith('A-') and r['success'] and r['string'] and not r['robot_net'] and not r['low_speed'])
    okB = ok - okA
    nA = sum(1 for r in rows_c if r['case_id'].startswith('A-'))
    nB = n - nA
    qstr = sum(1 for r in rows_c if r['string'] and not r['robot_net'])
    frm = sum(1 for r in rows_c if r['frame'] and not r['string'])
    rnet = sum(1 for r in rows_c if r['robot_net'])
    low = sum(1 for r in rows_c if r['string'] and r['low_speed'])
    bnet = sum(1 for r, raw in zip(rows_c, raw_c) if r['string'] and raw.get('shuttle_net_contact'))
    loob = sum(1 for r, raw in zip(rows_c, raw_c) if r['string'] and not r['success'] and (raw.get('first_landing_m') or [0, 0])[0] > 4.0)
    woob = sum(1 for r, raw in zip(rows_c, raw_c) if r['string'] and not r['success'] and abs((raw.get('first_landing_m') or [0, 0])[1]) > 1.0)
    fit = sum(score_episode_es(r, raw, r['case_id'].startswith('A-')) for r, raw in zip(rows_c, raw_c)) / max(1, n)
    return dict(n=n, ok=ok, okA=okA, nA=nA, okB=okB, nB=nB, str=qstr, frm=frm, rnet=rnet,
                low=low, bnet=bnet, loob=loob, woob=woob, fit=fit)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--basis-pt', type=Path, required=True)
    p.add_argument('--cases-spec', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--g07-execution-root', type=Path, required=True)
    p.add_argument('--es-steps', type=int, default=12)
    p.add_argument('--sigma-start', type=float, default=0.22)
    p.add_argument('--sigma-end', type=float, default=0.08)
    p.add_argument('--lr', type=float, default=0.14)
    p.add_argument('--l2-reg', type=float, default=0.015)
    p.add_argument('--verify-every', type=int, default=4)
    p.add_argument('--seed', type=int, default=882072)
    p.add_argument('--snapshot-tick', type=int, default=197)
    p.add_argument('--max-ticks', type=int, default=444)
    p.add_argument('--physics-device', default='cuda:0')
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()

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
    from quadsmash.physics.full_racket_v4 import string_polygon
    from quadsmash.rl.supermimic_v1 import load_ppo_initialized_actor
    from quadsmash.rl.supermimic_task_contract import (
        resolve_task_contract, make_task_cfg, require_resume_contract,
        zero_panel_reset_noise, validate_sweep_contract)
    from quadsmash.rl.supermimic_terminal_scoring import score_terminal_episode, approach_ball_scoring_protocol
    from quadsmash.rl.supermimic_policy_execution_v1 import resolve_contract, policy_forward, recovery_gate
    from quadsmash.rl.supermimic_native_approach_reset import NativeApproachRestart
    from quadsmash.rl.supermimic_native_approach_search import native_fitness
    from quadsmash.rl.supermimic_physical_loop_loading_v1 import APPROACH_NAMES
    from quadsmash.tasks.stand.observation_contract import inverse_rotate
    from quadsmash.tasks.stand.supermimic_env_v1 import SuperMimicEnvV1
    from quadsmash.tasks.stand.full_racket_eval_env_v4 import FrozenPanelEnv

    class EsEnv(SuperMimicEnvV1, FrozenPanelEnv):
        pass

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    basis_blob = torch.load(args.basis_pt, map_location='cpu', weights_only=False)
    center_ckpt_path = Path(basis_blob['center_ckpt'])
    names_sorted = basis_blob['names_sorted']
    shapes = basis_blob['shapes']
    w_center_cpu = basis_blob['w_center'].float()
    basis_cpu = basis_blob['basis'].float()  # [D_sub, D_full]
    D_sub = basis_cpu.shape[0]
    stage0_soups = basis_blob['stage0_soups']

    all_cases = json.loads(args.cases_spec.read_text())['cases']
    assert len(all_cases) >= 128 and len(all_cases) % 32 == 0
    n_quarters = len(all_cases) // 32
    quarters = [all_cases[q * 32:(q + 1) * 32] for q in range(n_quarters)]
    lanes = 256

    execution_root = args.g07_execution_root.resolve()
    trained = torch.load(center_ckpt_path, map_location='cpu', weights_only=False,
                         mmap=zipfile.is_zipfile(center_ckpt_path))
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

    init_cases = all_cases[:64] * 4
    cfg = make_task_cfg(lanes, contract, protocol, receipt, execution_root=execution_root)
    cfg.seed, cfg.sim.device = args.seed, args.physics_device
    validate_sweep_contract(cfg, contract, None)
    noise = zero_panel_reset_noise(lanes, args.seed)
    env = EsEnv(cfg, cases=init_cases, reset_noise=noise,
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
    arm_prep = policy_execution.get('arm_preparation_seconds', 0.25)
    actor.configure_approach_gate(arm_prep)
    actor.eval()
    rec_mode = policy_execution['recovery_gate']

    w_center = w_center_cpu.to(device)
    basis_dev = basis_cpu.to(device)
    # Build unpack slices for names_sorted -> (W0, b0, W2, b2, W4, b4)
    slices = {}
    cur = 0
    for k in names_sorted:
        sz = int(np.prod(shapes[k]))
        slices[k] = (cur, cur + sz, shapes[k])
        cur += sz

    def unpack_app(w_flat):
        def get(k):
            a, b, sh = slices[k]
            return w_flat[a:b].reshape(sh)
        return (
            get('actor.approach.0.weight'), get('actor.approach.0.bias'),
            get('actor.approach.2.weight'), get('actor.approach.2.bias'),
            get('actor.approach.4.weight'), get('actor.approach.4.bias'),
        )

    def save_ckpt(w_flat_cpu, out_path: Path, meta_extra=None):
        st = {k: v.detach().cpu().clone() for k, v in trained['actor'].items()}
        for k in names_sorted:
            a, b, sh = slices[k]
            st[k] = w_flat_cpu[a:b].reshape(sh).to(st[k].dtype)
        for k, v in trained['actor'].items():
            if k not in APPROACH_NAMES and torch.is_tensor(v):
                assert torch.equal(st[k], v.cpu()), f'Frozen param mutated: {k}'
        out_ck = copy.deepcopy(trained)
        out_ck['actor'] = st
        torch.save(out_ck, out_path)
        if meta_extra is not None:
            out_path.with_suffix('.json').write_text(json.dumps(meta_extra, indent=2) + '\n')

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

    # Capture ONE-TIME standing snapshot @ tick 197
    t_pre = time.perf_counter()
    obs, _ = wrapper.reset()
    with torch.inference_mode():
        for tick in range(args.snapshot_tick):
            p_obs = obs['policy']
            act = policy_forward(actor, p_obs, env.rally, mode=rec_mode, sample=False)[0].clamp(-1.0, 1.0)
            if env.approach_ball_termination is not None:
                env.record_approach_gate(actor.approach_action_gate(p_obs))
            obs, _, _, _ = wrapper.step(act)
    snapshot = NativeApproachRestart.capture(env)
    obs_snap = obs['policy'].detach().clone()
    print(f'[ES-256] Snapshot @ tick {args.snapshot_tick} captured in {time.perf_counter()-t_pre:.1f}s! (D_sub={D_sub})', flush=True)

    # Bit-exact parity self-test of batched multi-candidate forward!
    mod = actor.actor
    def batched_forward(policy_obs, app_tuples, chunk):
        norm = mod.normalizer(policy_obs)
        rg = recovery_gate(env.rally, policy_obs, mode=rec_mode).to(policy_obs.dtype)
        if rg.ndim == 1:
            rg = rg[:, None]
        frozen = mod.base(norm) + rg * mod.recovery(norm) * mod.recovery_scale
        agate = actor.approach_action_gate(policy_obs)
        app_all = torch.empty((lanes, 18), dtype=policy_obs.dtype, device=device)
        for m, (W0, b0, W2, b2, W4, b4) in enumerate(app_tuples):
            sl = slice(m * chunk, (m + 1) * chunk)
            h = F.elu(F.linear(norm[sl], W0, b0))
            h = F.elu(F.linear(h, W2, b2))
            app_all[sl] = F.linear(h, W4, b4)
        return (frozen + agate * app_all * mod.approach_scale).clamp(-1.0, 1.0), agate

    with torch.inference_mode():
        ref_act = policy_forward(actor, obs_snap, env.rally, mode=rec_mode, sample=False)[0].clamp(-1.0, 1.0)
        test_act, _ = batched_forward(obs_snap, [unpack_app(w_center)] * 8, 32)
        parity_err = float((ref_act - test_act).abs().max())
        assert parity_err < 1e-5, f'Batched forward parity failed: {parity_err}'
        print(f'[ES-256] Batched 8-way forward parity verified: max_abs_diff={parity_err:.2e}', flush=True)

    n_sim_ticks = args.max_ticks - args.snapshot_tick

    def eval_candidates(w_list, panel_cases):
        """Evaluate M candidates (M in {2, 4, 8}) on identical panel_cases (len = 256 // M)."""
        M = len(w_list)
        chunk = lanes // M
        assert len(panel_cases) == chunk and M * chunk == lanes
        full_cases = panel_cases * M
        app_tuples = [unpack_app(w.to(device)) for w in w_list]
        snapshot.restore(env)
        env.cases = full_cases
        policy_obs = obs_snap.clone()
        first = {}
        cursor = 0
        with torch.inference_mode():
            for _ in range(n_sim_ticks):
                if stop_request['signal']:
                    break
                act, agate = batched_forward(policy_obs, app_tuples, chunk)
                if env.approach_ball_termination is not None:
                    env.record_approach_gate(agate)
                obs_n, _, _, _ = wrapper.step(act)
                policy_obs = obs_n['policy']
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
        rows = native_fitness(ordered_raw, env.court_config, signed_string_margin)['rows']
        return [summarize_chunk(rows[m * chunk:(m + 1) * chunk], ordered_raw[m * chunk:(m + 1) * chunk]) for m in range(M)]

    log_path = args.output / 'es_log.jsonl'
    def log_ev(d):
        with log_path.open('a') as f:
            f.write(json.dumps(d) + '\n')

    # === STAGE 0: Screen 8 strategic basin soups on 64 stratified serves (2 rollouts!) ===
    panel64 = all_cases[:64]
    s0_results = []
    for half in (0, 1):
        sub = stage0_soups[half * 4:(half + 1) * 4]
        t0 = time.perf_counter()
        stats = eval_candidates([item['w_exact'] for item in sub], panel64)
        dt = time.perf_counter() - t0
        for item, st in zip(sub, stats):
            s0_results.append((item, st))
            print(f"[STAGE0] ({dt:5.1f}s) {item['name']:24s}: ok={st['ok']:2d}/64 (A={st['okA']:2d}/{st['nA']} B={st['okB']:2d}/{st['nB']}) "
                  f"str={st['str']:2d} loob={st['loob']:2d} low={st['low']:2d} bnet={st['bnet']} woob={st['woob']} rnet={st['rnet']} fit={st['fit']:+.3f}", flush=True)
            log_ev(dict(stage='stage0', name=item['name'], **st))

    # Pick Stage-0 winner (rnet == 0 required; rank by ok then fit!)
    valid_s0 = [(item, st) for item, st in s0_results if st['rnet'] == 0]
    valid_s0.sort(key=lambda x: (x[1]['ok'], x[1]['fit']), reverse=True)
    best_s0_item, best_s0_st = valid_s0[0]
    print(f"[STAGE0 WINNER] {best_s0_item['name']} (ok={best_s0_st['ok']}/64, A={best_s0_st['okA']}/{best_s0_st['nA']}, B={best_s0_st['okB']}/{best_s0_st['nB']}, fit={best_s0_st['fit']:+.3f})", flush=True)

    # Anchor ES center directly at Stage-0 winner's EXACT weight vector (zero projection error!)
    w0_base = best_s0_item['w_exact'].float().clone()
    theta = torch.zeros(D_sub, dtype=torch.float32)
    best_theta = theta.clone()
    best_w = w0_base.clone()
    save_ckpt(best_w, args.output / 'es_stage0_winner.pt', dict(stage='stage0', name=best_s0_item['name'], **best_s0_st))

    # Initial 128-serve baseline check of Stage-0 winner vs H8d (exact 128x2 paired check!)
    panel128 = all_cases[:128]
    t_b = time.perf_counter()
    st_h8_128, st_s0_128 = eval_candidates([w_center_cpu, w0_base], panel128)
    print(f"[VERIFY128 INIT] ({time.perf_counter()-t_b:5.1f}s) "
          f"H8d={st_h8_128['ok']:3d}/128 (A={st_h8_128['okA']}/{st_h8_128['nA']} B={st_h8_128['okB']}/{st_h8_128['nB']} fit={st_h8_128['fit']:+.3f} rnet={st_h8_128['rnet']}) | "
          f"S0({best_s0_item['name']})={st_s0_128['ok']:3d}/128 (A={st_s0_128['okA']}/{st_s0_128['nA']} B={st_s0_128['okB']}/{st_s0_128['nB']} fit={st_s0_128['fit']:+.3f} rnet={st_s0_128['rnet']})", flush=True)
    log_ev(dict(stage='verify128_init', h8d=st_h8_128, s0=st_s0_128))
    if st_s0_128['rnet'] == 0 and (st_s0_128['ok'], st_s0_128['fit']) >= (st_h8_128['ok'], st_h8_128['fit']):
        best_128 = st_s0_128
    else:
        w0_base = w_center_cpu.clone()
        best_w = w0_base.clone()
        best_128 = st_h8_128
        print('[VERIFY128 INIT] Reverting center to H8d on 128-serve benchmark!', flush=True)
    save_ckpt(best_w, args.output / 'es_best.pt', dict(step=0, **best_128))

    # === STAGE 1: Antithetic Subspace ES with Common Random Numbers (4 pairs x 32 serves = 256 lanes) ===
    rng = np.random.default_rng(args.seed + 7711)
    m_adam = torch.zeros(D_sub, dtype=torch.float32)
    v_adam = torch.zeros(D_sub, dtype=torch.float32)
    beta1, beta2 = 0.60, 0.90
    verified_elites = [(best_128['ok'], best_128['fit'], best_w.clone())]

    for step in range(1, args.es_steps + 1):
        if stop_request['signal']:
            break
        q_idx = (step - 1) % n_quarters
        q_cases = quarters[q_idx]
        frac = (step - 1) / max(1, args.es_steps - 1)
        sigma = args.sigma_end + 0.5 * (args.sigma_start - args.sigma_end) * (1.0 + math.cos(math.pi * frac))

        # 4 orthogonalized antithetic unit perturbations via QR -> zero redundant variance!
        Z = rng.standard_normal((D_sub, 4))
        Q, _ = np.linalg.qr(Z)
        U_dirs = torch.tensor((Q * math.sqrt(D_sub)).T, dtype=torch.float32)  # [4, D_sub], ||u|| = sqrt(D_sub)

        w_cands = []
        th_cands = []
        for k in range(4):
            tp = theta + sigma * U_dirs[k]
            tm = theta - sigma * U_dirs[k]
            th_cands.extend([tp, tm])
            w_cands.extend([w0_base + tp @ basis_cpu, w0_base + tm @ basis_cpu])

        t_s = time.perf_counter()
        stats = eval_candidates(w_cands, q_cases)
        dt = time.perf_counter() - t_s

        fits = np.array([st['fit'] for st in stats], dtype=np.float64)
        oks = [st['ok'] for st in stats]
        rnets = [st['rnet'] for st in stats]
        # Antithetic CRN gradient estimate (scale-normalized!)
        diffs = np.array([fits[2 * k] - fits[2 * k + 1] for k in range(4)], dtype=np.float64)
        d_scale = max(0.15, float(np.std(fits)))
        grad = torch.zeros(D_sub, dtype=torch.float32)
        for k in range(4):
            grad += float(diffs[k] / (2.0 * d_scale)) * (U_dirs[k] / math.sqrt(D_sub))
        grad = grad / 4.0 - args.l2_reg * theta

        # Adam ascent step on theta
        m_adam = beta1 * m_adam + (1.0 - beta1) * grad
        v_adam = beta2 * v_adam + (1.0 - beta2) * (grad * grad)
        m_hat = m_adam / (1.0 - beta1 ** step)
        v_hat = v_adam / (1.0 - beta2 ** step)
        step_vec = args.lr * m_hat / (v_hat.sqrt() + 1e-6)
        theta = (theta + step_vec.clamp(-0.35, 0.35)).clamp(-2.8, 2.8)
        w_curr = w0_base + theta @ basis_cpu

        print(f"[ES Step {step:02d}/{args.es_steps:02d} Q{q_idx}] ({dt:5.1f}s) sig={sigma:.3f} | "
              f"oks={oks} max={max(oks):2d}/32 rnets={rnets} | fit_mean={fits.mean():+.3f} d_max={np.abs(diffs).max():.3f} ||th||={theta.norm():.3f}", flush=True)
        log_ev(dict(stage='es_step', step=step, quarter=q_idx, sigma=sigma, oks=oks, rnets=rnets,
                    fits=fits.tolist(), theta_norm=float(theta.norm())))

        # Periodic 128-serve paired check (theta_curr vs best_w on ALL 128 serves!)
        if step % args.verify_every == 0 or step == args.es_steps:
            t_v = time.perf_counter()
            st_best_v, st_curr_v = eval_candidates([best_w, w_curr], panel128)
            dt_v = time.perf_counter() - t_v
            improved = (st_curr_v['rnet'] == 0) and ((st_curr_v['ok'], st_curr_v['fit']) > (st_best_v['ok'], st_best_v['fit']))
            tag = 'PROMOTED!' if improved else 'kept_best'
            print(f"[VERIFY128 Step {step:02d}] ({dt_v:5.1f}s) "
                  f"BEST={st_best_v['ok']:3d}/128 (A={st_best_v['okA']}/{st_best_v['nA']} B={st_best_v['okB']}/{st_best_v['nB']} str={st_best_v['str']} loob={st_best_v['loob']} fit={st_best_v['fit']:+.3f}) | "
                  f"CURR={st_curr_v['ok']:3d}/128 (A={st_curr_v['okA']}/{st_curr_v['nA']} B={st_curr_v['okB']}/{st_curr_v['nB']} str={st_curr_v['str']} loob={st_curr_v['loob']} fit={st_curr_v['fit']:+.3f} rnet={st_curr_v['rnet']}) -> {tag}", flush=True)
            log_ev(dict(stage='verify128', step=step, best=st_best_v, curr=st_curr_v, promoted=improved))
            save_ckpt(w_curr, args.output / f'es_step{step:02d}.pt', dict(step=step, **st_curr_v))
            if st_curr_v['rnet'] == 0:
                verified_elites.append((st_curr_v['ok'], st_curr_v['fit'], w_curr.clone()))
            if improved:
                best_w = w_curr.clone()
                best_theta = theta.clone()
                best_128 = st_curr_v
                save_ckpt(best_w, args.output / 'es_best.pt', dict(step=step, **best_128))
            elif st_curr_v['ok'] <= st_best_v['ok'] - 4:
                # Soft elastic pullback toward best_theta if wandering off-basin
                theta = 0.65 * best_theta + 0.35 * theta
                print(f'  [Pullback] Pulled theta softly back toward best_theta (||th||={theta.norm():.3f})', flush=True)

    # Save Top-3 Elite Soup checkpoint too!
    verified_elites.sort(key=lambda x: (x[0], x[1]), reverse=True)
    top_k = verified_elites[:min(3, len(verified_elites))]
    w_soup = sum(x[2] for x in top_k) / len(top_k)
    save_ckpt(w_soup, args.output / 'es_elite_soup.pt', dict(n_elites=len(top_k), top_oks=[x[0] for x in top_k]))
    print(f'[ES-256 DONE] best_128={best_128["ok"]}/128 (A={best_128["okA"]}/{best_128["nA"]}, B={best_128["okB"]}/{best_128["nB"]}, rnet={best_128["rnet"]}) | saved es_best.pt & es_elite_soup.pt!', flush=True)


if __name__ == '__main__':
    main()
