#!/usr/bin/env python3
"""Evaluate up to 4 policy checkpoints simultaneously in one 256-lane PhysX batch on identical serves."""
import argparse
import faulthandler
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('QUADSMASH_DISABLE_APP_SHUTDOWN', '1')
from isaaclab.app import AppLauncher


def write_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--g07-execution-root', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=882072)
    parser.add_argument('--physics-device', default='cuda:0')
    parser.add_argument('--max-ticks', type=int, default=550)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    from quadsmash.rl.supermimic_run_lifecycle import stage as lifecycle_stage, execute_scoped_native

    fault_fd = os.open(args.output / 'native_fault.log', os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    faulthandler.enable(file=fault_fd, all_threads=True)
    lifecycle_stage(args.output, 'BEFORE_APP')
    stop_request = {'signal': None}
    app = AppLauncher(args, fast_shutdown=False).app
    faulthandler.enable(file=fault_fd, all_threads=True)
    lifecycle_stage(args.output, 'AFTER_APP')
    signal.signal(signal.SIGINT, lambda s, f: stop_request.__setitem__('signal', signal.Signals(s).name))
    signal.signal(signal.SIGTERM, lambda s, f: stop_request.__setitem__('signal', signal.Signals(s).name))
    execute_scoped_native(args.output, lambda: _run_native(args, stop_request), app)


def _run_native(args, stop_request):
    from quadsmash.rl.supermimic_run_lifecycle import stage as lifecycle_stage, close_resource
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from quadsmash.rl.supermimic_v1 import load_ppo_initialized_actor
    from quadsmash.rl.supermimic_task_contract import (
        resolve_task_contract, make_task_cfg, require_resume_contract,
        zero_panel_reset_noise, validate_sweep_contract)
    from quadsmash.rl.supermimic_terminal_scoring import (
        score_terminal_episode, approach_ball_scoring_protocol)
    from quadsmash.rl.supermimic_policy_execution_v1 import resolve_contract, recovery_gate
    from quadsmash.physics.full_racket_v4 import string_polygon
    from quadsmash.rl.supermimic_native_approach_search import native_fitness
    from quadsmash.tasks.stand.observation_contract import inverse_rotate
    from quadsmash.tasks.stand.supermimic_env_v1 import SuperMimicEnvV1
    from quadsmash.tasks.stand.full_racket_eval_env_v4 import FrozenPanelEnv

    class EvaluationEnv(SuperMimicEnvV1, FrozenPanelEnv):
        pass

    env = None
    try:
        torch.manual_seed(args.seed)
        base_cases = json.loads(args.spec.read_text())['cases']
        n_cases = len(base_cases)
        n_models = len(args.checkpoints)
        cases = base_cases * n_models
        lanes = len(cases)

        first_ckpt = torch.load(args.checkpoints[0], map_location='cpu', weights_only=False,
                                mmap=zipfile.is_zipfile(args.checkpoints[0]))
        init_meta = first_ckpt['metadata']['init_policy_checkpoint']
        source_path = Path(init_meta['path'])
        execution_root = args.g07_execution_root or Path(init_meta['g07_execution_root'])
        source = torch.load(source_path, map_location='cpu', weights_only=False)
        contract, protocol, _, receipt = resolve_task_contract(
            source, source_path, execution_root=execution_root, root=execution_root)
        scoring_protocol = dict(protocol, _root=str(execution_root))
        contract = dict(contract, approach_ball_scoring=approach_ball_scoring_protocol())
        scoring_protocol['approach_ball_scoring'] = contract['approach_ball_scoring']
        scoring_protocol['relaxed_scoring'] = None

        cfg = make_task_cfg(lanes, contract, protocol, receipt, execution_root=execution_root)
        cfg.seed, cfg.sim.device = args.seed, args.physics_device
        validate_sweep_contract(cfg, contract, None)
        noise = zero_panel_reset_noise(lanes, args.seed)
        env = EvaluationEnv(cfg, cases=cases, reset_noise=noise,
                            residual_policy_checkpoint=source, g07_execution_root=execution_root,
                            contact_flight_recovery=contract['contact_flight_recovery'],
                            relaxed_protocol=None,
                            approach_ball_protocol=contract['approach_ball_scoring'],
                            relaxed_survival_diagnostic=False)
        wrapper = RslRlVecEnvWrapper(env, clip_actions=None)
        obs, _ = wrapper.reset()

        actors = []
        policy_execution = None
        for path in args.checkpoints:
            ck = torch.load(path, map_location='cpu', weights_only=False, mmap=zipfile.is_zipfile(path))
            require_resume_contract(ck, contract)
            pe, _ = resolve_contract(ck)
            if policy_execution is None:
                policy_execution = pe
            actor, _ = load_ppo_initialized_actor(source_path, env.device)
            actor.load_state_dict(ck['actor'], strict=True)
            actor.eval()
            if pe.get('arm_preparation_seconds') is not None:
                actor.configure_approach_gate(pe['arm_preparation_seconds'])
            actors.append(actor)

        first = {}
        cursor = 0
        start = time.perf_counter()
        with torch.inference_mode():
            for tick in range(args.max_ticks):
                if stop_request['signal']:
                    break
                before = obs['policy']
                latch = recovery_gate(env.rally, before, mode=policy_execution['recovery_gate'])
                actions = []
                gates = []
                for m_idx, actor in enumerate(actors):
                    sl = slice(m_idx * n_cases, (m_idx + 1) * n_cases)
                    b_sl = before[sl]
                    l_sl = latch[sl]
                    actions.append(actor._gated_distribution(b_sl, l_sl).loc.clamp(-1.0, 1.0))
                    gates.append(actor.approach_action_gate(b_sl))
                action = torch.cat(actions, dim=0)
                gate_vec = torch.cat(gates, dim=0)
                if env.approach_ball_termination is not None:
                    env.record_approach_gate(gate_vec)
                obs, _, done, _ = wrapper.step(action)

                for record in env.rally_records[cursor:]:
                    lane = int(record['env_id'])
                    if lane not in first:
                        measured = score_terminal_episode(record, scoring_protocol)
                        if measured['episode_index'] != 0:
                            raise RuntimeError(f'Lane {lane} missed episode 0')
                        first[lane] = measured
                cursor = len(env.rally_records)

                if (tick + 1) % 100 == 0 or len(first) == lanes:
                    print(f'MULTI_EVAL tick={tick+1} settled={len(first)}/{lanes} elapsed={time.perf_counter()-start:.1f}s',
                          flush=True)
                if len(first) == lanes:
                    break

        polygon = torch.tensor(string_polygon(env.racket_config))
        edge = polygon.roll(-1, 0) - polygon
        inward = torch.nn.functional.normalize(torch.stack((-edge[:, 1], edge[:, 0]), -1), dim=-1)
        boundary = (polygon * inward).sum(-1)

        def signed_margin(motion):
            if not all(k in motion for k in ('point_m', 'racket_center_m', 'racket_quaternion_w')):
                return None
            local = inverse_rotate(
                polygon.new_tensor([motion['racket_quaternion_w']]),
                polygon.new_tensor([motion['point_m']]) - polygon.new_tensor([motion['racket_center_m']]))
            return float((local[0, 1:] @ inward.T - boundary).min()) if local.isfinite().all() else None

        raw_ep0 = {row['env_id']: row for row in env.rally_records if row['episode_index'] == 0}
        import math as _math
        import numpy as _np
        results = []
        base_succs = None
        for m_idx, path in enumerate(args.checkpoints):
            sl_ids = list(range(m_idx * n_cases, (m_idx + 1) * n_cases))
            group = [raw_ep0[i] for i in sl_ids]
            scored = native_fitness(group, env.court_config, signed_margin)
            ret_mean = sum(first[i]['native_episode_return'] for i in sl_ids) / n_cases
            low_n = sum(r['low_speed'] for r in scored['rows'])
            frm_n = sum(r['frame'] and not r['string'] for r in scored['rows'])
            noc_n = sum(not r['string'] and not r['frame'] for r in scored['rows'])
            bnet_str_n = 0
            long_oob_n = 0
            short_oob_n = 0
            wide_oob_n = 0
            lxs, spds, nms = [], [], []
            for ep_row in group:
                if ep_row['qualified_airborne_string_contact']:
                    lm = ep_row.get('first_landing_m')
                    ov = ep_row.get('outgoing_velocity_mps')
                    nm = ep_row.get('net_margin_m')
                    if lm and _math.isfinite(lm[0]):
                        lxs.append(float(lm[0]))
                    if ov and all(_math.isfinite(x) for x in ov):
                        spds.append(float(_math.sqrt(sum(x * x for x in ov))))
                    if nm is not None and _math.isfinite(nm):
                        nms.append(float(nm))
                    if not ep_row['ball_side_complete_except_recovery']:
                        if ep_row['shuttle_net_contact'] or ep_row.get('net_blocked_fault') or not ep_row['crossed_net']:
                            bnet_str_n += 1
                        elif not ep_row['landed_target'] and lm and _math.isfinite(lm[0]):
                            if lm[0] > 4.0:
                                long_oob_n += 1
                            elif lm[0] < 2.0:
                                short_oob_n += 1
                            else:
                                wide_oob_n += 1
            lx_med = float(_np.median(lxs)) if lxs else float('nan')
            lx_q90 = float(_np.quantile(lxs, 0.90)) if lxs else float('nan')
            spd_q10 = float(_np.quantile(spds, 0.10)) if spds else float('nan')
            nm_q10 = float(_np.quantile(nms, 0.10)) if nms else float('nan')
            succs = [bool(r['success']) for r in scored['rows']]
            half = n_cases // 2
            a_succ = sum(succs[:half])
            b_succ = sum(succs[half:])
            pair_str = ''
            if base_succs is None:
                base_succs = succs
            else:
                gained = sum((not b) and s for b, s in zip(base_succs, succs))
                lost = sum(b and (not s) for b, s in zip(base_succs, succs))
                pair_str = f' [vs0: +{gained}/-{lost} net={gained-lost:+d}]'
            row_out = dict(
                checkpoint=str(path), name=path.name,
                success=scored['success'], panel_a=a_succ, panel_b=b_succ,
                qualified=scored['qualified'],
                frame=frm_n, no_contact=noc_n, low_speed=low_n,
                ball_net=bnet_str_n, long_oob=long_oob_n, short_oob=short_oob_n, wide_oob=wide_oob_n,
                robot_net=scored['robot_net'], margin_score=scored['margin_score'],
                return_mean=ret_mean,
                lx_med=lx_med, lx_q90=lx_q90, spd_q10=spd_q10, nm_q10=nm_q10,
            )
            results.append(row_out)
            print(f'CANDIDATE {path.name:<32}: succ={row_out["success"]:3d}/{n_cases} (A={a_succ:3d}/{half} B={b_succ:3d}/{n_cases-half}) '
                  f'str={row_out["qualified"]:3d} frm={frm_n:2d} noc={noc_n:2d} '
                  f'loob={long_oob_n:2d} low={low_n:2d} bnet={bnet_str_n:2d} woob={wide_oob_n:2d} rnet={row_out["robot_net"]} '
                  f'lx={lx_med:.2f}(q90={lx_q90:.2f}) spd10={spd_q10:.2f} nm10={nm_q10:.2f} '
                  f'ret={ret_mean:.2f}{pair_str}', flush=True)

        write_json(args.output / 'comparison.json', dict(n_cases=n_cases, elapsed_s=time.perf_counter()-start, results=results))
        lifecycle_stage(args.output, 'COMPLETE')
    finally:
        if env is not None:
            close_resource(args.output, 'ENV', env, primary_failed=False)


if __name__ == '__main__':
    os.environ['QUADSMASH_SKIP_KIT_CLOSE'] = '1'
    try:
        main()
    except BaseException:
        traceback.print_exc()
        os._exit(1)
    else:
        os._exit(0)
