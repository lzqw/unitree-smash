#!/usr/bin/env python3
"""Run batched native PhysX rollouts with optional per-lane smooth pre-impact perturbations.

Preserves the exact physics, G07 execution contract, causal recovery gate, and scoring
rules of evaluate_supermimic_v1.py while recording full compact trajectories for
closed-loop native teacher discovery and distillation.
"""
import argparse
import faulthandler
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
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
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--g07-execution-root', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=882072)
    parser.add_argument('--physics-device', default='cuda:0')
    parser.add_argument('--start-capture-tick', type=int, default=200)
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
    from quadsmash.rl.supermimic_policy_execution_v1 import resolve_contract, policy_forward, recovery_gate
    from quadsmash.rl.supermimic_physical_loop_loading_v1 import tensor_digest
    from quadsmash.rl.supermimic_physical_dataset_v1 import STATE_FIELDS, encode_physical
    from quadsmash.rl.supermimic_physical_observation_v1 import capture_native_inputs
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
        spec = json.loads(args.spec.read_text())
        cases = spec['cases']
        lanes = len(cases)

        trained = torch.load(args.checkpoint, map_location='cpu', weights_only=False,
                             mmap=zipfile.is_zipfile(args.checkpoint))
        init_meta = trained['metadata']['init_policy_checkpoint']
        source_path = Path(init_meta['path'])
        execution_root = args.g07_execution_root or Path(init_meta['g07_execution_root'])
        source = torch.load(source_path, map_location='cpu', weights_only=False)
        contract, protocol, _, receipt = resolve_task_contract(
            source, source_path, execution_root=execution_root, root=execution_root)
        scoring_protocol = dict(protocol, _root=str(execution_root))
        contract = dict(contract, approach_ball_scoring=approach_ball_scoring_protocol())
        require_resume_contract(trained, contract)
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

        actor, _ = load_ppo_initialized_actor(source_path, env.device)
        actor.load_state_dict(trained['actor'], strict=True)
        actor.eval()
        policy_execution, _ = resolve_contract(trained)
        if policy_execution.get('arm_preparation_seconds') is not None:
            actor.configure_approach_gate(policy_execution['arm_preparation_seconds'])
        actor_sha = tensor_digest(actor.state_dict().items())

        # Per-lane perturbation specification
        has_offsets = 'offsets' in spec and spec['offsets'] is not None
        offsets = torch.tensor(spec['offsets'], dtype=torch.float32, device=env.device) if has_offsets else None
        ramp_offsets = (torch.tensor(spec['ramp_offsets'], dtype=torch.float32, device=env.device)
                        if spec.get('ramp_offsets') is not None else None)
        peak_ticks = (torch.tensor(spec['peak_ticks'], dtype=torch.float32, device=env.device)
                      if spec.get('peak_ticks') is not None else torch.full((lanes,), 275.0, device=env.device))
        width_ticks = (torch.tensor(spec['width_ticks'], dtype=torch.float32, device=env.device)
                       if spec.get('width_ticks') is not None else torch.full((lanes,), 55.0, device=env.device))
        leg_offsets = (torch.tensor(spec['leg_offsets'], dtype=torch.float32, device=env.device)
                       if spec.get('leg_offsets') is not None else None)
        leg_peak_ticks = (torch.tensor(spec['leg_peak_ticks'], dtype=torch.float32, device=env.device)
                          if spec.get('leg_peak_ticks') is not None else torch.full((lanes,), 246.0, device=env.device))
        leg_width_ticks = (torch.tensor(spec['leg_width_ticks'], dtype=torch.float32, device=env.device)
                           if spec.get('leg_width_ticks') is not None else torch.full((lanes,), 12.0, device=env.device))
        gate_mode = spec.get('gate_mode', 'approach_gate')  # 'approach_gate' or 'present_precontact'
        record_tape = bool(spec.get('record_tape', True))

        frames = []
        first = {}
        cursor = 0
        start = time.perf_counter()

        with torch.inference_mode():
            for tick in range(args.max_ticks):
                if stop_request['signal']:
                    break
                before = obs['policy']
                latch = recovery_gate(env.rally, before, mode=policy_execution['recovery_gate'])
                base_req = actor._gated_distribution(before, latch).loc
                gate_vec = actor.approach_action_gate(before)
                if has_offsets:
                    t = float(tick)
                    # Smooth C1 ramp peaking at peak_ticks, active before contact
                    rise = ((t - (peak_ticks - width_ticks)) / width_ticks.clamp_min(1.0)).clamp(0.0, 1.0)
                    fall = ((t - peak_ticks) / 16.0).clamp(0.0, 1.0)
                    env_pos = torch.where(t <= peak_ticks, torch.sin(0.5 * math.pi * rise).square(),
                                          torch.cos(0.5 * math.pi * fall).square())
                    pre = (~env.rally.had_contact).float()
                    delta = offsets * (env_pos * pre)[:, None]
                    if ramp_offsets is not None:
                        ramp = ((t - (peak_ticks - 18.0)) / 18.0).clamp(0.0, 1.0) * (t <= peak_ticks + 4.0)
                        delta = delta + ramp_offsets * (ramp * pre)[:, None]
                    if gate_mode == 'approach_gate':
                        action = (base_req + gate_vec * delta * actor.actor.approach_scale).clamp(-1.0, 1.0)
                    else:
                        causal_pre = (env.ball_observation_enabled & (~env.rally.had_contact) & (env.episode_index == 0)).float()
                        total_delta = delta * causal_pre[:, None]
                        if leg_offsets is not None:
                            lrise = ((t - (leg_peak_ticks - leg_width_ticks)) / leg_width_ticks.clamp_min(1.0)).clamp(0.0, 1.0)
                            lfall = ((t - leg_peak_ticks) / 8.0).clamp(0.0, 1.0)
                            lenv = torch.where(t <= leg_peak_ticks, torch.sin(0.5 * math.pi * lrise).square(),
                                               torch.cos(0.5 * math.pi * lfall).square())
                            total_delta = total_delta + leg_offsets * (lenv * causal_pre)[:, None]
                        action = (base_req + total_delta).clamp(-1.0, 1.0)
                else:
                    action = base_req.clamp(-1.0, 1.0)

                if env.approach_ball_termination is not None:
                    env.record_approach_gate(gate_vec)

                frame = None
                if record_tape and tick >= args.start_capture_tick:
                    physical = capture_native_inputs(env)
                    frame = dict(
                        state=encode_physical({k: getattr(physical, k) for k, _ in STATE_FIELDS}).half().cpu(),
                        observation=before.half().cpu(),
                        request=action.float().cpu(),
                        gate=gate_vec.half().cpu(),
                        elapsed=physical.elapsed_s.float().cpu(),
                        present=physical.ball_present.bool().cpu(),
                        episode=env.episode_index.to(torch.int16).cpu(),
                        had_contact=env.rally.had_contact.bool().cpu(),
                        released=env.rally.released.bool().cpu(),
                    )

                obs, _, done, _ = wrapper.step(action)
                if frame is not None:
                    frames.append(frame)

                for record in env.rally_records[cursor:]:
                    lane = int(record['env_id'])
                    if lane not in first:
                        measured = score_terminal_episode(record, scoring_protocol)
                        if measured['episode_index'] != 0:
                            raise RuntimeError(f'Lane {lane} missed episode 0')
                        first[lane] = measured
                        with (args.output / 'episodes.jsonl').open('a') as stream:
                            stream.write(json.dumps(record) + '\n')
                cursor = len(env.rally_records)

                if (tick + 1) % 50 == 0 or len(first) == lanes:
                    print(f'PROBE_PROGRESS tick={tick+1} settled={len(first)}/{lanes} elapsed={time.perf_counter()-start:.1f}s',
                          flush=True)
                if len(first) == lanes:
                    break

        elapsed_s = time.perf_counter() - start
        rows = [dict(first[i], source_case_index=i, case_id=cases[i]['case_id']) for i in range(lanes)]
        write_json(args.output / 'episodes_summary.json', rows)
        contacts = [event for event in env.contact_records if event['episode_index'] == 0]
        write_json(args.output / 'native_contact_motion.json', contacts)

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
        scored = native_fitness([raw_ep0[i] for i in range(lanes)], env.court_config, signed_margin)
        summary = dict(
            lanes=lanes,
            settled=len(first),
            elapsed_s=elapsed_s,
            actor_sha256=actor_sha,
            checkpoint=str(args.checkpoint),
            success=scored['success'],
            qualified=scored['qualified'],
            robot_net=scored['robot_net'],
            margin_score=scored['margin_score'],
            rows=scored['rows'],
            branch_meta=spec.get('branch_meta'),
        )
        write_json(args.output / 'probe_summary.json', summary)

        if record_tape and frames:
            tape = {k: torch.stack([f[k] for f in frames]) for k in frames[0]}
            torch.save(dict(
                schema='native-guided-probe-v1',
                actor_sha256=actor_sha,
                start_capture_tick=args.start_capture_tick,
                case_ids=[c['case_id'] for c in cases],
                branch_meta=spec.get('branch_meta'),
                tape=tape,
            ), args.output / 'probe_tape.pt')
        lifecycle_stage(args.output, 'COMPLETE')
        print(f'PROBE_DONE lanes={lanes} success={scored["success"]} qualified={scored["qualified"]} '
              f'robot_net={scored["robot_net"]} elapsed={elapsed_s:.1f}s', flush=True)
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
