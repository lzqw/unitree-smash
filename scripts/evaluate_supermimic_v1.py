#!/usr/bin/env python3
"""Frozen TRAIN128 evaluation with source task and pre-reset terminal evidence.

This is a development regression, not the independent A/B formal acceptance.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import signal
import faulthandler

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--checkpoint', type=Path)
    source_group.add_argument('--init-policy-checkpoint', type=Path,
                              help='Zero-update native source evaluation, no world model or optimizers')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--g07-execution-root', type=Path)
    parser.add_argument('--precision-receipt', type=Path)
    parser.add_argument('--seed', type=int, default=882072)
    parser.add_argument('--sweep-audit-iterations', type=int, default=None,
                        help='Inherit source audit by default; only an identical explicit value is accepted')
    parser.add_argument('--physics-device', default='cuda:0')
    parser.add_argument('--record-physical-transitions', action='store_true',
                        help='Read-only pre-reset physical/executor archive; no model/actor updates')
    AppLauncher.add_app_launcher_args(parser)
    parser.add_argument('--shutdown-diagnostic-ticks', type=int, default=None,
                        help='Lifecycle-only partial run on original 128 lanes; never writes scores/completed.json')
    args = parser.parse_args()
    if args.shutdown_diagnostic_ticks is not None and not 1 <= args.shutdown_diagnostic_ticks < 800:
        parser.error('shutdown diagnostic ticks must be in [1,799]')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    startup = dict(argv=sys.argv, cwd=os.getcwd(), script=str(Path(__file__).resolve()),
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                   dirty_diff=subprocess.check_output(['git','diff','--no-ext-diff'],cwd=ROOT,text=True),
                   executable=sys.executable, pid=os.getpid(),
                   retained_exception_audit=os.environ.get('QUADSMASH_AUDIT_RETAINED_EXCEPTIONS') == '1',
                   shutdown_diagnostic_ticks=args.shutdown_diagnostic_ticks)
    write_json(args.output/'startup.json', startup)
    from quadsmash.rl.supermimic_run_lifecycle import stage as lifecycle_stage, execute_scoped_native
    # Keep the descriptor valid through interpreter/native finalization. The OS
    # closes it at process exit; do not disable faulthandler after app.close().
    fault_fd = os.open(args.output/'native_fault.log', os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    faulthandler.enable(file=fault_fd, all_threads=True)
    lifecycle_stage(args.output, 'BEFORE_APP')
    app = None
    stop_request = {'signal': None}
    try:
        app = AppLauncher(args, fast_shutdown=False).app
        faulthandler.enable(file=fault_fd, all_threads=True)
        lifecycle_stage(args.output, 'AFTER_APP')
        def request_stop(signum, frame):
            stop_request['signal'] = signal.Signals(signum).name
        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)
        # No native environment, wrapper, actor or tensor escapes this scope.
        execute_scoped_native(args.output, lambda: _run_native(args, stop_request), app)
    except BaseException as error:
        if not (args.output/'failure.json').exists():
            trace = traceback.format_exc()
            (args.output/'failure_traceback.txt').write_text(trace)
            write_json(args.output/'failure.json', dict(
                stage='APPLICATION_OR_SCOPED_CLEANUP', type=type(error).__name__, message=str(error)))
            print(trace, file=sys.stderr, flush=True)
        raise


def _run_native(args, stop_request):
    from quadsmash.rl.supermimic_run_lifecycle import stage as lifecycle_stage, close_resource
    env = None
    tick = -1
    ticks_completed = 0
    mapping = None
    stage = 'AFTER_APP'
    primary_failed = False
    try:
        import torch
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaacsim.core.simulation_manager import SimulationManager
        from quadsmash.rl.supermimic_v1 import load_ppo_initialized_actor
        from quadsmash.rl.supermimic_task_contract import (
            resolve_task_contract, make_task_cfg, require_resume_contract,
            zero_panel_reset_noise, validate_sweep_contract)
        from quadsmash.rl.supermimic_terminal_scoring import score_terminal_episode
        from quadsmash.rl.supermimic_timebase_v2 import full_range_targets
        from quadsmash.tasks.stand.supermimic_env_v1 import SuperMimicEnvV1
        from quadsmash.tasks.stand.full_racket_eval_env_v4 import FrozenPanelEnv
        from quadsmash.control.g07_shared_target_residual_v1 import make_g07_residual_controller
        from quadsmash.control.full_racket_residual_v4 import FullRacketResidualControllerV4

        class EvaluationEnv(SuperMimicEnvV1, FrozenPanelEnv):
            """Use the historical panel's ground-reset procedure without noise."""

        torch.manual_seed(args.seed)
        trained = None
        if args.checkpoint:
            trained = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
            if trained.get('algorithm') != 'supermimic_v1':
                raise ValueError('Expected SuperMimic checkpoint')
            init_meta = trained['metadata']['init_policy_checkpoint']
            source_path = Path(init_meta['path'])
            execution_root = args.g07_execution_root or init_meta.get('g07_execution_root')
            if hashlib.sha256(source_path.read_bytes()).hexdigest() != init_meta['sha256']:
                raise ValueError('Initialization source changed')
        else:
            source_path, execution_root = args.init_policy_checkpoint, args.g07_execution_root
        source = torch.load(source_path, map_location='cpu', weights_only=False)
        contract, protocol, cases, receipt = resolve_task_contract(
            source, source_path, execution_root=execution_root, precision_receipt=args.precision_receipt)
        scoring_protocol = dict(protocol, _root=str(ROOT))
        if trained is not None:
            require_resume_contract(trained, contract)
        write_json(args.output/'task_contract.json', contract)
        cfg = make_task_cfg(128, contract, protocol, receipt)
        cfg.seed, cfg.sim.device = args.seed, args.physics_device
        validate_sweep_contract(cfg, contract, args.sweep_audit_iterations)
        noise = zero_panel_reset_noise(len(cases), args.seed)
        stage = 'BEFORE_ENV'
        env = EvaluationEnv(cfg, cases=cases, reset_noise=noise,
            residual_policy_checkpoint=source, g07_execution_root=execution_root,
            contact_flight_recovery=contract['contact_flight_recovery'])
        stage = 'AFTER_ENV'
        wrapper = RslRlVecEnvWrapper(env, clip_actions=None)
        obs, _ = wrapper.reset()
        stage = 'AFTER_RESET'
        write_json(args.output/'initial_states.json', env.initial_states)
        actor, init_meta = load_ppo_initialized_actor(source_path, env.device)
        if trained is not None:
            actor.load_state_dict(trained['actor'], strict=True)
        actor.eval()
        if args.record_physical_transitions:
            from quadsmash.rl.supermimic_physical_capture_v1 import PhysicalTransitionArchive
            if env._sm_residual_controller is None:
                raise ValueError('Physical capture requires the initialized residual execution contract')
            env.physical_transition_archive = PhysicalTransitionArchive(
                args.output/'physical_transitions', metadata=dict(
                    source_checkpoint=str(source_path), source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                    task_contract=contract, body_names=list(env.robot.body_names),
                    control_dt_s=float(env.step_dt), num_envs=env.num_envs,
                    seed=args.seed, runtime_commit=subprocess.check_output(
                        ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
        reference = None
        if trained is None:
            reference = (make_g07_residual_controller(source,128,env.device,execution_root)
                         if source['schema']=='g07-shared-target-residual-ppo-v1'
                         else FullRacketResidualControllerV4(source,128,env.device))
            reference.reset()
        physics = dict(device=str(SimulationManager.get_physics_sim_device()),
                       gpu_dynamics=bool(SimulationManager.is_gpu_dynamics_enabled()),
                       origin_spacing_m=cfg.scene.env_spacing,
                       actual_origins_m=env.scene.env_origins.cpu().tolist(),
                       ground_size_m=list(cfg.ground_size_m), sweep_audit_iterations=cfg.sweep_audit_iterations)
        write_json(args.output/'physics.json', physics)
        first, cursor, numerical = {}, 0, 0
        mapping = dict(max_final_target_error_rad=0., tolerance_rad=2e-6,
                       phase_samples={str(i):0 for i in range(5)}, phase_max_error={str(i):0. for i in range(5)},
                       checked_native_ticks=0, scope='Independent old controller versus new actor+executor on the SAME live observations; not a separate old physics rollout')
        start = time.perf_counter()
        stage = 'EVALUATING'
        with torch.inference_mode():
            for tick in range(args.shutdown_diagnostic_ticks or 800):
                if stop_request['signal'] is not None:
                    raise InterruptedError(f"Evaluation stopped at control boundary: {stop_request['signal']}")
                before = obs['policy'].clone()
                old_target = None
                if reference is not None:
                    old_final = reference.action(obs, env._sm_residual_default, env._sm_residual_limits)
                    old_target = full_range_targets(old_final, env._sm_residual_limits).clone()
                action = actor(before, sample=False)[0].clamp(-1.,1.)
                if not torch.isfinite(action).all():
                    raise FloatingPointError('Nonfinite evaluation action')
                obs, _, done, _ = wrapper.step(action)
                ticks_completed += 1
                if not torch.isfinite(obs['policy']).all():
                    raise FloatingPointError('Nonfinite evaluation observation')
                if old_target is not None:
                    errors=(env._sm_joint_targets-old_target).abs().amax(1)
                    maximum=float(errors.max())
                    mapping['max_final_target_error_rad']=max(mapping['max_final_target_error_rad'],maximum)
                    phase=before[:,770:775].argmax(1)
                    for i in range(5):
                        mask=phase==i
                        mapping['phase_samples'][str(i)]+=int(mask.sum())
                        if mask.any():
                            mapping['phase_max_error'][str(i)]=max(mapping['phase_max_error'][str(i)],float(errors[mask].max()))
                    mapping['checked_native_ticks']+=1
                    if maximum > mapping['tolerance_rad']:
                        write_json(args.output/'command_consistency.json',mapping)
                        raise RuntimeError('Independent old/new final target mismatch')
                    reset_ids = torch.where(done.bool())[0]
                    if reset_ids.numel():
                        reference.reset(reset_ids)
                for record in env.rally_records[cursor:]:
                    lane=int(record['env_id'])
                    if lane not in first:
                        measured=score_terminal_episode(record, scoring_protocol)
                        if measured['episode_index'] != 0:
                            raise RuntimeError('Missing first episode; refusing later episode substitution')
                        first[lane]=measured
                        with (args.output/'episodes.jsonl').open('a') as stream:
                            stream.write(json.dumps(record)+'\n')
                cursor=len(env.rally_records)
                if (tick+1)%100==0:
                    progress=dict(control_ticks=tick+1,first_episodes=len(first),elapsed_s=time.perf_counter()-start,
                                  first_episode_numerical_audit_count=sum(r['numerical_failures'] for r in first.values()),
                                  active_episode_sweep_unresolved=int(env.sweep_unresolved.sum()),
                                  active_episode_sweep_misses=int(env.sweep_misses.sum()))
                    write_json(args.output/'progress.json',progress)
                    write_json(args.output/'command_consistency.json',mapping)
                    print('SUPERMIMIC_EVAL_PROGRESS='+json.dumps(progress),flush=True)
        if env.physical_transition_archive is not None:
            env.physical_transition_archive.flush()
        if args.shutdown_diagnostic_ticks is not None:
            write_json(args.output/'command_consistency.json', mapping)
            write_json(args.output/'shutdown_diagnostic.json', dict(
                control_ticks=ticks_completed, num_envs=128,
                requested_ticks=args.shutdown_diagnostic_ticks,
                first_episodes=len(first), elapsed_s=time.perf_counter()-start,
                scope='Shutdown diagnostic only; no task score or formal completion',
                actor_updates=0, critic_updates=0, world_updates=0))
            lifecycle_stage(args.output, 'SHUTDOWN_DIAGNOSTIC_FINISHED', control_ticks=ticks_completed)
            print('SUPERMIMIC_SHUTDOWN_DIAGNOSTIC_FINISHED', flush=True)
            return
        if len(first)!=128:
            raise RuntimeError(f'Only {len(first)}/128 pre-reset records; no live-state backfill permitted')
        rows=[dict(first[i],source_case_index=i,case_id=cases[i].get('case_id',i)) for i in range(128)]
        from quadsmash.rl.supermimic_native_metrics import summarize_native_episodes
        native_summary = summarize_native_episodes(rows)
        n=native_summary['conditional_recovery_denominator']
        recovery=native_summary['conditional_recovery_count']
        metrics=dict(evaluation_scope='TRAIN128 development, not A/B acceptance',
            num_envs=128,control_ticks=800,episode_seconds=8.,
            qualified_airborne_string_contact=n,
            ball_side_complete_except_recovery=sum(r['ball_side_complete_except_recovery'] for r in rows),
            recovery_success=sum(r['recovery_success'] for r in rows),
            physical_recovery_success=sum(r['physical_recovery_success'] for r in rows),
            scoring_schema='historical-independent-complete5-v1',
            conditional_recovery_count=recovery,conditional_recovery_denominator=n,
            recovery_given_airborne_string_contact=recovery/n if n else None,
            complete5=sum(r['complete5'] for r in rows),
            body_ground_seen=sum(r['body_ground_seen'] for r in rows),
            no_head_contact_recovery=None,no_independent_nonfoot_contact_recovery=None,
            missing_contact_audit_reason='No independent paired head/all-body contact labels in this environment; never default to false',
            numerical_failures=sum(r['numerical_failures'] for r in rows),
            extra_mpc_interventions=0,extra_teacher_interventions=0,
            historical_base_and_recovery_assistance_preserved=contract['historical_base_and_recovery_assistance_preserved'],
            task_contract=contract,source=init_meta,actor_updated=trained is not None,
            elapsed_s=time.perf_counter()-start,physics=physics,
            evaluated_checkpoint_sha256=hashlib.sha256((args.checkpoint or source_path).read_bytes()).hexdigest())
        metrics['native_episode_metrics'] = native_summary
        metrics['conditional_recovery_scope'] = native_summary['conditional_recovery_scope']
        metrics['native_episode_return_mean'] = native_summary['native_episode_return_mean']
        write_json(args.output/'episodes_summary.json',rows)
        write_json(args.output/'metrics.json',metrics)
        write_json(args.output/'command_consistency.json',mapping)
        write_json(args.output/'completed.json',dict(control_ticks=800,episodes=128,formal_acceptance=False))
        stage='FINISHED'
        lifecycle_stage(args.output, 'SIMULATION_FINISHED', control_ticks=800, episodes=128)
        print('SUPERMIMIC_EVAL_RESULT='+json.dumps(metrics),flush=True)
    except BaseException as error:
        primary_failed = True
        text=traceback.format_exc()
        (args.output/'failure_traceback.txt').write_text(text)
        write_json(args.output/'failure.json',dict(stage=stage,ticks_completed=ticks_completed,
                   attempted_tick=tick+1, stop_requested=stop_request['signal'],
                   type=type(error).__name__,message=str(error)))
        if mapping is not None:
            write_json(args.output/'command_consistency.json',mapping)
        print(text,file=sys.stderr,flush=True)
        raise
    finally:
        close_resource(args.output, 'ENV', env, primary_failed=primary_failed)


if __name__=='__main__':
    main()
