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
import zipfile

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
    parser.add_argument('--init-policy-recovery-gate',choices=('legacy','causal'),default='legacy',
                        help='Explicit zero-update execution contract; causal matches physical training')
    parser.add_argument('--diagnostic-grounded-miss', action='store_true',
                        help='Frozen TRAIN128 comparison: also activate Recovery after a confirmed grounded miss')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--relaxed-scoring', action='store_true')
    parser.add_argument('--approach-ball-only', action='store_true',
                        help='Score the complete measured ball outcome independently of robot recovery')
    parser.add_argument('--early-arm-seconds', type=float, default=None,
                        help='Zero-update early-arm execution window for the Approach-only A0 panel')
    parser.add_argument('--relaxed-survival-diagnostic', action='store_true',
                        help='Continue successful lanes for post-success survival and uncensored strict scoring')
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
    panels = parser.add_mutually_exclusive_group()
    panels.add_argument('--research-fixed128', action='store_true',
                        help='Score the predeclared research cases held out from native training')
    panels.add_argument('--independent-panel-batch', type=int, choices=range(4),
                        help='Evaluate one stage-end batch of the declared independent 512 cases')
    parser.add_argument('--native-action-probe-spec',type=Path,default=None)
    parser.add_argument('--native-action-probe-branch',choices=('nominal_a','nominal_b','minus','plus'),default=None)
    args = parser.parse_args()
    if args.early_arm_seconds is not None and not args.approach_ball_only:
        parser.error('Early-arm override requires the ball-only A0 evaluation protocol')
    if args.approach_ball_only and (args.relaxed_scoring or args.relaxed_survival_diagnostic):
        parser.error('Choose a single evaluation termination protocol')
    if (args.independent_panel_batch is not None or args.research_fixed128) and (args.native_action_probe_spec or args.record_physical_transitions
            or args.shutdown_diagnostic_ticks is not None):
        parser.error('Independent panel evaluation requires a complete score-only rollout')
    if args.diagnostic_grounded_miss and (not args.checkpoint or args.independent_panel_batch is not None
            or args.research_fixed128 or args.native_action_probe_spec or args.record_physical_transitions
            or args.shutdown_diagnostic_ticks is not None):
        parser.error('The grounded-miss diagnostic uses a learned checkpoint and a full TRAIN128 score')
    if bool(args.native_action_probe_spec)!=bool(args.native_action_probe_branch):
        parser.error('Probe spec and branch are required together')
    if args.native_action_probe_spec:
        if not args.physics_device.startswith('cuda') or not args.headless:
            parser.error('Paired native probes require headless GPU PhysX on the original128 lanes')
        if args.shutdown_diagnostic_ticks is not None or args.record_physical_transitions:
            parser.error('Probe is a separate diagnostic, not evaluation/archive mode')
        from quadsmash.rl.supermimic_native_pair_capture_v1 import validate_spec
        probe_spec=validate_spec(json.loads(args.native_action_probe_spec.read_text()))
        args.shutdown_diagnostic_ticks=probe_spec['tick']+1

    if args.shutdown_diagnostic_ticks is not None and not 1 <= args.shutdown_diagnostic_ticks < 800:
        parser.error('shutdown diagnostic ticks must be in [1,799]')
    if args.checkpoint and args.init_policy_recovery_gate!='legacy':
        parser.error('Learned checkpoints must use their recorded execution contract')
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
        from quadsmash.rl.supermimic_terminal_scoring import (
            score_terminal_episode, relaxed_scoring_protocol, approach_ball_scoring_protocol, STRICT_SCORING)
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
            trained = torch.load(args.checkpoint, map_location='cpu', weights_only=False,
                                 mmap=zipfile.is_zipfile(args.checkpoint))
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
            source, source_path, execution_root=execution_root,
            precision_receipt=args.precision_receipt, root=execution_root)
        scoring_protocol = dict(protocol, _root=str(ROOT))
        if trained is not None:
            stored = trained.get('metadata', {}).get('task_contract', {})
            if 'relaxed_scoring' in stored:
                contract = dict(contract, relaxed_scoring=relaxed_scoring_protocol())
            if 'approach_ball_scoring' in stored:
                contract = dict(contract, approach_ball_scoring=approach_ball_scoring_protocol())
                if args.early_arm_seconds is not None:
                    raise ValueError('A trained ball-only checkpoint retains its recorded gate')
            require_resume_contract(trained, contract)
        if args.relaxed_scoring:
            contract = dict(contract, relaxed_scoring=relaxed_scoring_protocol())
        if args.approach_ball_only:
            contract.pop('relaxed_scoring', None)
            contract['approach_ball_scoring'] = approach_ball_scoring_protocol()
        scoring_protocol['approach_ball_scoring'] = contract.get('approach_ball_scoring')
        scoring_protocol['relaxed_scoring'] = contract.get('relaxed_scoring')
        if args.relaxed_survival_diagnostic and scoring_protocol['relaxed_scoring'] is None:
            raise ValueError('Post-success survival requires relaxed scoring')
        capture_contract = contract
        if args.independent_panel_batch is not None:
            from quadsmash.rl.supermimic_independent_panel import panel_batch
            cases, panel_identity = panel_batch(args.independent_panel_batch, contract['protocol_sha256'], args.seed)
            capture_contract = dict(contract, independent_evaluation=panel_identity)
        if args.research_fixed128:
            from quadsmash.rl.supermimic_independent_panel import fixed128_cases
            cases, panel_identity = fixed128_cases(contract['protocol_sha256'], args.seed)
            capture_contract = dict(contract, research_evaluation=panel_identity)
        if args.native_action_probe_spec:
            from scripts.declare_supermimic_contact_training_cases import resolve_probe_cases
            spec = json.loads(args.native_action_probe_spec.read_text())
            cases, training_digest = resolve_probe_cases(spec, cases)
            if training_digest:
                capture_contract = dict(contract, contact_training_panel_sha256=training_digest,
                                        contact_training_only=True)
        write_json(args.output/'task_contract.json', capture_contract)
        cfg = make_task_cfg(128, contract, protocol, receipt)
        cfg.seed, cfg.sim.device = args.seed, args.physics_device
        validate_sweep_contract(cfg, contract, args.sweep_audit_iterations)
        noise = zero_panel_reset_noise(len(cases), args.seed)
        stage = 'BEFORE_ENV'
        env = EvaluationEnv(cfg, cases=cases, reset_noise=noise,
            residual_policy_checkpoint=source, g07_execution_root=execution_root,
            contact_flight_recovery=contract['contact_flight_recovery'],
            relaxed_protocol=contract.get('relaxed_scoring'),
            approach_ball_protocol=contract.get('approach_ball_scoring'),
            relaxed_survival_diagnostic=args.relaxed_survival_diagnostic)
        stage = 'AFTER_ENV'
        wrapper = RslRlVecEnvWrapper(env, clip_actions=None)
        obs, _ = wrapper.reset()
        stage = 'AFTER_RESET'
        write_json(args.output/'initial_states.json', env.initial_states)
        actor, init_meta = load_ppo_initialized_actor(source_path, env.device)
        if trained is not None:
            actor.load_state_dict(trained['actor'], strict=True)
        actor.eval()
        from quadsmash.rl.supermimic_policy_execution_v1 import resolve_contract,policy_forward,recovery_gate,contract as execution_contract,POSTHIT,POSTHIT_OR_GROUNDED_MISS
        from quadsmash.rl.supermimic_physical_loop_loading_v1 import tensor_digest
        actor_state_before=tensor_digest(actor.state_dict().items())
        policy_execution,execution_provenance=resolve_contract(trained)
        if trained is None and args.init_policy_recovery_gate=='causal':
            policy_execution=execution_contract(POSTHIT);execution_provenance='explicit frozen initialization under training causal gate'
        if args.early_arm_seconds is not None:
            policy_execution = execution_contract(POSTHIT, arm_preparation_seconds=args.early_arm_seconds)
            execution_provenance = 'explicit early-arm zero-update baseline'
        if policy_execution.get('arm_preparation_seconds') is not None:
            actor.configure_approach_gate(policy_execution['arm_preparation_seconds'])
        if args.diagnostic_grounded_miss:
            if policy_execution != execution_contract(POSTHIT):
                raise ValueError('Grounded-miss comparison requires a post-hit checkpoint')
            policy_execution = execution_contract(POSTHIT_OR_GROUNDED_MISS)
            execution_provenance = 'frozen TRAIN128 grounded-miss gate comparison, same checkpoint weights'
        write_json(args.output/'policy_execution.json',dict(contract=policy_execution,provenance=execution_provenance))
        if args.record_physical_transitions:
            from quadsmash.rl.supermimic_physical_capture_v1 import PhysicalTransitionArchive
            if env._sm_residual_controller is None:
                raise ValueError('Physical capture requires the initialized residual execution contract')
            env.physical_transition_archive = PhysicalTransitionArchive(
                args.output/'physical_transitions', metadata=dict(
                    source_checkpoint=str(source_path), source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                    task_contract=contract,policy_execution=policy_execution,actor_state_sha256=actor_state_before,policy_id='frozen-evaluation',body_names=list(env.robot.body_names),
                    control_dt_s=float(env.step_dt), num_envs=env.num_envs,
                    seed=args.seed, runtime_commit=subprocess.check_output(
                        ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
            env.physical_transition_archive.attach_task_sequence(env)
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
        from quadsmash.rl.supermimic_actuator_identification import actuator_contract
        write_json(args.output/'actuator_contract.json', actuator_contract(env, contract))
        first, cursor, numerical = {}, 0, 0
        head_counts = torch.zeros(2, dtype=torch.long, device=env.device)
        if args.diagnostic_grounded_miss:
            gate_first_tick = torch.full((128,), -1, device=env.device, dtype=torch.long)
            gate_ticks = torch.zeros_like(gate_first_tick)
            first_active = torch.ones_like(gate_ticks, dtype=torch.bool)
        mapping = dict(max_final_target_error_rad=0., tolerance_rad=2e-6,
                       phase_samples={str(i):0 for i in range(5)}, phase_max_error={str(i):0. for i in range(5)},
                       checked_native_ticks=0, scope='Independent old controller versus new actor+executor on the SAME live observations; not a separate old physics rollout')
        if policy_execution['recovery_gate']==POSTHIT:
            mapping['scope']='Independent original executor and new executor with identical causal-gated requests; not legacy policy equivalence'
        probe_capture=None
        if args.native_action_probe_spec:
            from quadsmash.rl.supermimic_native_pair_capture_v1 import NativePairCapture
            if policy_execution['recovery_gate']!=POSTHIT:
                raise ValueError('Native training probe requires the causal execution contract')
            probe_capture=NativePairCapture(spec,args.native_action_probe_branch)
        start = time.perf_counter()
        step_wall_seconds = 0.
        phase_timing = None
        if scoring_protocol['approach_ball_scoring'] is not None:
            from quadsmash.rl.supermimic_native_phase_timing import NativePhaseTiming
            phase_timing = NativePhaseTiming(env)
        stage = 'EVALUATING'
        with torch.inference_mode():
            for tick in range(args.shutdown_diagnostic_ticks or 800):
                if stop_request['signal'] is not None:
                    raise InterruptedError(f"Evaluation stopped at control boundary: {stop_request['signal']}")
                before = obs['policy'] if reference is None and probe_capture is None else obs['policy'].clone()
                old_target = None
                action = policy_forward(actor,before,env.rally,mode=policy_execution['recovery_gate'],sample=False)[0].clamp(-1.,1.)
                if env.approach_ball_termination is not None:
                    env.record_approach_gate(actor.approach_action_gate(before))
                if probe_capture is not None:action=probe_capture.before(tick,env,before,action)
                if policy_execution['recovery_gate'] == POSTHIT_OR_GROUNDED_MISS:
                    active_gate = recovery_gate(env.rally,before,mode=POSTHIT_OR_GROUNDED_MISS)
                    head_counts[0].add_(active_gate.sum())
                    if args.diagnostic_grounded_miss:
                        extra = first_active & active_gate & ~env.rally.had_contact
                        gate_first_tick.masked_fill_(extra & (gate_first_tick < 0), tick)
                        gate_ticks.add_(extra)
                else:
                    head_counts[0].add_((env.rally.had_contact & env.rally.released).sum())
                head_counts[1].add_((actor.approach_gate(before) > .25).sum())
                if reference is not None:
                    if policy_execution['recovery_gate']==POSTHIT:
                        old_final=reference.apply_residual(obs,action,env._sm_residual_default,env._sm_residual_limits)
                    else:
                        old_final = reference.action(obs, env._sm_residual_default, env._sm_residual_limits)
                    old_target = full_range_targets(old_final, env._sm_residual_limits).clone()
                if not torch.isfinite(action).all():
                    raise FloatingPointError('Nonfinite evaluation action')
                step_start = time.perf_counter()
                obs, _, done, _ = (wrapper.step(action) if phase_timing is None else
                                   phase_timing.step(wrapper.step, action, before, tick))
                if probe_capture is not None:probe_capture.after(tick,env,done)
                ticks_completed += 1
                if not torch.isfinite(obs['policy']).all():
                    raise FloatingPointError('Nonfinite evaluation observation')
                step_wall_seconds += time.perf_counter() - step_start
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
                        if args.diagnostic_grounded_miss:
                            first_active[lane] = False
                        with (args.output/'episodes.jsonl').open('a') as stream:
                            stream.write(json.dumps(record)+'\n')
                    else:
                        with (args.output/'later_episodes.jsonl').open('a') as stream:
                            stream.write(json.dumps(record)+'\n')
                cursor=len(env.rally_records)
                if (tick+1)%100==0:
                    progress=dict(control_ticks=tick+1,first_episodes=len(first),elapsed_s=time.perf_counter()-start,
                                  first_episode_numerical_audit_count=sum(r['numerical_failures'] for r in first.values()),
                                  active_episode_sweep_unresolved=int(env.sweep_unresolved.sum()),
                                  active_episode_sweep_misses=int(env.sweep_misses.sum()))
                    returns=[r['native_episode_return'] for r in first.values()]
                    discounted=[r['native_episode_discounted_return'] for r in first.values()]
                    progress.update(native_episode_return_mean=sum(returns)/len(returns) if returns else None,
                        native_episode_discounted_return_mean=sum(discounted)/len(discounted) if discounted else None,
                        native_return_gamma=env._sm_native_returns.gamma)
                    write_json(args.output/'progress.json',progress)
                    write_json(args.output/'command_consistency.json',mapping)
                    print('SUPERMIMIC_EVAL_PROGRESS='+json.dumps(progress),flush=True)
                if scoring_protocol['approach_ball_scoring'] is not None and len(first) == 128:
                    break
        if env.physical_transition_archive is not None:
            env.physical_transition_archive.flush()
        if probe_capture is not None:
            probe_capture.save(args.output,actor,capture_contract,policy_execution,physics,args.seed)
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
        if args.research_fixed128 or args.independent_panel_batch is not None:
            if any(first[i]['case_id'] != cases[i]['case_id'] for i in range(128)):
                raise ValueError('Terminal case differs from the declared lane assignment')
        rows=[dict(first[i],source_case_index=i,case_id=cases[i].get('case_id',i)) for i in range(128)]
        if scoring_protocol['approach_ball_scoring'] is not None:
            contacts = [event for event in env.contact_records if event['episode_index'] == 0]
            write_json(args.output/'native_contact_motion.json', contacts)
            write_json(args.output/'native_phase_timing.json', phase_timing.report())
        from quadsmash.rl.supermimic_native_metrics import summarize_native_episodes
        native_summary = summarize_native_episodes(rows)
        n=native_summary['qualified_airborne_string_contact']
        recovery=native_summary['conditional_recovery_count']
        recovery_observed=native_summary['conditional_recovery_denominator']
        with (args.checkpoint or source_path).open('rb') as checkpoint_stream:
            digest = hashlib.sha256()
            for chunk in iter(lambda: checkpoint_stream.read(8 << 20), b''):
                digest.update(chunk)
            checkpoint_sha256 = digest.hexdigest()
        metrics=dict(policy_execution=policy_execution,policy_execution_provenance=execution_provenance,evaluation_scope='TRAIN128 development, not A/B acceptance',
            num_envs=128,control_ticks=ticks_completed,episode_seconds=8.,
            qualified_airborne_string_contact=n,
            ball_side_complete_except_recovery=sum(r['ball_side_complete_except_recovery'] for r in rows),
            recovery_success=recovery if recovery_observed or scoring_protocol['approach_ball_scoring'] is None else None,
            physical_recovery_success=sum(r['physical_recovery_success'] is True for r in rows)
                if any(r['physical_recovery_success'] is not None for r in rows) else None,
            scoring_schema=rows[0].get('scoring_schema', STRICT_SCORING),
            conditional_recovery_count=recovery,conditional_recovery_denominator=recovery_observed,
            recovery_given_airborne_string_contact=native_summary['recovery_given_airborne_string_contact'],
            complete5=native_summary['complete5'],
            body_ground_seen=sum(r['body_ground_seen'] for r in rows),
            no_head_contact_recovery=None,no_independent_nonfoot_contact_recovery=None,
            missing_contact_audit_reason='No independent paired head/all-body contact labels in this environment; never default to false',
            numerical_failures=sum(r['numerical_failures'] for r in rows),
            extra_mpc_interventions=0,extra_teacher_interventions=0,
            historical_base_and_recovery_assistance_preserved=contract['historical_base_and_recovery_assistance_preserved'],
            task_contract=contract,source=init_meta,actor_updated=trained is not None,
            elapsed_s=time.perf_counter()-start,physics=physics,
            native_step_and_observation_check_seconds=step_wall_seconds,
            evaluated_checkpoint_sha256=checkpoint_sha256)
        if args.diagnostic_grounded_miss:
            metrics.update(evaluation_scope='frozen TRAIN128 grounded-miss gate comparison',
                           checkpoint_policy_execution=execution_contract(POSTHIT),
                           first_episode_extra_gate_ticks=gate_ticks.tolist(),
                           first_episode_extra_gate_onset=gate_first_tick.tolist())
        if args.independent_panel_batch is not None:
            metrics.update(task_contract=capture_contract, evaluation_scope='independent512 stage-end research evaluation')
        if args.research_fixed128:
            metrics.update(task_contract=capture_contract, evaluation_scope='fixed128 held-out research development')
        actor_state_after=tensor_digest(actor.state_dict().items())
        if actor_state_after!=actor_state_before:raise RuntimeError('Frozen evaluation mutated actor weights')
        head_active_lane_steps = dict(zip(("recovery", "approach"), head_counts.tolist()))
        metrics.update(actor_state_sha256=actor_state_after,actor_parameters_unchanged=True,
            optimizer_updates=dict(actor=0,world=0,critic=0),head_active_lane_steps=head_active_lane_steps)
        metrics['native_episode_metrics'] = native_summary
        if scoring_protocol['relaxed_scoring'] is not None:
            strict = [r['strict_observed'] for r in rows]
            metrics['strict_observed'] = {
                name: dict(success=sum(r[name] is True for r in strict),
                           observed=sum(r[name] is not None for r in strict))
                for name in ('complete5', 'recovery_success')
            }
            metrics['terminated_success_count'] = sum(r['terminated_success'] for r in rows)
            if args.relaxed_survival_diagnostic:
                survival = [r['relaxed_survival_one_second'] for r in rows]
                observed = sum(r['observed'] for r in survival)
                survived = sum(r['survived'] is True for r in survival)
                metrics['relaxed_survival_one_second'] = dict(
                    reached=sum(r['success_reached'] for r in survival),
                    observed=observed, survived=survived, rate=survived / observed if observed else None)
        metrics['conditional_recovery_scope'] = native_summary['conditional_recovery_scope']
        metrics['native_episode_return_mean'] = native_summary['native_episode_return_mean']
        metrics['native_episode_discounted_return_mean'] = sum(r['native_episode_discounted_return'] for r in rows)/len(rows)
        metrics['native_return_gamma'] = env._sm_native_returns.gamma
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
    # Isaac Sim 5.1's PhysX Python bindings can run a module-level destructor
    # after Kit has already released the framework.  That destructor calls
    # PyThreadState_Get without the GIL and aborts with
    # ``PyThreadState_Get ... tstate is NULL``.  The scoped lifecycle above has
    # already flushed artifacts and closed native resources, so bypass Python's
    # second interpreter-finalization pass on the clean path.
    os.environ['QUADSMASH_SKIP_KIT_CLOSE'] = '1'
    try:
        main()
    except BaseException:
        # The failure artifact and traceback are written by main().  Avoid
        # interpreter finalization too, since it has the same PhysX/GIL bug.
        os._exit(1)
    else:
        os._exit(0)
