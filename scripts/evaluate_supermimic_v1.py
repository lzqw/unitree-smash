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
    parser.add_argument('--sweep-audit-iterations', type=int, default=0)
    parser.add_argument('--physics-device', default='cuda:0')
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    app = env = None
    tick = -1
    stage = 'BEFORE_APP'
    startup = dict(argv=sys.argv, cwd=os.getcwd(), script=str(Path(__file__).resolve()),
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                   dirty_diff=subprocess.check_output(['git','diff','--no-ext-diff'],cwd=ROOT,text=True),
                   executable=sys.executable, pid=os.getpid())
    write_json(args.output/'startup.json', startup)
    try:
        # Keep caught Python exceptions visible through shutdown.
        app = AppLauncher(args, fast_shutdown=False).app
        stage = 'AFTER_APP'
        import torch
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaacsim.core.simulation_manager import SimulationManager
        from quadsmash.rl.supermimic_v1 import load_ppo_initialized_actor
        from quadsmash.rl.supermimic_task_contract import (
            resolve_task_contract, make_task_cfg, require_resume_contract, summarize_episode,
            zero_panel_reset_noise)
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
        if trained is not None:
            require_resume_contract(trained, contract)
        write_json(args.output/'task_contract.json', contract)
        cfg = make_task_cfg(128, contract, protocol, receipt)
        cfg.seed, cfg.sim.device = args.seed, args.physics_device
        cfg.sweep_audit_iterations = args.sweep_audit_iterations
        if args.sweep_audit_iterations < 0:
            raise ValueError('Sweep audit count cannot be negative')
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
            for tick in range(800):
                before = obs['policy'].clone()
                old_target = None
                if reference is not None:
                    old_final = reference.action(obs, env._sm_residual_default, env._sm_residual_limits)
                    old_target = full_range_targets(old_final, env._sm_residual_limits).clone()
                action = actor(before, sample=False)[0].clamp(-1.,1.)
                if not torch.isfinite(action).all():
                    raise FloatingPointError('Nonfinite evaluation action')
                obs, _, done, _ = wrapper.step(action)
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
                        measured=summarize_episode(record)
                        if measured['episode_index'] != 0:
                            raise RuntimeError('Missing first episode; refusing later episode substitution')
                        first[lane]=measured
                        with (args.output/'episodes.jsonl').open('a') as stream:
                            stream.write(json.dumps(record)+'\n')
                cursor=len(env.rally_records)
                if (tick+1)%100==0:
                    progress=dict(control_ticks=tick+1,first_episodes=len(first),elapsed_s=time.perf_counter()-start)
                    write_json(args.output/'progress.json',progress)
                    write_json(args.output/'command_consistency.json',mapping)
                    print('SUPERMIMIC_EVAL_PROGRESS='+json.dumps(progress),flush=True)
        if len(first)!=128:
            raise RuntimeError(f'Only {len(first)}/128 pre-reset records; no live-state backfill permitted')
        rows=[dict(first[i],source_case_index=i,case_id=cases[i].get('case_id',i)) for i in range(128)]
        contact=[r['qualified_airborne_string_contact'] for r in rows]
        n=sum(contact)
        recovery=sum(r['recovery_success'] for r,c in zip(rows,contact) if c)
        metrics=dict(evaluation_scope='TRAIN128 development, not A/B acceptance',
            num_envs=128,control_ticks=800,episode_seconds=8.,
            qualified_airborne_string_contact=n,
            ball_side_complete_except_recovery=sum(r['ball_side_complete_except_recovery'] for r in rows),
            recovery_success=sum(r['recovery_success'] for r in rows),
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
        write_json(args.output/'episodes_summary.json',rows)
        write_json(args.output/'metrics.json',metrics)
        write_json(args.output/'command_consistency.json',mapping)
        write_json(args.output/'completed.json',dict(control_ticks=800,episodes=128,formal_acceptance=False))
        stage='FINISHED'
        print('SUPERMIMIC_EVAL_RESULT='+json.dumps(metrics),flush=True)
    except BaseException as error:
        text=traceback.format_exc()
        (args.output/'failure_traceback.txt').write_text(text)
        write_json(args.output/'failure.json',dict(stage=stage,ticks_completed=tick+1,
                   type=type(error).__name__,message=str(error)))
        print(text,file=sys.stderr,flush=True)
        raise
    finally:
        if env is not None:
            try:
                env.close()
            except BaseException:
                (args.output/'env_cleanup_failure.txt').write_text(traceback.format_exc())
        if app is not None:
            try:
                app.close()
            except BaseException:
                (args.output/'app_cleanup_failure.txt').write_text(traceback.format_exc())


if __name__=='__main__':
    main()
