#!/usr/bin/env python3
"""PPO-initialized, seconds-based BPTT/GAE training. Never launches an evaluator.

Code review only does not launch this script. Native tests and a working
initialization contract remain necessary before accepting training results.
"""
from __future__ import annotations
import argparse
from collections import deque
import hashlib
import json
import inspect
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--init-policy-checkpoint', type=Path, required=True)
    parser.add_argument('--timebase-config', type=Path, default=ROOT/'configs/supermimic_task_horizon_v2.json')
    parser.add_argument('--iterations', type=int, default=40000)
    parser.add_argument('--num-envs', type=int, default=512)
    parser.add_argument('--seed', type=int, default=20261113)
    parser.add_argument('--world-batch', type=int, default=2048)
    parser.add_argument('--world-updates', type=int, default=3)
    parser.add_argument('--imagination-batch', type=int, default=2048)
    parser.add_argument('--actor-microbatch', type=int, default=128)
    parser.add_argument('--actor-lr', type=float, default=5e-5)
    parser.add_argument('--world-lr', type=float, default=2e-4)
    parser.add_argument('--critic-lr', type=float, default=2e-4)
    parser.add_argument('--policy-anchor-coeff', type=float, default=0.05)
    parser.add_argument('--replay-capacity', type=int, default=250000)
    parser.add_argument('--checkpoint-every', type=int, default=500)
    parser.add_argument('--persist-replay', action='store_true')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--imagination-seconds', type=float)
    parser.add_argument('--world-window-seconds', type=float)
    parser.add_argument('--auto-stop-convergence', action='store_true')
    parser.add_argument('--convergence-min-iterations', type=int, default=2000)
    parser.add_argument('--convergence-window', type=int, default=200)
    parser.add_argument('--convergence-patience', type=int, default=3)
    parser.add_argument('--convergence-relative-tolerance', type=float, default=0.02)
    parser.add_argument('--convergence-return-absolute-tolerance', type=float, default=0.10)
    parser.add_argument('--convergence-max-grad-norm', type=float, default=0.50)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if min(args.iterations, args.num_envs, args.world_batch, args.world_updates,
           args.imagination_batch, args.checkpoint_every) <= 0:
        parser.error('Counts must be positive')
    if args.convergence_min_iterations < 1 or args.convergence_window < 1 or args.convergence_patience < 1:
        parser.error('Convergence counts must be positive')
    if args.convergence_relative_tolerance <= 0 or args.convergence_return_absolute_tolerance <= 0 or args.convergence_max_grad_norm <= 0:
        parser.error('Convergence tolerances must be positive')
    args.output = args.output.expanduser().resolve()
    if args.output.exists():
        raise FileExistsError(f'Refusing to reuse existing smoke/training output: {args.output}')
    args.output.mkdir(parents=True, exist_ok=False)
    stage_path = args.output / 'startup_stages.jsonl'
    startup_path = args.output / 'startup.json'
    current_stage = 'MAIN_ENTER'
    last_tick = -1
    app = None
    env = None

    def stage(name, **fields):
        nonlocal current_stage
        current_stage = name
        record = {'stage': name, 'time_unix': time.time(), 'pid': os.getpid(), **fields}
        with stage_path.open('a') as stream:
            stream.write(json.dumps(record, default=str) + '\n')
            stream.flush()

    def git_value(*argv):
        try:
            return subprocess.check_output(['git', *argv], cwd=ROOT, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    startup = {
        'argv': list(sys.argv),
        'cwd': os.getcwd(),
        'script': str(Path(__file__).resolve()),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'git_commit': git_value('rev-parse', 'HEAD'),
        'git_dirty_diff': git_value('diff', '--no-ext-diff'),
        'sys_executable': sys.executable,
        'output': str(args.output),
        'pid': os.getpid(),
        'started_unix': time.time(),
    }
    startup_path.write_text(json.dumps(startup, indent=2, default=str) + '\n')
    stage('MAIN_ENTER', output=str(args.output))

    def record_failure(error):
        payload = {
            'status': 'failed', 'stage': current_stage, 'exception_type': type(error).__name__,
            'message': str(error), 'pid': os.getpid(), 'last_tick': last_tick,
            'output': str(args.output), 'time_unix': time.time(),
        }
        (args.output / 'failure.json').write_text(json.dumps(payload, indent=2) + '\n')
        (args.output / 'failure_traceback.txt').write_text(traceback.format_exc())
        stage('FAILURE_RECORDED', failure=payload)

    def close_app():
        if app is None:
            return
        try:
            signature = inspect.signature(app.close)
            if 'fast_shutdown' in signature.parameters:
                app.close(fast_shutdown=False)
            else:
                app.close()
        except BaseException as cleanup_error:
            cleanup_record = {
                'status': 'cleanup_error', 'exception_type': type(cleanup_error).__name__,
                'message': str(cleanup_error), 'time_unix': time.time(),
            }
            (args.output / 'cleanup_failure.json').write_text(json.dumps(cleanup_record, indent=2) + '\n')
            (args.output / 'cleanup_failure_traceback.txt').write_text(traceback.format_exc())

    try:
        stage('BEFORE_APP')
        app = AppLauncher(args).app
        stage('AFTER_APP', app_type=type(app).__name__)
        import torch
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from quadsmash.rl.supermimic_v1 import (load_ppo_initialized_actor,
                                               SuperMimicWorldModel, SuperMimicCritic)
        from quadsmash.rl.supermimic_timebase_v2 import Timebase, full_range_targets
        from quadsmash.rl.supermimic_sequence_replay_v2 import SequenceReplay
        from quadsmash.rl.supermimic_horizon_learner_v2 import HorizonTrainer
        from quadsmash.tasks.stand.supermimic_env_v1 import SuperMimicEnvV1, make_supermimic_cfg
        from quadsmash.tasks.stand.observation_contract import observe_native
        from quadsmash.control.full_racket_residual_v4 import FullRacketResidualControllerV4
        stage('IMPORTS_OK')

        torch.manual_seed(args.seed)
        stage('OUTPUT_READY', output=str(args.output))
        temporal = json.loads(args.timebase_config.read_text())
        if args.imagination_seconds is not None:
            temporal['imagination_s'] = args.imagination_seconds
        if args.world_window_seconds is not None:
            temporal['world_window_s'] = args.world_window_seconds
        timebase = Timebase(**temporal)
        source = torch.load(args.init_policy_checkpoint, map_location='cpu', weights_only=False)
        if source.get('schema') != 'full-racket-residual-ppo-v4':
            raise ValueError('This first v2 entry requires the audited full-racket residual contract')
        stage('SOURCE_LOADED', source=str(args.init_policy_checkpoint.resolve()),
              source_sha256=hashlib.sha256(args.init_policy_checkpoint.read_bytes()).hexdigest())
        cfg = make_supermimic_cfg(args.num_envs)
        cfg.seed, cfg.sim.device = args.seed, args.device or 'cuda:0'
        cfg.sweep_audit_iterations = 0
        if abs(cfg.sim.dt * cfg.decimation - timebase.control_dt_s) > 1e-10:
            raise ValueError('Timebase does not match the real environment; physics is not changed automatically')
        if abs(cfg.episode_length_s - timebase.episode_s) > 1e-10:
            raise ValueError('Episode contract mismatch')
        protocol = json.loads((ROOT/'configs/g07_full_racket_eval_v4_r3.json').read_text())
        cases0 = json.loads((ROOT/protocol['training_suite']['path']).read_text())['cases']
        cases = [cases0[i % len(cases0)] for i in range(args.num_envs)]
        stage('BEFORE_ENV', num_envs=args.num_envs, device=str(cfg.sim.device))
        env = SuperMimicEnvV1(cfg, cases=cases, residual_policy_checkpoint=source)
        stage('AFTER_ENV')
        wrapper = RslRlVecEnvWrapper(env, clip_actions=None)
        stage('BEFORE_RESET')
        obs, _ = wrapper.reset()
        stage('AFTER_RESET', obs_shape=list(obs['policy'].shape))
        obs = obs['policy']
        actor, init_meta = load_ppo_initialized_actor(args.init_policy_checkpoint, env.device)
        reference_actor, reference_meta = load_ppo_initialized_actor(args.init_policy_checkpoint, env.device)
        if reference_meta['sha256'] != init_meta['sha256'] or reference_meta['actor_kind'] != init_meta['actor_kind']:
            raise RuntimeError('Reference actor does not match initialized actor source')
        replay = SequenceReplay(args.replay_capacity, args.num_envs, obs.shape[-1], 18, env.device)
        if replay.slots < timebase.world_steps:
            raise ValueError('Replay too short for configured physical-time window')
        trainer = HorizonTrainer(actor, SuperMimicWorldModel(obs.shape[-1], 18, structured_history=True).to(env.device),
                                 SuperMimicCritic(obs.shape[-1]).to(env.device), replay, timebase,
                                 actor_lr=args.actor_lr, world_lr=args.world_lr, critic_lr=args.critic_lr,
                                 reference_actor=reference_actor,
                                 policy_anchor_coeff=args.policy_anchor_coeff,
                                 actor_microbatch=args.actor_microbatch, actor_kind=init_meta['actor_kind'],
                                 observation_clock_index=775 if obs.shape[-1] == 794 else None)
        stage('TRAINER_READY', obs_dim=int(obs.shape[-1]), action_dim=18)
        metadata = {'init_policy_checkpoint': init_meta, 'old_checkpoint': None, 'mpc': None, 'teacher': None,
                    'bc': None, 'num_envs': args.num_envs, 'seed': args.seed,
                    'timebase': timebase.manifest(), 'init_source_sha256': hashlib.sha256(args.init_policy_checkpoint.read_bytes()).hexdigest(),
                    'world_representation': 'structured_native_history_v1_next_frame_plus_tail',
                    'policy_anchor': {'source': init_meta, 'coefficient': args.policy_anchor_coeff},
                    'learning_rates': {'actor': args.actor_lr, 'world': args.world_lr, 'critic': args.critic_lr},
                    'formal_evaluation_started': False}
        convergence_history = deque(maxlen=max(2 * args.convergence_window, 2))
        convergence_stable_windows = 0
        convergence_stop_reason = None
        convergence_config = {
            'enabled': args.auto_stop_convergence,
            'min_iterations': args.convergence_min_iterations,
            'window': args.convergence_window,
            'patience': args.convergence_patience,
            'relative_tolerance': args.convergence_relative_tolerance,
            'return_absolute_tolerance': args.convergence_return_absolute_tolerance,
            'max_actor_grad_norm': args.convergence_max_grad_norm,
            'note': 'Training convergence proxy only; task success still requires frozen native evaluation.'
        }
        (args.output/'convergence_config.json').write_text(json.dumps(convergence_config, indent=2)+'\n')
        total_native = total_iterations = 0
        record_cursor = len(env.rally_records)
        native_event_totals = {
            'episodes': 0, 'qualified_airborne_string_contact': 0,
            'ball_side_complete_except_recovery': 0, 'recovery_success': 0,
            'complete5': 0, 'frame_hit': 0, 'nonfoot_contact': 0,
            'body_ground_seen': 0, 'rally_fault': 0,
        }
        if args.resume:
            prior = trainer.load(args.resume)
            if prior['init_source_sha256'] != metadata['init_source_sha256']:
                raise ValueError('Resume source checkpoint changed')
            total_native = int(prior.get('native_steps', 0))
            total_iterations = int(prior.get('iterations_completed', 0))
        # Independent old controller, not another call to the same new actor.
        reference = FullRacketResidualControllerV4(source, args.num_envs, env.device)
        reference.reset()
        reference_targets = None
        if not args.resume:
            public = observe_native(env)
            default, limits = env._sm_residual_default, env._sm_residual_limits
            old_final = reference.action(public, default, limits)
            reference_targets = full_range_targets(old_final, limits).clone()
        metadata['zero_update_audit_scope'] = 'first native tick only; full phase trace audit still required'
        (args.output/'training_manifest.json').write_text(json.dumps(metadata, indent=2)+'\n')
        stopping = False
        def stop(*_):
            nonlocal stopping, convergence_stop_reason
            stopping = True
            convergence_stop_reason = 'signal'
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        def sync():
            if torch.device(env.device).type == 'cuda':
                torch.cuda.synchronize(env.device)
        start = time.perf_counter()
        for local_iteration in range(args.iterations):
            if stopping:
                break
            sync(); tick_start = time.perf_counter()
            current = obs.detach().clone()  # clone before simulator-owned buffers mutate
            last_tick = local_iteration
            episode = env.episode_index.detach().clone()
            elapsed = env.episode_length_buf.float().clone() * env.step_dt
            action = trainer.action(current, sample=(local_iteration != 0 or args.resume is not None))
            nxt, reward, done, info = wrapper.step(action)
            new_records = env.rally_records[record_cursor:]
            record_cursor += len(new_records)
            for record in new_records:
                contact = bool(record.get('legal_initial_contact', False)
                               and record.get('string_hit', False)
                               and record.get('airborne_string_hit', False))
                native_event_totals['episodes'] += 1
                native_event_totals['qualified_airborne_string_contact'] += int(contact)
                native_event_totals['ball_side_complete_except_recovery'] += int(
                    contact and record.get('released', False)
                    and record.get('crossed_net', False)
                    and record.get('landed_target', False))
                native_event_totals['recovery_success'] += int(record.get('recovery_success', False))
                native_event_totals['complete5'] += int(record.get('complete5', False))
                native_event_totals['frame_hit'] += int(record.get('frame_hit', False))
                recovery = record.get('contact_flight_recovery') or {}
                # The current episode record exposes body/non-foot landing,
                # not a head-specific collision label. Keep this explicitly
                # named so the training log cannot overclaim head contact.
                native_event_totals['nonfoot_contact'] += int(recovery.get('body_landing', False))
                native_event_totals['body_ground_seen'] += int(record.get('ground_seen', False))
                native_event_totals['rally_fault'] += int(any(
                    bool(record.get(key, False)) for key in (
                        'robot_net_fault', 'body_ball_fault', 'wrong_side_fault',
                        'double_hit_fault', 'racket_invasion_fault', 'body_over_net_fault',
                        'under_net_obstruction_fault', 'bad_return_fault', 'net_blocked_fault')))
            if local_iteration == 0:
                stage('FIRST_STEP_OK', reward_finite=bool(torch.isfinite(reward).all().item()),
                      done_count=int(done.sum().item()))
            if reference_targets is not None:
                error = float((env._sm_joint_targets-reference_targets).abs().max())
                (args.output/'zero_update_targets.json').write_text(json.dumps({'max_error_rad': error, 'tolerance_rad': 2e-6,
                    'scope': 'first native tick, not a full native baseline'}, indent=2)+'\n')
                if error > 2e-6:
                    raise RuntimeError('Old/new final joint targets differ before learning')
                reference_targets = None
            true_next = nxt['policy'].detach().clone()
            valid = env.final_observation_valid
            true_next[valid] = env.final_observation[valid]
            replay.add(current, action, true_next, reward, done, elapsed, episode)
            if local_iteration == 0:
                stage('FIRST_REPLAY_WRITE', replay_size=len(replay))
            obs = nxt['policy']
            total_native += args.num_envs
            total_iterations += 1
            sync(); collect_end = time.perf_counter()
            wm = trainer.fit_world(args.world_updates, args.world_batch)
            sync(); wm_end = time.perf_counter()
            # No 800-tick/full-panel gate. Fit a real complete model window first.
            actor_stats = trainer.imagine_update(args.imagination_batch) if wm.get('world_updates', 0) else {'imagine_update': False}
            sync(); learn_end = time.perf_counter()
            metadata.update(native_steps=total_native, iterations_completed=total_iterations)
            row = {'iteration': total_iterations, 'native_steps': total_native, 'replay_size': len(replay),
                   'native_collect_s': collect_end-tick_start, 'world_update_s': wm_end-collect_end,
                   'imagination_bptt_s': learn_end-wm_end, **wm, **actor_stats,
                   'world_updates_total': trainer.world_updates, 'actor_updates_total': trainer.actor_updates,
                   'critic_updates_total': trainer.critic_updates,
                   'native_reward_mean': float(reward.mean().item()),
                   'native_reward_positive_fraction': float((reward > 0).float().mean().item()),
                   'native_done_count': int(done.sum().item()),
                   'native_event_totals': dict(native_event_totals)}
            if total_iterations % args.checkpoint_every == 0:
                trainer.save(args.output/f'checkpoint_{total_iterations:06d}.pt', metadata, args.persist_replay)
            sync(); row['logging_checkpoint_s'] = time.perf_counter()-learn_end
            row['iteration_s'] = time.perf_counter()-tick_start
            row['gpu_peak_allocated_bytes'] = torch.cuda.max_memory_allocated(env.device) if torch.device(env.device).type=='cuda' else 0
            with (args.output/'training_history.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            if args.auto_stop_convergence and row.get('imagine_update', False):
                convergence_history.append({key: row.get(key) for key in (
                    'world_loss', 'critic_loss', 'imagined_return', 'actor_grad_norm')})
                convergence_record = {
                    'iteration': total_iterations,
                    'stable_windows': convergence_stable_windows,
                    'history_size': len(convergence_history),
                    'converged_window': False,
                }
                window = args.convergence_window
                if (total_iterations >= args.convergence_min_iterations
                        and len(convergence_history) >= 2 * window
                        and total_iterations % window == 0):
                    history = list(convergence_history)
                    previous = history[-2 * window:-window]
                    current_window = history[-window:]

                    def finite_mean(rows, key):
                        values = [float(item[key]) for item in rows if item.get(key) is not None]
                        if len(values) != len(rows) or not all(value == value and abs(value) != float('inf') for value in values):
                            return None
                        return sum(values) / len(values)

                    previous_means = {key: finite_mean(previous, key) for key in (
                        'world_loss', 'critic_loss', 'imagined_return', 'actor_grad_norm')}
                    current_means = {key: finite_mean(current_window, key) for key in (
                        'world_loss', 'critic_loss', 'imagined_return', 'actor_grad_norm')}
                    convergence_record['previous_means'] = previous_means
                    convergence_record['current_means'] = current_means
                    if all(value is not None for value in (*previous_means.values(), *current_means.values())):
                        relative_changes = {
                            key: abs(current_means[key] - previous_means[key]) / max(abs(previous_means[key]), 1.0)
                            for key in ('world_loss', 'critic_loss')
                        }
                        return_change = abs(current_means['imagined_return'] - previous_means['imagined_return'])
                        converged_window = (
                            all(value <= args.convergence_relative_tolerance for value in relative_changes.values())
                            and return_change <= args.convergence_return_absolute_tolerance
                            and current_means['actor_grad_norm'] <= args.convergence_max_grad_norm
                        )
                        convergence_record.update(relative_changes=relative_changes,
                                                  return_change=return_change,
                                                  converged_window=converged_window)
                        if converged_window:
                            convergence_stable_windows += 1
                        else:
                            convergence_stable_windows = 0
                        convergence_record['stable_windows'] = convergence_stable_windows
                        if convergence_stable_windows >= args.convergence_patience:
                            stopping = True
                            convergence_stop_reason = 'converged'
                            convergence_record['stop'] = True
                with (args.output/'convergence_history.jsonl').open('a') as stream:
                    stream.write(json.dumps(convergence_record)+'\n')
            if total_iterations % 10 == 0 or local_iteration == 0:
                print('SUPERMIMIC_V2_PROGRESS='+json.dumps(row), flush=True)
                (args.output/'progress.json').write_text(json.dumps(row, indent=2)+'\n')
            if convergence_stop_reason == 'converged':
                break
        trainer.save(args.output/'checkpoint.pt', metadata, args.persist_replay)
        status = {'status': convergence_stop_reason or ('interrupted' if stopping else 'training_loop_completed'),
                  'iterations': total_iterations, 'native_steps': total_native,
                  'actor_updates': trainer.actor_updates, 'elapsed_s': time.perf_counter()-start,
                  'formal_success_rate': None,
                  'convergence': convergence_config if args.auto_stop_convergence else None}
        (args.output/'training_status.json').write_text(json.dumps(status, indent=2)+'\n')
        stage('SMOKE_FINISHED', iterations=total_iterations, native_steps=total_native,
              actor_updates=trainer.actor_updates, world_updates=trainer.world_updates,
              critic_updates=trainer.critic_updates)
    except BaseException as error:
        record_failure(error)
        raise
    finally:
        if env is not None:
            try:
                env.close()
            except BaseException as cleanup_error:
                (args.output / 'env_cleanup_failure.txt').write_text(traceback.format_exc())
        close_app()


if __name__ == '__main__':
    main()
