#!/usr/bin/env python3
"""PPO-initialized, seconds-based BPTT/GAE training. Never launches an evaluator.

Code review only does not launch this script. Native tests and a working
initialization contract remain necessary before accepting training results.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import signal
import time

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
    parser.add_argument('--replay-capacity', type=int, default=250000)
    parser.add_argument('--checkpoint-every', type=int, default=500)
    parser.add_argument('--persist-replay', action='store_true')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--imagination-seconds', type=float)
    parser.add_argument('--world-window-seconds', type=float)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if min(args.iterations, args.num_envs, args.world_batch, args.world_updates,
           args.imagination_batch, args.checkpoint_every) <= 0:
        parser.error('Counts must be positive')
    app = AppLauncher(args).app
    env = None
    try:
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

        torch.manual_seed(args.seed)
        args.output.mkdir(parents=True, exist_ok=False)
        temporal = json.loads(args.timebase_config.read_text())
        if args.imagination_seconds is not None:
            temporal['imagination_s'] = args.imagination_seconds
        if args.world_window_seconds is not None:
            temporal['world_window_s'] = args.world_window_seconds
        timebase = Timebase(**temporal)
        source = torch.load(args.init_policy_checkpoint, map_location='cpu', weights_only=False)
        if source.get('schema') != 'full-racket-residual-ppo-v4':
            raise ValueError('This first v2 entry requires the audited full-racket residual contract')
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
        env = SuperMimicEnvV1(cfg, cases=cases, residual_policy_checkpoint=source)
        wrapper = RslRlVecEnvWrapper(env, clip_actions=None)
        obs, _ = wrapper.reset()
        obs = obs['policy']
        actor, init_meta = load_ppo_initialized_actor(args.init_policy_checkpoint, env.device)
        replay = SequenceReplay(args.replay_capacity, args.num_envs, obs.shape[-1], 18, env.device)
        if replay.slots < timebase.world_steps:
            raise ValueError('Replay too short for configured physical-time window')
        trainer = HorizonTrainer(actor, SuperMimicWorldModel(obs.shape[-1], 18).to(env.device),
                                 SuperMimicCritic(obs.shape[-1]).to(env.device), replay, timebase,
                                 actor_microbatch=args.actor_microbatch, actor_kind=init_meta['actor_kind'],
                                 observation_clock_index=775 if obs.shape[-1] == 794 else None)
        metadata = {'init_policy_checkpoint': init_meta, 'old_checkpoint': None, 'mpc': None, 'teacher': None,
                    'bc': None, 'num_envs': args.num_envs, 'seed': args.seed,
                    'timebase': timebase.manifest(), 'init_source_sha256': hashlib.sha256(args.init_policy_checkpoint.read_bytes()).hexdigest(),
                    'world_representation': 'legacy_observation_delta_adapter_not_official_g1_representation',
                    'formal_evaluation_started': False}
        total_native = total_iterations = 0
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
            nonlocal stopping
            stopping = True
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
            episode = env.episode_index.detach().clone()
            elapsed = env.episode_length_buf.float().clone() * env.step_dt
            action = trainer.action(current, sample=(local_iteration != 0 or args.resume is not None))
            nxt, reward, done, info = wrapper.step(action)
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
                   'critic_updates_total': trainer.critic_updates}
            if total_iterations % args.checkpoint_every == 0:
                trainer.save(args.output/f'checkpoint_{total_iterations:06d}.pt', metadata, args.persist_replay)
            sync(); row['logging_checkpoint_s'] = time.perf_counter()-learn_end
            row['iteration_s'] = time.perf_counter()-tick_start
            row['gpu_peak_allocated_bytes'] = torch.cuda.max_memory_allocated(env.device) if torch.device(env.device).type=='cuda' else 0
            with (args.output/'training_history.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            if total_iterations % 10 == 0 or local_iteration == 0:
                print('SUPERMIMIC_V2_PROGRESS='+json.dumps(row), flush=True)
                (args.output/'progress.json').write_text(json.dumps(row, indent=2)+'\n')
        trainer.save(args.output/'checkpoint.pt', metadata, args.persist_replay)
        status = {'status': 'interrupted' if stopping else 'training_loop_completed',
                  'iterations': total_iterations, 'native_steps': total_native,
                  'actor_updates': trainer.actor_updates, 'elapsed_s': time.perf_counter()-start,
                  'formal_success_rate': None}
        (args.output/'training_status.json').write_text(json.dumps(status, indent=2)+'\n')
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == '__main__':
    main()
