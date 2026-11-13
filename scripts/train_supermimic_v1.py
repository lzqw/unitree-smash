#!/usr/bin/env python3
"""Train the SuperMimic-style full-body policy with optional PPO warm start."""
from __future__ import annotations

import argparse, hashlib, json, signal, time
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--iterations', type=int, default=1000)
parser.add_argument('--num-envs', type=int, default=512)
parser.add_argument('--seed', type=int, default=20261111)
parser.add_argument('--batch-size', type=int, default=2048)
parser.add_argument('--world-updates', type=int, default=3)
parser.add_argument('--imagine-horizon', type=int, default=96,
                    help='GPU imagination horizon; 96 control steps = 0.96 s at 100 Hz')
parser.add_argument('--replay-capacity', type=int, default=250000)
parser.add_argument('--model-warmup-iterations', type=int, default=0,
                    help='additional iteration gate; actor still waits for actor-warmup-steps')
parser.add_argument('--actor-warmup-steps', type=int, default=102400,
                    help='native transitions before actor updates; 102400 = one 8 s, 128-env panel')
parser.add_argument('--resume', type=Path, default=None,
                    help='resume a SuperMimic checkpoint; independent of init-policy-checkpoint')
parser.add_argument('--init-policy-checkpoint', type=Path, default=None,
                    help='strictly load an existing native PPO actor as the initial policy')
parser.add_argument('--checkpoint-every', type=int, default=100)
parser.add_argument('--persist-replay', action='store_true',
                    help='materialize the GPU replay ring only at checkpoints')
parser.add_argument('--resume-without-replay', action='store_true',
                    help='restore actor/world/optimizers but intentionally rebuild replay from new native transitions')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from quadsmash.rl.supermimic_v1 import SuperMimicTrainer, load_ppo_initialized_actor
from quadsmash.tasks.stand.supermimic_env_v1 import SuperMimicEnvV1, make_supermimic_cfg


def sync_cuda(device):
    if torch.cuda.is_available() and torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def main():
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = make_supermimic_cfg(args.num_envs)
    cfg.seed = args.seed
    cfg.sim.device = args.device or 'cuda:0'
    # Training uses native PhysX contact events; the expensive swept-geometry
    # audit is kept at high fidelity by the formal evaluator only.
    cfg.sweep_audit_iterations = 0
    protocol = json.loads((ROOT / 'configs/g07_full_racket_eval_v4_r3.json').read_text())
    suite = protocol['training_suite']
    base_cases = json.loads((ROOT / suite['path']).read_text())['cases']
    cases = [base_cases[i % len(base_cases)] for i in range(args.num_envs)]
    residual_policy = None
    if args.init_policy_checkpoint is not None:
        probe = torch.load(args.init_policy_checkpoint, map_location='cpu', weights_only=False)
        if probe.get('schema') == 'full-racket-residual-ppo-v4':
            residual_policy = probe
    env = SuperMimicEnvV1(cfg, cases=cases, residual_policy_checkpoint=residual_policy)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
    obs, _ = wrapped.reset()
    obs_tensor = obs['policy']
    init_meta = None
    if args.init_policy_checkpoint is not None:
        actor, init_meta = load_ppo_initialized_actor(args.init_policy_checkpoint, env.device)
        trainer = SuperMimicTrainer(obs_tensor.shape[-1], 18, env.device,
            imagine_horizon=args.imagine_horizon, replay_capacity=args.replay_capacity,
            actor=actor, actor_kind=init_meta['actor_kind'])
    else:
        trainer = SuperMimicTrainer(obs_tensor.shape[-1], 18, env.device,
            imagine_horizon=args.imagine_horizon, replay_capacity=args.replay_capacity)
    manifest = {
        'algorithm': 'supermimic_v1', 'random_initialization': args.init_policy_checkpoint is None,
        'old_checkpoint': None, 'teacher': None, 'mpc': None, 'bc': None,
        'num_envs': args.num_envs, 'iterations': args.iterations,
        'control_dt_s': 0.01, 'episode_seconds': 8.0,
        'obs_dim': int(obs_tensor.shape[-1]), 'action_dim': 18,
        'seed': args.seed, 'event_reward_version': 'supermimic-native-v1',
        'init_policy_checkpoint': init_meta,
        'critic_initialization': 'random_refit',
        'old_normalizer_frozen_contract': bool(init_meta is not None),
        'actor_warmup_steps': int(args.actor_warmup_steps),
        'episode_seconds_contract': 8.0,
        'control_ticks_per_episode_contract': 800,
        'model_window_steps': 16,
        'model_window_seconds': 0.16,
        'imagine_horizon_steps': int(args.imagine_horizon),
        'imagine_horizon_seconds': float(args.imagine_horizon) * 0.01,
        'physical_action_contract': {
            'learner_action': 'residual_request_or_native_action',
            'final_action_source': 'SuperMimicEnvV1._sm_final_action',
            'request_final_delta_logged': True,
            'native_execution': True,
        },
    }
    if init_meta is not None:
        # Do not compare trainer.action() with the same trainer.actor() and
        # call that an execution-contract test.  The meaningful check needs
        # identical saved simulator states and controller memory replayed
        # through the old complete chain and this new entry point.  That
        # native replay is deliberately a separate CPU/Isaac validation step;
        # this learner records the pending status instead of manufacturing a
        # zero-to-zero result.
        manifest['initialization_contract_validation'] = {
            'status': 'native_saved_state_full_chain_replay_pending',
            'scope': ['prepare', 'jump', 'contact', 'recovery'],
            'request_vs_final_difference_is_not_mapping_error': True,
        }
    resumed_native_steps = 0
    if args.resume is not None:
        prior = trainer.load(args.resume, restore_replay=not args.resume_without_replay)
        prior_meta = prior.get('metadata', {})
        if prior_meta.get('old_checkpoint') is not None or prior.get('algorithm') != 'supermimic_v1':
            raise ValueError('resume must be a from-scratch SuperMimic checkpoint')
        manifest['resumed_from'] = str(args.resume.resolve())
        manifest['resume_update_steps'] = int(prior.get('update_steps', 0))
        manifest['replay_resumed'] = bool(prior.get('replay', {}).get('obs') is not None)
        manifest['replay_rebuilt_after_resume'] = bool(args.resume_without_replay)
        resumed_native_steps = int(prior.get('metadata', {}).get('native_steps', 0))
    (args.output / 'training_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    total_steps = resumed_native_steps; started = time.perf_counter(); stop = False
    def halt(*_):
        nonlocal stop; stop = True
    signal.signal(signal.SIGINT, halt); signal.signal(signal.SIGTERM, halt)
    records = []
    for iteration in range(args.iterations):
        if stop: break
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats(env.device)
        t_iter = time.perf_counter(); t_transfer = 0.0
        sync_cuda(env.device); t0 = time.perf_counter()
        with torch.no_grad():
            action = trainer.action(obs_tensor, sample=True)
        episode_before = env.episode_index.detach().clone()
        stream_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
        next_obs, reward, done, info = wrapped.step(action)
        sync_cuda(env.device)
        t_native = time.perf_counter() - t0
        requested_action = getattr(env, '_sm_requested_action', action)
        final_action = getattr(env, '_sm_final_action', action)
        execution_delta = (final_action - requested_action).abs()
        nxt = next_obs['policy']
        # Preserve the true pre-reset terminal observation for model fitting.
        final_valid = getattr(env, 'final_observation_valid', torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
        final_obs = getattr(env, 'final_observation', nxt)
        transition_next = nxt.clone()
        if bool(final_valid.any()):
            transition_next[final_valid] = final_obs[final_valid]
        event = (env.rally.in_play | env.rally.had_contact | env.observed_airborne | env.body_ground_seen | done).detach()
        # Clone the simulator-owned tensors before writing the replay.  This
        # creates an explicit CUDA dependency boundary and prevents Isaac's
        # observation buffers from aliasing the learner's ring buffer.
        t0 = time.perf_counter()
        trainer.replay.add(obs_tensor.detach().clone(), action.detach().clone(), transition_next.detach().clone(),
                            reward.detach().clone(), done.detach().clone(), event.clone(),
                            stream=stream_ids, episode=episode_before)
        t_transfer += time.perf_counter() - t0
        obs_tensor = nxt
        total_steps += env.num_envs
        sync_cuda(env.device); t0 = time.perf_counter()
        model_stats = trainer.fit_world(args.world_updates, args.batch_size)
        sync_cuda(env.device)
        t_world = time.perf_counter() - t0
        sync_cuda(env.device); t0 = time.perf_counter()
        actor_ready = (total_steps >= int(args.actor_warmup_steps) and
                       len(env.rally_records) >= env.num_envs)
        imagine_stats = ({'imagine_update': False, 'model_warmup': True,
                          'actor_warmup': not actor_ready}
                          if (iteration + 1 <= args.model_warmup_iterations or not actor_ready)
                          else trainer.imagine_update(args.batch_size))
        if not actor_ready:
            imagine_stats.update({'actor_warmup': True})
        sync_cuda(env.device); t_imagine = time.perf_counter() - t0
        peak_alloc = (torch.cuda.max_memory_allocated(env.device) if torch.cuda.is_available() else 0)
        peak_reserved = (torch.cuda.max_memory_reserved(env.device) if torch.cuda.is_available() else 0)
        row = {'iteration': iteration + 1, 'native_steps': total_steps, 'replay_size': len(trainer.replay),
               'elapsed_s': time.perf_counter() - started,
               'iter_s': time.perf_counter() - t_iter,
               'native_collect_s': t_native,
               'world_update_s': t_world,
               'imagination_bptt_s': t_imagine,
               'data_transfer_s': t_transfer,
               'logging_save_s': 0.0,
               'gpu_peak_allocated_bytes': int(peak_alloc),
               'gpu_peak_reserved_bytes': int(peak_reserved),
               'learner_device': str(env.device), **model_stats, **imagine_stats,
               'actor_ready': bool(actor_ready),
               'native_episode_records': len(env.rally_records),
               'native_complete5_records': sum(bool(r.get('complete5', False)) for r in env.rally_records),
               'requested_final_action_max_abs': float(execution_delta.max().detach()),
               'requested_final_action_mean_abs': float(execution_delta.mean().detach())}
        records.append(row)
        t0 = time.perf_counter()
        if (iteration + 1) % 10 == 0 or iteration == 0:
            print('SUPERMIMIC_PROGRESS=' + json.dumps(row), flush=True)
            (args.output / 'progress.json').write_text(json.dumps(row, indent=2) + '\n')
        if (iteration + 1) % args.checkpoint_every == 0:
            trainer.save(args.output / f'checkpoint_{iteration+1:06d}.pt', manifest | {'iteration': iteration + 1}, include_replay=args.persist_replay)
        row['logging_save_s'] = time.perf_counter() - t0
    trainer.save(args.output / 'checkpoint.pt', manifest | {'iterations_completed': len(records), 'native_steps': total_steps}, include_replay=args.persist_replay)
    (args.output / 'training_history.json').write_text(json.dumps(records, indent=2) + '\n')
    keys = ('native_collect_s', 'world_update_s', 'imagination_bptt_s', 'data_transfer_s', 'logging_save_s', 'iter_s')
    profile = {k: {'mean_s': (sum(float(r.get(k, 0.0)) for r in records) / max(1, len(records))),
                   'max_s': max((float(r.get(k, 0.0)) for r in records), default=0.0)} for k in keys}
    profile.update({'device': str(env.device), 'iterations': len(records), 'native_steps': total_steps,
                    'peak_gpu_allocated_bytes': max((int(r.get('gpu_peak_allocated_bytes', 0)) for r in records), default=0),
                    'peak_gpu_reserved_bytes': max((int(r.get('gpu_peak_reserved_bytes', 0)) for r in records), default=0),
                    'replay_device': str(trainer.replay.storage_device), 'replay_size': len(trainer.replay)})
    (args.output / 'profile_summary.json').write_text(json.dumps(profile, indent=2) + '\n')
    (args.output / 'completed.json').write_text(json.dumps({'iterations': len(records), 'native_steps': total_steps, 'elapsed_s': time.perf_counter()-started}, indent=2) + '\n')
    env.close()


try:
    main()
finally:
    app.close()
