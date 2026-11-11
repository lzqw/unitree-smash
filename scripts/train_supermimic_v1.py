#!/usr/bin/env python3
"""Train the independent SuperMimic-style full-body policy from scratch."""
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
parser.add_argument('--imagine-horizon', type=int, default=32)
parser.add_argument('--replay-capacity', type=int, default=250000)
parser.add_argument('--model-warmup-iterations', type=int, default=1000)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from quadsmash.rl.supermimic_v1 import SuperMimicTrainer
from quadsmash.tasks.stand.supermimic_env_v1 import SuperMimicEnvV1, make_supermimic_cfg


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
    env = SuperMimicEnvV1(cfg, cases=cases)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
    obs, _ = wrapped.reset()
    obs_tensor = obs['policy']
    trainer = SuperMimicTrainer(obs_tensor.shape[-1], 18, env.device,
        imagine_horizon=args.imagine_horizon, replay_capacity=args.replay_capacity)
    manifest = {
        'algorithm': 'supermimic_v1', 'random_initialization': True,
        'old_checkpoint': None, 'teacher': None, 'mpc': None, 'bc': None,
        'num_envs': args.num_envs, 'iterations': args.iterations,
        'control_dt_s': 0.01, 'episode_seconds': 8.0,
        'obs_dim': int(obs_tensor.shape[-1]), 'action_dim': 18,
        'seed': args.seed, 'event_reward_version': 'supermimic-native-v1',
    }
    (args.output / 'training_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    total_steps = 0; started = time.perf_counter(); stop = False
    def halt(*_):
        nonlocal stop; stop = True
    signal.signal(signal.SIGINT, halt); signal.signal(signal.SIGTERM, halt)
    records = []
    for iteration in range(args.iterations):
        if stop: break
        with torch.no_grad():
            action = trainer.action(obs_tensor, sample=True)
        next_obs, reward, done, info = wrapped.step(action)
        nxt = next_obs['policy']
        event = (env.rally.in_play | env.rally.had_contact | env.observed_airborne | env.body_ground_seen | done).detach()
        # Clone the simulator-owned tensors before writing the replay.  This
        # creates an explicit CUDA dependency boundary and prevents Isaac's
        # observation buffers from aliasing the learner's ring buffer.
        trainer.replay.add(obs_tensor.detach().clone(), action.detach().clone(), nxt.detach().clone(),
                            reward.detach().clone(), done.detach().clone(), event.clone())
        obs_tensor = nxt
        total_steps += env.num_envs
        model_stats = trainer.fit_world(args.world_updates, args.batch_size)
        imagine_stats = ({'imagine_update': False, 'model_warmup': True}
                          if iteration + 1 <= args.model_warmup_iterations
                          else trainer.imagine_update(args.batch_size))
        row = {'iteration': iteration + 1, 'native_steps': total_steps, 'replay_size': len(trainer.replay),
               'elapsed_s': time.perf_counter() - started, **model_stats, **imagine_stats,
               'native_complete_records': len(env.rally_records)}
        records.append(row)
        if (iteration + 1) % 10 == 0 or iteration == 0:
            print('SUPERMIMIC_PROGRESS=' + json.dumps(row), flush=True)
            (args.output / 'progress.json').write_text(json.dumps(row, indent=2) + '\n')
        if (iteration + 1) % 100 == 0:
            trainer.save(args.output / f'checkpoint_{iteration+1:06d}.pt', manifest | {'iteration': iteration + 1})
    trainer.save(args.output / 'checkpoint.pt', manifest | {'iterations_completed': len(records), 'native_steps': total_steps})
    (args.output / 'training_history.json').write_text(json.dumps(records, indent=2) + '\n')
    (args.output / 'completed.json').write_text(json.dumps({'iterations': len(records), 'native_steps': total_steps, 'elapsed_s': time.perf_counter()-started}, indent=2) + '\n')
    env.close()


try:
    main()
finally:
    app.close()
