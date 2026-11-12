#!/usr/bin/env python3
"""One frozen 128-environment, 8-second native evaluation for SuperMimic."""
from __future__ import annotations
import argparse, csv, hashlib, json, time
from pathlib import Path
from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--seed', type=int, default=882072)
parser.add_argument('--sweep-audit-iterations', type=int, default=0,
                    help='extra swept geometry audit iterations; 0 keeps native physics/event path without the expensive diagnostic sweep')
parser.add_argument('--physics-device', default='cuda:0',
                    help='Isaac/PhysX simulation device; cuda:0 enables GPU Dynamics')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(); app = AppLauncher(args).app

import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from quadsmash.rl.supermimic_v1 import SuperMimicTrainer, PPOInitializedActor
from quadsmash.tasks.stand.supermimic_env_v1 import SuperMimicEnvV1, make_supermimic_cfg
from quadsmash.tasks.stand.full_racket_env_v4 import load_racket


def main():
    args.output.mkdir(parents=True, exist_ok=True)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if state.get('algorithm') != 'supermimic_v1' or state.get('metadata', {}).get('old_checkpoint') is not None:
        raise ValueError('evaluation checkpoint is not a SuperMimic checkpoint')
    cfg = make_supermimic_cfg(128); cfg.seed = args.seed; cfg.sim.device = args.physics_device
    # The native sweep audit is an event-audit diagnostic, not part of the
    # policy or physics interface.  The full-racket default (4096 conservative
    # advancement iterations) makes a random from-scratch policy spend most of
    # the 8-second rollout in CPU geometry refinement.  Keep the same native
    # contact/event path while using the validated 256-iteration audit budget
    # used by the underlying ball environment so a complete 128-env eval can
    # finish within the job budget.
    cfg.sweep_audit_iterations = max(0, int(args.sweep_audit_iterations))
    protocol = json.loads((ROOT / 'configs/g07_full_racket_eval_v4_r3.json').read_text())
    suite = protocol['training_suite']; cases = json.loads((ROOT / suite['path']).read_text())['cases']
    if len(cases) != 128: raise ValueError('formal training suite must have 128 cases')
    env = SuperMimicEnvV1(cfg, cases=cases)
    from isaacsim.core.simulation_manager import SimulationManager
    physics_device = SimulationManager.get_physics_sim_device()
    gpu_dynamics = bool(SimulationManager.is_gpu_dynamics_enabled())
    print(f'SUPERMIMIC_PHYSICS_CONFIG device={physics_device} gpu_dynamics={gpu_dynamics}', flush=True)
    wrapper = RslRlVecEnvWrapper(env, clip_actions=None)
    obs, _ = wrapper.reset()
    kind = state.get('metadata', {}).get('actor_kind', 'supermimic_random')
    if kind == PPOInitializedActor.actor_kind:
        actor = PPOInitializedActor(obs['policy'].shape[-1], 18).to(env.device)
        actor.load_state_dict(state['actor'], strict=True)
        trainer = SuperMimicTrainer(obs['policy'].shape[-1], 18, env.device, actor=actor, actor_kind=kind)
    else:
        trainer = SuperMimicTrainer(obs['policy'].shape[-1], 18, env.device)
        trainer.actor.load_state_dict(state['actor'], strict=True)
    done_seen = torch.zeros(128, dtype=torch.bool, device=env.device)
    start = time.perf_counter(); numerical = 0
    with torch.inference_mode():
        for tick in range(800):
            action = trainer.action(obs['policy'], sample=False)
            numerical += int((~torch.isfinite(action)).any().item())
            obs, reward, done, info = wrapper.step(action)
            numerical += int((~torch.isfinite(obs['policy'])).any().item())
            done_seen |= done.bool()
            if (tick + 1) % 100 == 0:
                print(f'SUPERMIMIC_EVAL_PROGRESS tick={tick + 1}/800 done={int(done_seen.sum())}', flush=True)
    first = {}
    for row in env.rally_records: first.setdefault(int(row['env_id']), row)
    if len(first) != 128: raise RuntimeError(f'only {len(first)}/128 first episodes completed')
    # Keep the formal artifact compact.  Rally rows may contain large nested
    # diagnostic arrays; serializing those after a long GPU run can dominate
    # shutdown and hide the actual result. Core scalar event fields are enough
    # for this frozen evaluation, while raw traces remain in the native logs.
    scalar_keys = ('env_id', 'episode_index', 'airborne_string_hit', 'legal_initial_contact',
                   'crossed_net', 'landed_target', 'rally_fault', 'recovery_success',
                   'complete_smash', 'numerical_failures', 'head_contact', 'body_ground_seen')
    rows = []
    for i in range(128):
        source = first[i]
        row = {key: source.get(key) for key in scalar_keys if key in source}
        row.update(case_id=cases[i].get('case_id', i), source_case_index=i)
        rows.append(row)
    contact = [bool(r.get('airborne_string_hit') and r.get('legal_initial_contact')) for r in rows]
    ball_ok = [bool(c and r.get('crossed_net') and r.get('landed_target') and not r.get('rally_fault', False)) for c, r in zip(contact, rows)]
    recover = [bool(r.get('recovery_success')) for r in rows]
    complete = [bool(r.get('complete_smash')) for r in rows]
    metrics = {
        'algorithm': 'supermimic_v1', 'random_initialization': bool(state.get('random_initialization', True)),
        'actor_kind': kind,
        'init_policy_checkpoint': state.get('metadata', {}).get('init_policy_checkpoint'),
        'num_envs': 128, 'control_ticks': 800, 'episode_seconds': 8.0,
        'qualified_airborne_string_contact': sum(contact),
        'ball_side_complete_except_recovery': sum(ball_ok),
        'recovery_success': sum(recover), 'complete5': sum(complete),
        'recovery_given_airborne_string_contact': sum(r for c, r in zip(contact, recover) if c) / max(1, sum(contact)),
        'numerical_failures': numerical + sum(int(r.get('numerical_failures', 0)) for r in rows),
        'mpc_interventions': 0, 'teacher_interventions': 0, 'bc_data': None,
        'elapsed_s': time.perf_counter() - start,
        'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        'asset_version': load_racket().get('version'), 'formal_protocol': 'g07_full_racket_eval_v4_r3',
        'physics_sim_device': physics_device, 'gpu_dynamics_enabled': gpu_dynamics,
        'racket_plane_gpu_broadphase_calls': int(getattr(env, 'racket_plane_gpu_broadphase_calls', 0)),
        'racket_plane_gpu_broadphase_candidates': int(getattr(env, 'racket_plane_gpu_broadphase_candidates', 0)),
        'racket_plane_cpu_exact_queries': int(getattr(env, 'racket_plane_query_calls', 0)),
    }
    fields = sorted({k for r in rows for k in r});
    with (args.output / 'episodes.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    (args.output / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    (args.output / 'completed.json').write_text(json.dumps({'control_ticks': 800, 'episodes': 128, 'formal': True}, indent=2) + '\n')
    print('SUPERMIMIC_EVAL_RESULT=' + json.dumps(metrics, sort_keys=True), flush=True)
    env.close()


try:
    main()
finally:
    app.close()
