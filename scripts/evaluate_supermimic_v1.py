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
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(); app = AppLauncher(args).app

import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from quadsmash.rl.supermimic_v1 import SuperMimicTrainer
from quadsmash.tasks.stand.supermimic_env_v1 import SuperMimicEnvV1, make_supermimic_cfg
from quadsmash.tasks.stand.full_racket_env_v4 import load_racket


def main():
    args.output.mkdir(parents=True, exist_ok=True)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if state.get('algorithm') != 'supermimic_v1' or state.get('metadata', {}).get('old_checkpoint') is not None:
        raise ValueError('evaluation checkpoint is not a from-scratch SuperMimic checkpoint')
    cfg = make_supermimic_cfg(128); cfg.seed = args.seed; cfg.sim.device = args.device or 'cuda:0'
    protocol = json.loads((ROOT / 'configs/g07_full_racket_eval_v4_r3.json').read_text())
    suite = protocol['training_suite']; cases = json.loads((ROOT / suite['path']).read_text())['cases']
    if len(cases) != 128: raise ValueError('formal training suite must have 128 cases')
    env = SuperMimicEnvV1(cfg, cases=cases)
    wrapper = RslRlVecEnvWrapper(env, clip_actions=None)
    obs, _ = wrapper.reset(); trainer = SuperMimicTrainer(obs['policy'].shape[-1], 18, env.device)
    trainer.actor.load_state_dict(state['actor'])
    done_seen = torch.zeros(128, dtype=torch.bool, device=env.device)
    start = time.perf_counter(); numerical = 0
    with torch.inference_mode():
        for tick in range(800):
            action = trainer.action(obs['policy'], sample=False)
            numerical += int((~torch.isfinite(action)).any().item())
            obs, reward, done, info = wrapper.step(action)
            numerical += int((~torch.isfinite(obs['policy'])).any().item())
            done_seen |= done.bool()
    first = {}
    for row in env.rally_records: first.setdefault(int(row['env_id']), row)
    if len(first) != 128: raise RuntimeError(f'only {len(first)}/128 first episodes completed')
    rows = [dict(first[i], case_id=cases[i].get('case_id', i), source_case_index=i) for i in range(128)]
    contact = [bool(r.get('airborne_string_hit') and r.get('legal_initial_contact')) for r in rows]
    ball_ok = [bool(c and r.get('crossed_net') and r.get('landed_target') and not r.get('rally_fault', False)) for c, r in zip(contact, rows)]
    recover = [bool(r.get('recovery_success')) for r in rows]
    complete = [bool(r.get('complete_smash')) for r in rows]
    metrics = {
        'algorithm': 'supermimic_v1', 'random_initialization': True,
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
