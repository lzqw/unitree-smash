#!/usr/bin/env python3
"""Opt-in full-joint trajectory search ranked by ordered net/landing outcomes.

Run only through run_gpu_job.py with a currently valid authorization window.
All 18 position targets and their timing can depart from the old teacher.
Search results are demonstrations, never autonomous-policy acceptance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--num-envs', type=int, default=64)
parser.add_argument('--generations', type=int, default=8)
parser.add_argument('--seed', type=int, default=714001)
parser.add_argument('--physics-dt', type=float, default=.00025)
parser.add_argument('--launch-config', type=Path, required=True)
parser.add_argument('--init-trace', type=Path, default=ROOT/'reports/G07/root_probe_all01/trajectory.json')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs < 8 or args.generations < 1:
    parser.error('num-envs >= 8 and generations >= 1 are required')
if args.physics_dt <= 0 or abs(.01 / args.physics_dt - round(.01 / args.physics_dt)) > 1e-8:
    parser.error('physics-dt must divide the fixed 10 ms policy period')
app = AppLauncher(args).app

import numpy as np
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from quadsmash.physics.smash_events import rank_candidate
from quadsmash.control.smash_search import add_flight_search_diagnostics, search_candidate_key
from quadsmash.tasks.stand.ball_env import BallEnvCfg
from quadsmash.tasks.stand.smash_env_v3 import OrderedSmashEnv

stopping = False


def stop(*_):
    global stopping
    stopping = True


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)


class FullJointSearchEnv(OrderedSmashEnv):
    def _pre_physics_step(self, actions):
        # New opt-in command interface: absolute joint positions cover audited
        # joint limits. Actual actuator efforts/velocities/gains are unchanged.
        self.action_delta.copy_((actions.clamp(-1, 1) - self.last_action).square().sum(1))
        self._record_action(actions)
        self.action_clip_count += (actions.abs() > 1).sum(1)
        limits = self.robot.data.joint_pos_limits[:, self.joint_ids]
        q = limits[..., 0] + .5 * (self.last_action + 1) * (limits[..., 1] - limits[..., 0])
        self.targets.copy_(self.robot.data.default_joint_pos)
        self.targets[:, self.joint_ids] = q
        self._contact_bonus.zero_()


def main():
    args.output.mkdir(parents=True, exist_ok=True)
    launch = json.loads(args.launch_config.read_text())
    cfg = BallEnvCfg()
    cfg.scene.num_envs, cfg.seed = args.num_envs, args.seed
    cfg.sim.device, cfg.sim.dt = 'cuda:0', args.physics_dt
    cfg.decimation = round(.01 / args.physics_dt)
    cfg.sim.render_interval = cfg.decimation
    cfg.episode_length_s = 4.
    cfg.include_net = True
    # Only launch/command fields may change here; hardware/net are fixed.
    allowed = {'shuttle_start', 'shuttle_velocity', 'event_goal_position',
               'event_goal_normal', 'event_goal_time_s', 'event_goal_velocity'}
    for key, value in launch.items():
        if key not in allowed:
            raise ValueError(f'Unsupported physical-launch override: {key}')
        setattr(cfg, key, tuple(value) if isinstance(value, list) else value)
    env = FullJointSearchEnv(cfg)
    try:
        wrapper = RslRlVecEnvWrapper(env, clip_actions=None)
        rng = torch.Generator(device=env.device).manual_seed(args.seed)
        limits = env.robot.data.joint_pos_limits[0, env.joint_ids]
        lower, upper = limits[:, 0], limits[:, 1]
        default = env.robot.data.default_joint_pos[0, env.joint_ids]
        normalize = lambda q: 2 * (q - lower) / (upper - lower) - 1
        times = torch.tensor([0., .5, .75, .95, 1.10, 1.25, 1.40, 1.60, 1.90, 2.5, 4.], device=env.device)
        trace = json.loads(args.init_trace.read_text())
        sample_times = np.array([r['time_s'] for r in trace])
        actions = np.array([r['action'] for r in trace])
        initial = []
        for t in times.cpu().tolist():
            old = torch.tensor([np.interp(t, sample_times, actions[:, j]) for j in range(18)],
                               device=env.device, dtype=torch.float32)
            scale = torch.tensor([.25 if t < .8 else .55]*12 + [2.]*6, device=env.device)
            initial.append(normalize(default + scale * old).clamp(-1, 1))
        mean = torch.stack(initial)
        sigma = torch.full_like(mean, .18)
        sigma[0] = 0
        sigma[-2:] = .04
        timing_mean, timing_sigma = 1., .18
        history, all_best = [], None
        for generation in range(args.generations):
            knots = (mean[None] + sigma[None] * torch.randn(
                (args.num_envs, len(times), 18), generator=rng, device=env.device)).clamp(-1, 1)
            knots[:, 0] = normalize(default)
            knots[0] = mean
            warp = (timing_mean + timing_sigma * torch.randn(args.num_envs, generator=rng, device=env.device)).clamp(.7, 1.5)
            warp[0] = timing_mean
            env._episode_has_stepped.zero_()
            env._metrics_started = False
            env.episode_records.clear()
            env.ball_episode_records.clear()
            env.smash_episode_records.clear()
            obs, _ = wrapper.reset()
            with torch.inference_mode():
                for _ in range(400):
                    clock = obs['policy'][:, 775] / warp
                    index = torch.searchsorted(times, clock, right=True).clamp(1, len(times)-1)
                    fraction = ((clock-times[index-1])/(times[index]-times[index-1])).clamp(0, 1)
                    lane = torch.arange(args.num_envs, device=env.device)
                    action = torch.lerp(knots[lane, index-1], knots[lane, index], fraction[:, None])
                    obs, _, _, _ = wrapper.step(action)
            first = {}
            for row in env.smash_episode_records:
                first.setdefault(row['env_id'], row)
            if len(first) != args.num_envs:
                raise RuntimeError('Incomplete physical candidate batch')
            rows = [first[i] for i in range(args.num_envs)]
            add_flight_search_diagnostics(rows)
            order = sorted(range(args.num_envs), key=lambda i: search_candidate_key(rows[i]), reverse=True)
            elite = torch.tensor(order[:max(4, args.num_envs//8)], device=env.device)
            mean = .25*mean + .75*knots[elite].mean(0)
            sigma = (.25*sigma + .75*knots[elite].std(0)).clamp(.025, .4)
            sigma[0] = 0
            timing_mean = float(warp[elite].mean())
            timing_sigma = max(.025, float(warp[elite].std()))
            best = rows[order[0]]
            entry = dict(generation=generation, complete=sum(r['complete_smash'] for r in rows),
                         ordered_landing=sum(r['ordered_landed_target'] for r in rows),
                         ordered_net=sum(r['ordered_crossed_net'] for r in rows),
                         airborne_contacts=sum(r['airborne_contact'] for r in rows), best=best)
            history.append(entry)
            if all_best is None or search_candidate_key(best) > all_best[0]:
                all_best = (search_candidate_key(best), best)
                torch.save(dict(knots=knots[order[0]].cpu(), knot_times=times.cpu(),
                                time_scale=float(warp[order[0]]), launch=launch,
                                action_interface='absolute-joint-limits-18D-v3',
                                kind='open-loop demonstration candidate, not learned policy', result=best),
                           args.output/'best_candidate.pt')
            # Retain real useful shots even when recovery still fails, so
            # later recovery work can replay the exact joint/time candidate.
            promising = [i for i in order if rank_candidate(rows[i])[2]]
            if promising:
                bank = args.output/'feasible_shots'
                bank.mkdir(exist_ok=True)
                for i in promising:
                    torch.save(dict(knots=knots[i].cpu(), knot_times=times.cpu(),
                                    time_scale=float(warp[i]), launch=launch, result=rows[i],
                                    generation=generation, env_id=i, seed=args.seed,
                                    kind='measured ordered-net demonstration; not policy acceptance'),
                               bank/f'generation_{generation:03d}_lane_{i:03d}.pt')
            (args.output/f'episodes_{generation:03d}.json').write_text(json.dumps(rows, indent=2)+'\n')
            (args.output/'metrics.json').write_text(json.dumps(dict(history=history,
                env_steps=(generation+1)*args.num_envs*400, best=all_best[1],
                method='full 18-joint spline + timing CEM', protocol='ordered-smash-v3',
                policy_acceptance=False), indent=2)+'\n')
            print(json.dumps(entry), flush=True)
            if stopping:
                break
    finally:
        env.close()


try:
    main()
finally:
    app.close(skip_cleanup=True)
