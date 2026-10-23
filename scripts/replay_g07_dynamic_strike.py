#!/usr/bin/env python3
"""Independently replay a frozen, physical near-net controller.

This records repeatability and measured motion. It is explicitly a curriculum
evaluation and cannot replace the canonical-ground G07 acceptance panels.
"""
import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--candidate', type=Path, required=True)
parser.add_argument('--num-envs', type=int, default=64)
parser.add_argument('--seed', type=int, default=714501)
parser.add_argument('--policy', type=Path,
                    help='Frozen learned PWA policy; candidate supplies scene configuration only')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from quadsmash.control.dynamic_reach_teacher import dynamic_reach_action
from quadsmash.control.smash_search import add_flight_search_diagnostics, search_candidate_key
from quadsmash.tasks.stand.ball_env import BallEnvCfg
from quadsmash.tasks.stand.smash_env_v3 import FullJointSmashEnv


def main():
    args.output.mkdir(parents=True, exist_ok=True)
    candidate = torch.load(args.candidate, map_location='cpu', weights_only=False)
    launch, base = candidate['launch'], candidate['base']
    cfg = BallEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.sim.device = 'cuda:0'
    cfg.sim.dt, cfg.decimation, cfg.sim.render_interval = .00025, 40, 40
    cfg.episode_length_s = 4.
    cfg.include_net = True
    cfg.robot.init_state.pos = (float(base['root_start_x_m']), 0., .323)
    cfg.shuttle_start = tuple(float(v) for v in launch['shuttle_start'])
    cfg.shuttle_velocity = tuple(float(v) for v in launch['shuttle_velocity'])
    cfg.event_goal_position = tuple(float(v) for v in base['virtual_ball_center_m'])
    cfg.event_goal_normal = tuple(float(v) for v in base['face_normal_world'])
    cfg.event_goal_time_s = float(launch['arrival_time_s'])

    class ReplayEnv(FullJointSmashEnv):
        def __init__(self, *a, **kw):
            self.net_trace = []
            super().__init__(*a, **kw)

        def _consume_physics_frame(self, frame):
            fresh = frame > self._last_physics_frame
            super()._consume_physics_frame(frame)
            # Denser measurements around contact/net crossing, without forces.
            if fresh and frame % 4 == 0 and 1.43 <= frame * self.physics_dt <= 1.57:
                self.net_trace.append((frame * self.physics_dt,
                    (self.shuttle.data.root_pos_w - self.scene.env_origins).cpu().numpy(),
                    self.shuttle.data.root_lin_vel_w.cpu().numpy()))

    env = ReplayEnv(cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
    if args.policy:
        from quadsmash.control.piecewise_affine import PiecewiseAffinePolicy
        learned=PiecewiseAffinePolicy(torch.load(args.policy,map_location='cpu',weights_only=False)).to(env.device)
    else:
        params = candidate['parameters'].to(env.device).repeat(args.num_envs, 1)
        arm = candidate['arm_pose'].to(env.device)
    default = env.robot.data.default_joint_pos[0, env.joint_ids]
    limits = env.robot.data.joint_pos_limits[0, env.joint_ids]
    obs, _ = wrapped.reset()
    observations, actions, poses, states = [], [], [], []
    try:
        with torch.inference_mode():
            for tick in range(400):
                action = (learned(obs['policy']) if args.policy else
                          dynamic_reach_action(obs, params, default, limits, arm))
                observations.append(obs['policy'].cpu())
                actions.append(action.cpu())
                if tick % 2 == 0:
                    data = env.robot.data
                    pos = data.body_link_pos_w - env.scene.env_origins[:, None, :]
                    poses.append(torch.cat((pos, data.body_link_quat_w), -1).cpu().numpy())
                    states.append(dict(time_s=tick * .01,
                        ball=(env.shuttle.data.root_pos_w-env.scene.env_origins).cpu().numpy(),
                        velocity=env.shuttle.data.root_lin_vel_w.cpu().numpy(),
                        tool=(data.body_pos_w[:,env.racket_body_id]-env.scene.env_origins).cpu().numpy(),
                        root=(data.root_pos_w-env.scene.env_origins).cpu().numpy(),
                        airborne=env.observed_airborne.cpu().numpy(),
                        contact=env.contact_seen.cpu().numpy()))
                obs, _, _, _ = wrapped.step(action)
        first = {}
        for row in env.smash_episode_records:
            first.setdefault(row['env_id'], row)
        if len(first) != args.num_envs:
            raise RuntimeError('Incomplete independent replay')
        rows = [first[i] for i in range(args.num_envs)]
        add_flight_search_diagnostics(rows)
        selected = max(range(args.num_envs), key=lambda i: search_candidate_key(rows[i]))
        trajectory = [dict(time_s=s['time_s'], ball_xyz=s['ball'][selected].tolist(),
            ball_v=s['velocity'][selected].tolist(), tool_xyz=s['tool'][selected].tolist(),
            root_xyz=s['root'][selected].tolist(), airborne=bool(s['airborne'][selected]),
            contact=bool(s['contact'][selected])) for s in states]
        summary = dict(kind='independent frozen near-net curriculum replay',
            net_geometry_version=cfg.net_geometry_version,
            controller='learned piecewise affine feedback' if args.policy else 'searched teacher',
            policy=str(args.policy) if args.policy else None,
            policy_sha256=hashlib.sha256(args.policy.read_bytes()).hexdigest() if args.policy else None,
            teacher_prefix_steps=0,
            formal_policy_acceptance=False, root_start_x_m=base['root_start_x_m'],
            candidate=str(args.candidate), candidate_sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
            seed=args.seed, episodes=len(rows), selected_video_lane=selected,
            selection='highest ordered result; selected example, not a random success-rate estimate',
            airborne_contact=sum(r['airborne_contact'] for r in rows),
            ordered_net=sum(r['ordered_crossed_net'] for r in rows),
            ordered_landing=sum(r['ordered_landed_target'] for r in rows),
            recovery=sum(r['recovery_success'] for r in rows),
            complete=sum(r['complete_smash'] for r in rows),
            numerical_failures=sum(r['numerical_failures'] for r in rows), best=rows[selected])
        (args.output/'episodes.json').write_text(json.dumps(rows, indent=2)+'\n')
        (args.output/'jump_episodes.json').write_text(json.dumps(env.episode_records, indent=2)+'\n')
        (args.output/'trajectory.json').write_text(json.dumps(trajectory)+'\n')
        np.save(args.output/'link_poses.npy', np.stack(poses)[:,selected])
        (args.output/'body_names.json').write_text(json.dumps(env.robot.body_names)+'\n')
        net = [dict(time_s=t, ball_xyz=p[selected].tolist(), ball_v=v[selected].tolist())
               for t,p,v in env.net_trace]
        (args.output/'net_trace.json').write_text(json.dumps(net)+'\n')
        torch.save(dict(observations=torch.stack(observations), actions=torch.stack(actions),
                        episodes=rows, seed=args.seed, candidate=summary['candidate_sha256']),
                   args.output/('policy_dataset.pt' if args.policy else 'teacher_dataset.pt'))
        (args.output/'metrics.json').write_text(json.dumps(summary, indent=2)+'\n')
        (args.output/'completed.json').write_text(json.dumps(dict(episodes=len(rows), env_steps=400*args.num_envs))+'\n')
        print(json.dumps(summary), flush=True)
    finally:
        env.close()


try:
    main()
except BaseException:
    args.output.mkdir(parents=True, exist_ok=True)
    error = traceback.format_exc()
    (args.output/'error.txt').write_text(error)
    print(error, file=sys.stderr, flush=True)
    raise
finally:
    app.close(skip_cleanup=True)
