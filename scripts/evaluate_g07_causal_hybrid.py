#!/usr/bin/env python3
"""Independent real-ball evaluation of the learned hybrid on a frozen panel."""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--launch-panel', type=Path, required=True)
parser.add_argument('--calibration', type=Path)
parser.add_argument('--protocol', type=Path)
parser.add_argument('--policy-lock', type=Path)
parser.add_argument('--fixed-phase-ablation', action='store_true')
parser.add_argument('--physics-dt', type=float, default=.0000625)
parser.add_argument('--episode-seconds', type=float, default=8.5)
parser.add_argument('--seed', type=int, default=717101)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from quadsmash.control.learned_hybrid import LearnedHybridPolicy
from quadsmash.tasks.stand.ball_env import BallEnvCfg
from quadsmash.tasks.stand.smash_env_v3 import FullJointSmashEnv
from quadsmash.tasks.stand.net_contact_monitor import NetContactMonitor, add_net_contact_sensors


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    panel = json.loads(args.launch_panel.read_text())
    scenarios = panel['scenarios']
    count = len(scenarios)
    if count < 1:
        raise ValueError('Empty evaluation panel')
    calibration = json.loads(args.calibration.read_text()) if args.calibration else checkpoint.get('phase_calibration')
    if args.fixed_phase_ablation:
        calibration = None
    formal_eligible = False
    protocol = None
    if args.protocol:
        if not args.policy_lock or args.fixed_phase_ablation:
            raise ValueError('Frozen evaluation needs a policy lock and the complete controller')
        protocol = json.loads(args.protocol.read_text())
        lock = json.loads(args.policy_lock.read_text())
        references = [p for p in protocol['panels'] if (ROOT/p['path']).resolve() == args.launch_panel.resolve()]
        if len(references) != 1 or references[0]['sha256'] != digest(args.launch_panel) or references[0]['episodes'] != count:
            raise ValueError('Panel differs from the frozen protocol')
        if lock['protocol_sha256'] != digest(args.protocol) or lock['policy_sha256'] != digest(args.checkpoint):
            raise ValueError('Policy/protocol differs from the evaluation lock')
        for path, expected in lock['source_hashes'].items():
            if digest(ROOT/path) != expected:
                raise ValueError('Evaluation implementation changed after policy lock: '+path)
        if (protocol['physics_dt_s'] != args.physics_dt or protocol['episode_seconds'] != args.episode_seconds
                or protocol['asset_sha256'] != digest(ROOT/'assets/quadsmash_standard_v2.usd')
                or protocol['acceptance_contract_sha256'] != digest(ROOT/'configs/acceptance.json')
                or protocol['initialization_contract_sha256'] != digest(ROOT/'configs/evaluation_protocol_v2.json')):
            raise ValueError('Frozen physics, hardware or acceptance contract mismatch')
        calibration_digest = hashlib.sha256(json.dumps(calibration, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if calibration_digest != protocol['calibration_canonical_sha256']:
            raise ValueError('Causal calibration differs from frozen protocol')
        formal_eligible = True
    physics = json.loads((ROOT/'configs/shuttle_physics_v2.json').read_text())
    cfg = BallEnvCfg()
    cfg.scene.num_envs = count
    cfg.seed = args.seed
    cfg.sim.device = 'cuda:0'
    cfg.sim.dt = args.physics_dt
    cfg.decimation = round(.01/args.physics_dt)
    cfg.sim.render_interval = cfg.decimation
    if abs(cfg.sim.dt*cfg.decimation-.01) > 1e-9:
        raise ValueError('Physics dt must divide the original 10ms control period')
    cfg.episode_length_s = args.episode_seconds
    cfg.include_net = True
    cfg.robot.init_state.pos = (0., 0., .323)
    cfg.event_goal_enabled = True
    cfg.event_goal_position = (checkpoint['walk_waypoint_x'], 0., .3)
    cfg.event_goal_normal = tuple(checkpoint['result']['face_normal_world'])
    cfg.event_goal_time_s = checkpoint['result']['event_time_s']
    starts = np.array([s['shuttle_start'] for s in scenarios])
    velocities = np.array([s['shuttle_velocity'] for s in scenarios])

    class PanelEnv(NetContactMonitor, FullJointSmashEnv):
        def _setup_scene(self):
            super()._setup_scene()
            add_net_contact_sensors(self)

        def _write_shuttle_reset(self, ids):
            state = self.shuttle.data.default_root_state[ids].clone()
            state[:, :3] = self.scene.env_origins[ids]+torch.tensor(starts, device=self.device, dtype=torch.float32)[ids]
            state[:, 7:10] = torch.tensor(velocities, device=self.device, dtype=torch.float32)[ids]
            self.shuttle.write_root_state_to_sim(state, env_ids=ids)
            if self.contact_events is not None:
                self.previous_ball_position[ids] = state[:, :3]
                self.previous_ball_velocity[ids] = state[:, 7:10]
                self.previous_drag[ids] = 0.

    env = PanelEnv(cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
    default = env.robot.data.default_joint_pos[0, env.joint_ids]
    limits = env.robot.data.joint_pos_limits[0, env.joint_ids]
    policy = LearnedHybridPolicy(checkpoint, ROOT/'configs/g07_go2_walk_transfer.json',
                                  count, env.device, default, limits, physics, calibration)
    try:
        obs, _ = wrapped.reset()
        ids = torch.arange(count, device=env.device)
        root = env.robot.data.root_state_w.clone()
        q = env.robot.data.default_joint_pos.clone()
        qd = torch.zeros_like(q)
        initial_rows = []
        for index, scenario in enumerate(scenarios):
            reset = scenario.get('robot_reset', {})
            root[index, :2] += torch.tensor(reset.get('xy_m', [0., 0.]), device=env.device)
            yaw = float(reset.get('yaw_rad', 0.))
            root[index, 3:7] = torch.tensor([np.cos(yaw/2), 0., 0., np.sin(yaw/2)], device=env.device)
            root[index, 7:] = 0.
            q[index, env.joint_ids] += torch.tensor(reset.get('joint_offset_rad', [0.]*18), device=env.device)
            qd[index, env.joint_ids] = torch.tensor(reset.get('joint_velocity_rad_s', [0.]*18), device=env.device)
        with torch.inference_mode():
            env.robot.write_root_state_to_sim(root, env_ids=ids)
            env.robot.write_joint_state_to_sim(q, qd, env_ids=ids)
            env.sim.forward()
            env.policy_history.reset(ids)
            obs = wrapped.get_observations()
        for index in range(count):
            initial_rows.append(dict(scenario=index, root_state=root[index].cpu().tolist(),
                                     joint_position=q[index].cpu().tolist(), joint_velocity=qd[index].cpu().tolist(),
                                     ground_init=bool(abs(float(root[index, 2])-.323) < 1e-5)))
        policy.initialize(obs)
        with torch.inference_mode():
            env.goal_position_w.copy_(env.scene.env_origins+policy.waypoints)
            if calibration is not None:
                env.goal_time_s.copy_(policy.arrival)
            obs = wrapped.get_observations()
        (args.output/'causal_plan.json').write_text(json.dumps(policy.plan_log, indent=2)+'\n')
        manifest = dict(protocol=protocol['version'] if protocol else panel['version'], panel=panel.get('panel_id'),
                        formal_policy_acceptance=False, episodes=count, ground_initialization_only=True,
                        formal_evaluation_eligible=formal_eligible,
                        protocol_sha256=digest(args.protocol) if args.protocol else None,
                        policy_lock_sha256=digest(args.policy_lock) if args.policy_lock else None,
                        checkpoint_sha256=digest(args.checkpoint), launch_panel_sha256=digest(args.launch_panel),
                        calibration_sha256=digest(args.calibration) if args.calibration else None,
                        source_git_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                        asset_sha256=digest(ROOT/'assets/quadsmash_standard_v2.usd'),
                        physics_dt_s=cfg.sim.dt, control_dt_s=.01, episode_seconds=cfg.episode_length_s,
                        net_geometry_version=cfg.net_geometry_version,
                        controller='frozen learned walking + learned PWA strike + frozen learned standing + initial-state predictor',
                        teacher_prefix_steps=0, teacher_action_function_calls=0,
                        causal_phase=calibration is not None, fixed_phase_ablation=args.fixed_phase_ablation,
                        command=sys.argv)
        (args.output/'eval_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        ticks = round(cfg.episode_length_s/.01)
        traces = [[] for _ in range(min(2, count))]
        first_episode_finished = torch.zeros(count, dtype=torch.bool, device=env.device)
        with torch.inference_mode():
            for tick in range(ticks):
                env.goal_position_w.copy_(env.scene.env_origins+policy.command_positions(obs))
                action = policy.action(obs)
                if not torch.isfinite(action).all():
                    raise RuntimeError('Nonfinite policy action')
                if tick % 2 == 0:
                    for index, trace in enumerate(traces):
                        if first_episode_finished[index]:
                            continue
                        trace.append(dict(time_s=tick*.01,
                            ball_xyz=(env.shuttle.data.root_pos_w[index]-env.scene.env_origins[index]).cpu().tolist(),
                            tool_xyz=(env.robot.data.body_pos_w[index, env.racket_body_id]-env.scene.env_origins[index]).cpu().tolist(),
                            root_xyz=(env.robot.data.root_pos_w[index]-env.scene.env_origins[index]).cpu().tolist(),
                            airborne=bool(env.observed_airborne[index]), contact=bool(env.contact_seen[index])))
                obs, _, dones, _ = wrapped.step(action)
                first_episode_finished |= dones.bool()
                if (tick+1)%50==0:print(json.dumps(dict(control_steps=tick+1,total_steps=ticks,simulation_time_s=(tick+1)*.01)),flush=True)
        first, nets = {}, {}
        for row in env.smash_episode_records:
            first.setdefault(row['env_id'], row)
        for row in env.net_episode_records:
            nets.setdefault(row['env_id'], row)
        if len(first) != count or len(nets) != count:
            raise RuntimeError('Missing first-episode outcomes')
        rows = [dict(first[i], **{k: v for k, v in nets[i].items() if k != 'env_id'},
                     scenario=i, panel=panel.get('panel_id'), ground_init=initial_rows[i]['ground_init']) for i in range(count)]
        air = sum(r['airborne_contact'] for r in rows)
        recovered = sum(r['airborne_contact'] and r['recovery_success'] and not r['robot_net_contact'] for r in rows)
        metrics = dict(sample_count=count, episodes=count,
            ground_init_fraction=sum(r['ground_init'] for r in rows)/count,
            airborne_contact_rate=air/count,
            recovery_given_airborne_contact=recovered/air if air else 0.,
            complete_smash_success_rate=sum(r['complete_smash'] and not r['robot_net_contact'] and r['ground_init'] for r in rows)/count,
            ordered_net=sum(r['ordered_crossed_net'] and not r['robot_net_contact'] for r in rows),
            ordered_landing=sum(r['ordered_landed_target'] and not r['robot_net_contact'] for r in rows),
            robot_net_contact=sum(r['robot_net_contact'] for r in rows),
            numerical_failures=sum(r['numerical_failures'] for r in rows),
            formal_policy_acceptance=False, formal_evaluation_eligible=formal_eligible,
            protocol=protocol['version'] if protocol else panel['version'], panel=panel.get('panel_id'),
            env_steps=count*ticks)
        (args.output/'episodes.json').write_text(json.dumps(rows, indent=2)+'\n')
        with (args.output/'episodes.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (args.output/'initial_states.json').write_text(json.dumps(initial_rows, indent=2)+'\n')
        (args.output/'traces_fixed_scenarios_0_1.json').write_text(json.dumps(traces)+'\n')
        (args.output/'jump_episodes.json').write_text(json.dumps(env.episode_records, indent=2)+'\n')
        (args.output/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
        saved = dict(checkpoint)
        if calibration is not None:
            saved['phase_calibration'] = calibration
            saved['policy_schema'] = 'hybrid-learned-causal-smash-v2'
        torch.save(saved, args.output/'evaluated_policy.pt')
        (args.output/'completed.json').write_text(json.dumps(dict(episodes=count, env_steps=count*ticks))+'\n')
        print(json.dumps(metrics), flush=True)
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
