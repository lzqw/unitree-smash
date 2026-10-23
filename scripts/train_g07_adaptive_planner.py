#!/usr/bin/env python3
"""Optimize causal phase/position features using actual ordered ball outcomes."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--panel', type=Path, required=True)
p.add_argument('--calibration', type=Path, required=True)
p.add_argument('--groups', type=int, default=16)
p.add_argument('--case-stride', type=int, default=4)
p.add_argument('--generations', type=int, default=6)
p.add_argument('--physics-dt', type=float, default=.00025)
p.add_argument('--seed', type=int, default=717501)
p.add_argument('--target-complete', type=float, default=.875)
p.add_argument('--resume-population', type=Path)
p.add_argument('--arm-pitch-basis', type=Path)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(a).app

import numpy as np
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from quadsmash.control.learned_hybrid import LearnedHybridPolicy
from quadsmash.control.smash_search import add_flight_search_diagnostics
from quadsmash.tasks.stand.ball_env import BallEnvCfg
from quadsmash.tasks.stand.smash_env_v3 import FullJointSmashEnv
from quadsmash.tasks.stand.net_contact_monitor import NetContactMonitor, add_net_contact_sensors

stopping = False


def stop(*_):
    global stopping
    stopping = True


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)


def main():
    a.output.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    arm_mode = a.arm_pitch_basis is not None
    if arm_mode:
        basis = json.loads(a.arm_pitch_basis.read_text())['arm_pitch_basis']
        checkpoint = dict(checkpoint, arm_pitch_basis=torch.tensor(basis, dtype=torch.float32))
    source = json.loads(a.panel.read_text())
    indices = list(range(0, len(source['scenarios']), a.case_stride))
    scenarios = [source['scenarios'][i] for i in indices]
    if any('robot_reset' in scenario for scenario in scenarios):
        raise ValueError('This training entry uses nominal robot initialization; reset perturbations must not be silently ignored')
    cases, n = len(scenarios), len(scenarios)*a.groups
    calibration = json.loads(a.calibration.read_text())
    physics = json.loads((ROOT/'configs/shuttle_physics_v2.json').read_text())
    starts = torch.tensor([s['shuttle_start'] for s in scenarios], device='cuda:0', dtype=torch.float32)
    velocities = torch.tensor([s['shuttle_velocity'] for s in scenarios], device='cuda:0', dtype=torch.float32)
    case_ids = torch.arange(n, device='cuda:0') % cases
    cfg = BallEnvCfg()
    cfg.scene.num_envs = n
    cfg.seed = a.seed
    cfg.sim.device = 'cuda:0'
    cfg.sim.dt, cfg.decimation = a.physics_dt, round(.01/a.physics_dt)
    cfg.sim.render_interval = cfg.decimation
    cfg.episode_length_s = 8.5
    cfg.include_net = True
    from quadsmash.physics.net_geometry import PROFILE_VERSION
    cfg.net_geometry_version = source.get('net_geometry_version', cfg.net_geometry_version)
    if cfg.net_geometry_version == PROFILE_VERSION:
        cfg.net_side_height_m = 1.55
    cfg.event_goal_enabled = True
    cfg.event_goal_position = (checkpoint['walk_waypoint_x'], 0., .3)
    cfg.event_goal_normal = tuple(checkpoint['result']['face_normal_world'])

    class TrainingEnv(NetContactMonitor, FullJointSmashEnv):
        def _setup_scene(self):
            super()._setup_scene()
            add_net_contact_sensors(self)

        def _write_shuttle_reset(self, ids):
            state = self.shuttle.data.default_root_state[ids].clone()
            state[:, :3] = self.scene.env_origins[ids]+starts[case_ids[ids]]
            state[:, 7:10] = velocities[case_ids[ids]]
            self.shuttle.write_root_state_to_sim(state, env_ids=ids)
            if self.contact_events is not None:
                self.previous_ball_position[ids] = state[:, :3]
                self.previous_ball_velocity[ids] = state[:, 7:10]
                self.previous_drag[ids] = 0.

    env = TrainingEnv(cfg)
    env.capture_sweep_failures, env.sweep_failure_records = True, []
    wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
    device = env.device
    policy = LearnedHybridPolicy(checkpoint, ROOT/'configs/g07_go2_walk_transfer.json', n, device,
        env.robot.data.default_joint_pos[0, env.joint_ids], env.robot.data.joint_pos_limits[0, env.joint_ids],
        physics, calibration)
    low = torch.tensor([-.04, -.08, -.02, -1.5, -.03, -.06, -.01, -.5, -.15, -.04, .7], device=device)
    high = torch.tensor([.04, .08, .02, 1.5, .03, .06, .01, .5, .15, .04, 1.3], device=device)
    baseline = torch.tensor([0.]*10+[1.], device=device)
    mean = checkpoint.get('planner_parameters', baseline.cpu()).to(device).clone().clamp(low, high)
    if arm_mode:
        low = torch.tensor([-.03, -.06, -.01, -.5], device=device)
        high = torch.tensor([.015, .01, .05, 1.], device=device)
        baseline = torch.zeros(4, device=device)
        mean = checkpoint.get('arm_pitch_parameters', torch.tensor([0., -.03, .02, .3])).to(device).clone().clamp(low, high)
        policy.arm_pitch_basis.copy_(checkpoint['arm_pitch_basis'].to(device))
        policy.has_arm_residual = True
    sigma = (high-low)*.15
    generator = torch.Generator(device=device).manual_seed(a.seed)
    if a.resume_population:
        resume=torch.load(a.resume_population,map_location='cpu',weights_only=False)
        mean.copy_(resume['mean'].to(device));sigma.copy_(resume['sigma'].to(device));generator.set_state(resume['rng'].cpu())
    history, best_score, converged = [], -float('inf'), False
    (a.output/'training_manifest.json').write_text(json.dumps(dict(
        kind='real-ball optimization of public-input arm residual' if arm_mode else 'real-ball optimization of public-input causal planner; formal panels unused',
        source_panel_sha256=hashlib.sha256(a.panel.read_bytes()).hexdigest(), source_scenario_indices=indices,
        groups=a.groups, cases_per_group=cases, physics_dt_s=a.physics_dt, target_complete=a.target_complete,
        net_geometry_version=cfg.net_geometry_version, net_center_height_m=cfg.net_height_m,
        net_side_height_m=cfg.net_side_height_m,
        checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()), indent=2)+'\n')
    try:
        for generation in range(a.generations):
            params = (mean+sigma*torch.randn((a.groups, len(low)), device=device, generator=generator)).clamp(low, high)
            params[0] = mean
            params[1] = baseline
            if generation==0 and a.groups>=3 and not arm_mode:
                params[2]=baseline
                params[2,1]=.03;params[2,2]=.01;params[2,5]=.02;params[2,9]=.015
            global_count = max(1, a.groups//4)
            params[-global_count:] = low+(high-low)*torch.rand((global_count, len(low)), device=device, generator=generator)
            order = torch.randperm(n, device=device, generator=generator)
            group_ids = order//cases
            case_ids.copy_(order % cases)
            if arm_mode:
                policy.arm_pitch_parameters.copy_(params[group_ids])
            else:
                policy.planner_parameters.copy_(params[group_ids])
            env._episode_has_stepped.zero_()
            env._metrics_started = False
            env.smash_episode_records.clear()
            env.net_episode_records.clear()
            env.episode_records.clear()
            env.sweep_failure_records.clear()
            with torch.inference_mode():
                env.goal_position_w.copy_(env.scene.env_origins+torch.tensor(cfg.event_goal_position, device=device))
            obs, _ = wrapped.reset()
            policy.initialize(obs)
            with torch.inference_mode():
                env.goal_position_w.copy_(env.scene.env_origins+policy.waypoints)
                env.goal_time_s.copy_(policy.arrival)
                obs = wrapped.get_observations()
                for tick in range(850):
                    env.goal_position_w.copy_(env.scene.env_origins+policy.command_positions(obs))
                    obs, _, _, _ = wrapped.step(policy.action(obs))
                    if (tick+1)%100==0:print(json.dumps(dict(generation=generation,control_steps=tick+1)),flush=True)
            first, nets = {}, {}
            for row in env.smash_episode_records:
                first.setdefault(row['env_id'], row)
            for row in env.net_episode_records:
                nets.setdefault(row['env_id'], row)
            if len(first) != n or len(nets) != n:
                raise RuntimeError('Missing physical training outcomes')
            groups = group_ids.cpu().tolist()
            source_cases = case_ids.cpu().tolist()
            rows = [dict(first[i], **{k: v for k, v in nets[i].items() if k != 'env_id'},
                         planner_group=groups[i], source_scenario=indices[source_cases[i]]) for i in range(n)]
            add_flight_search_diagnostics(rows)
            summaries = []
            for group in range(a.groups):
                batch = [r for r in rows if r['planner_group'] == group]
                free = lambda r: not r['robot_net_contact'] and not r['numerical_failures']
                full = sum(r['complete_smash'] and free(r) for r in batch)/cases
                target = sum(r['ordered_landed_target'] and free(r) for r in batch)/cases
                net = sum(r['ordered_crossed_net'] and free(r) for r in batch)/cases
                rec = sum(r['recovery_success'] and free(r) for r in batch)/cases
                contacts = sum(r['effective_face_contact'] for r in batch)/cases
                collisions = sum(r['robot_net_contact'] for r in batch)/cases
                faults = sum(r['numerical_failures'] for r in batch)
                violation = [r.get('coupled_flight_violation', 6.) for r in batch]
                auxiliary = sum(min(v, 6.) if math.isfinite(v) else 6. for v in violation)/cases
                score = 20*full+5*target+3*net+rec+.5*contacts-8*collisions-5*faults/cases-.15*auxiliary
                summaries.append(dict(group=group, score=score, complete_rate=full, target_rate=target,
                                      net_rate=net, recovery_rate=rec, robot_net_rate=collisions, numerical_failures=faults))
            scores = torch.tensor([r['score'] for r in summaries], device=device)
            index = int(scores.argmax())
            parameter_name = 'arm_pitch_parameters' if arm_mode else 'planner_parameters'
            result = dict(generation=generation, **{parameter_name: params[index].cpu().tolist()}, **summaries[index])
            history.append(result)
            if result['score'] > best_score:
                best_score = result['score']
                saved = dict(checkpoint, **{parameter_name: params[index].cpu()}, planner_training_result=result,
                             planner_training_manifest='training_manifest.json')
                torch.save(saved, a.output/'best_candidate.pt')
            elite = scores.topk(max(2, a.groups//4)).indices
            mean.copy_(.25*mean+.75*params[elite].mean(0))
            sigma.copy_((.25*sigma+.75*params[elite].std(0)).clamp(min=(high-low)*.025))
            torch.save(dict(parameters=params.cpu(), mean=mean.cpu(), sigma=sigma.cpu(), rng=generator.get_state().cpu(),
                            group_scores=summaries), a.output/f'population_{generation:03d}.pt')
            (a.output/f'episodes_{generation:03d}.json').write_text(json.dumps(rows, indent=2)+'\n')
            (a.output/f'unresolved_sweep_inputs_{generation:03d}.json').write_text(json.dumps(env.sweep_failure_records, indent=2)+'\n')
            (a.output/'metrics.json').write_text(json.dumps(dict(history=history, best=max(history, key=lambda r: r['score']),
                env_steps=(generation+1)*n*850, formal_acceptance=False), indent=2)+'\n')
            print(json.dumps(result), flush=True)
            converged = result['complete_rate'] >= a.target_complete and result['recovery_rate'] >= .875 and not result['numerical_failures']
            if stopping or converged:
                break
        if not stopping:
            (a.output/'completed.json').write_text(json.dumps(dict(generations=len(history),
                termination='declared development target reached' if converged else 'generation cap reached'))+'\n')
    finally:
        env.close()


try:
    main()
except BaseException:
    a.output.mkdir(parents=True, exist_ok=True)
    error = traceback.format_exc()
    (a.output/'error.txt').write_text(error)
    print(error, file=sys.stderr, flush=True)
    raise
finally:
    app.close(skip_cleanup=True)
