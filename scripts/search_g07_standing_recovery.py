#!/usr/bin/env python3
"""Keep a valid strike fixed and search learned-actor landing/standing transfer."""
import argparse
import json
import signal
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--candidate', type=Path, required=True)
parser.add_argument('--num-envs', type=int, default=128)
parser.add_argument('--generations', type=int, default=16)
parser.add_argument('--seed', type=int, default=716701)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from quadsmash.control.composed_reach_teacher import composed_reach_action
from quadsmash.control.go2_walk_transfer import Go2WalkTransfer
from quadsmash.control.go2_standing_transfer import Go2StandingTransfer
from quadsmash.tasks.stand.jump_env import JumpEnvCfg
from quadsmash.tasks.stand.loaded_walk_env import LoadedWalkEnv

stopping = False


def stop(*_):
    global stopping
    stopping = True


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)


def main():
    args.output.mkdir(parents=True, exist_ok=True)
    candidate = torch.load(args.candidate, map_location='cpu', weights_only=False)
    cfg = JumpEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.sim.device = 'cuda:0'
    cfg.sim.dt, cfg.decimation, cfg.sim.render_interval = .002, 5, 5
    cfg.episode_length_s = candidate['walk_seconds']+4.
    cfg.event_goal_enabled = True
    cfg.event_goal_position = (candidate['walk_waypoint_x'], 0., .3)
    env = LoadedWalkEnv(cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
    device = env.device
    manifest = ROOT/'configs/g07_go2_walk_transfer.json'
    walk = Go2WalkTransfer(manifest, args.num_envs, device)
    standing = Go2StandingTransfer(manifest, args.num_envs, device)
    default = env.robot.data.default_joint_pos[0, env.joint_ids]
    limits = env.robot.data.joint_pos_limits[0, env.joint_ids]
    strike = candidate['parameters'].to(device).repeat(args.num_envs, 1)
    walk_params = candidate['walk_parameters'].to(device).repeat(args.num_envs, 1)
    initial_arm = candidate['arm_initial_pose'].to(device)
    arm = candidate['arm_pose'].to(device)
    recovery_arm = initial_arm.clone()
    if strike.shape[1] == 44:
        recovery_arm[1] -= strike[0, 43]
    target = torch.tensor(candidate['result']['virtual_ball_center_m'], device=device)
    walk_seconds = candidate['walk_seconds']
    walk_ticks = round(walk_seconds/.01)
    ticks = round(cfg.episode_length_s/.01)
    low = torch.tensor([-.2, -.3, -.35, -.8, .5, 1.42], device=device)
    high = torch.tensor([.5, .3, .6, .8, 1.4, 1.85], device=device)
    mean = torch.cat((walk_params[0, 3:8], torch.tensor([1.5], device=device))).clamp(low, high)
    sigma = (high-low)*.16
    generator = torch.Generator(device=device).manual_seed(args.seed)
    history, best_score = [], -float('inf')
    try:
        for generation in range(args.generations):
            params = (mean+sigma*torch.randn((args.num_envs, 6), device=device,
                                             generator=generator)).clamp(low, high)
            params[0] = mean
            global_count = args.num_envs//5
            params[-global_count:] = low+(high-low)*torch.rand((global_count, 6), device=device,
                                                              generator=generator)
            env._episode_has_stepped.zero_()
            env._metrics_started = False
            env.episode_records.clear()
            env.net_episode_records.clear()
            env.goal_position_w.copy_(env.scene.env_origins+torch.tensor(cfg.event_goal_position, device=device))
            walk.reset()
            standing.reset()
            obs, _ = wrapped.reset()
            finished = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
            prefix_checked = finished.clone()
            prefix_valid = finished.clone()
            final = torch.zeros((args.num_envs, 3), device=device)
            velocity = final.clone()
            tilt = torch.zeros(args.num_envs, device=device)
            with torch.inference_mode():
                for tick in range(ticks):
                    active = ~finished
                    local_time = obs['policy'][:, 775]-walk_seconds
                    check = active & (~prefix_checked) & (local_time >= candidate['result']['local_event_time_s'])
                    prefix_valid[check] = (env.observed_airborne & (~env.net_robot_seen)
                                           & (~env.body_ground_seen))[check]
                    prefix_checked |= check
                    final[active] = (env.robot.data.root_pos_w-env.scene.env_origins)[active]
                    velocity[active] = env.robot.data.root_link_lin_vel_w[active]
                    tilt[active] = env.robot.data.projected_gravity_b[active, :2].norm(dim=1)
                    if tick < walk_ticks:
                        action = walk.action(obs, walk_params, default, limits)
                    else:
                        if tick == walk_ticks:
                            env.goal_position_w.copy_(env.scene.env_origins+target)
                        local_obs = {'policy': obs['policy'].clone()}
                        local_obs['policy'][:, 775] -= walk_seconds
                        action = composed_reach_action(local_obs, strike, default, limits, arm, initial_arm)
                        use_standing = active & (local_time >= params[:, 5])
                        recovery_action = standing.action(obs, params[:, :5], default, limits,
                                                          recovery_arm, use_standing)
                        action = torch.where(use_standing[:, None], recovery_action, action)
                    obs, _, done, _ = wrapped.step(action)
                    finished |= done.bool()
            jumps, nets = {}, {}
            for row in env.episode_records:
                jumps.setdefault(row['env_id'], row)
            for row in env.net_episode_records:
                nets.setdefault(row['env_id'], row)
            if len(jumps) != args.num_envs or len(nets) != args.num_envs:
                raise RuntimeError('Incomplete recovery telemetry')
            net_bad = torch.tensor([nets[i]['robot_net_contact'] for i in range(args.num_envs)], device=device)
            body_bad = torch.tensor([jumps[i]['body_landing'] for i in range(args.num_envs)], device=device)
            success = torch.tensor([jumps[i]['success'] for i in range(args.num_envs)], device=device)
            success &= prefix_valid & (~net_bad)
            quiet = torch.tensor([jumps[i]['recovery_s'] for i in range(args.num_envs)], device=device)
            force = torch.tensor([jumps[i]['maximum_body_ground_force_n'] for i in range(args.num_envs)], device=device)
            scores = (100*success.float()+20*(~body_bad).float()+10*quiet.clamp(max=.8)
                      -50*(~prefix_valid).float()-20*net_bad.float()-torch.log1p(force)
                      -3*tilt-2*velocity.norm(dim=1))
            elite = scores.topk(max(8, args.num_envs//8)).indices
            mean = .25*mean+.75*params[elite].mean(0)
            sigma = (.25*sigma+.75*params[elite].std(0)).clamp(min=(high-low)*.025)
            index = int(scores.argmax())
            result = dict(generation=generation, score=float(scores[index]),
                          recovery_parameters=params[index].cpu().tolist(), recovery_success=bool(success[index]),
                          successes=int(success.sum()), prefix_valid=bool(prefix_valid[index]),
                          prefix_valid_lanes=int(prefix_valid.sum()), body_landing=bool(body_bad[index]),
                          robot_net_contact=bool(net_bad[index]), root_final_m=final[index].cpu().tolist(),
                          root_velocity_mps=velocity[index].cpu().tolist(), jump_metrics=jumps[index],
                          kind='no-ball fixed strike plus frozen learned standing actor; not real-ball complete success')
            history.append(result)
            if float(scores[index]) > best_score:
                best_score = float(scores[index])
                saved = dict(candidate, recovery_parameters=params[index, :5].cpu(),
                             recovery_start_s=float(params[index, 5]), recovery_arm_target=recovery_arm.cpu(),
                             recovery_search_result=result,
                             controller_system='frozen learned walking + searched strike + frozen learned standing')
                torch.save(saved, args.output/'best_candidate.pt')
            torch.save(dict(parameters=params.cpu(), scores=scores.cpu(), jumps=jumps, nets=nets,
                            mean=mean.cpu(), sigma=sigma.cpu(), rng=generator.get_state().cpu()),
                       args.output/f'population_{generation:03d}.pt')
            (args.output/'metrics.json').write_text(json.dumps(dict(history=history,
                 best=max(history, key=lambda row: row['score']), env_steps=(generation+1)*args.num_envs*ticks), indent=2)+'\n')
            print(json.dumps(result), flush=True)
            if stopping:
                break
        if len(history) == args.generations:
            (args.output/'completed.json').write_text(json.dumps(dict(generations=len(history)))+'\n')
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
