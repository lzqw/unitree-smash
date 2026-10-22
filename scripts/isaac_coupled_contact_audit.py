#!/usr/bin/env python3
"""Full standard loaded robot impact/reaction diagnostic, reset-only launch."""
import argparse
import json
from pathlib import Path
import sys
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output",type=Path,required=True)
parser.add_argument("--dt",type=float)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.output.mkdir(parents=True,exist_ok=True)
app = AppLauncher(args).app
env = None
try:
    import torch
    from isaaclab.utils.math import quat_apply,quat_apply_inverse
    from quadsmash.tasks.stand.ball_env import BallEnv,BallEnvCfg
    cfg = BallEnvCfg()
    cfg.seed = 1927
    cfg.scene.num_envs = 4
    if args.dt is not None:
        cfg.sim.dt = args.dt
    args.dt = cfg.sim.dt
    cfg.decimation = round(.01/args.dt)
    cfg.robot.init_state.pos = (0.,0.,5.)
    cfg.episode_length_s = 10.
    env = BallEnv(cfg)
    with torch.inference_mode():
        env.reset()
        zero = torch.zeros((4,18),device=env.device)
        env.step(zero)
        tool = env.robot.data.body_pos_w[:,env.racket_body_id].clone()
        quat = env.robot.data.body_quat_w[:,env.racket_body_id].clone()
        normal = quat_apply(quat,torch.tensor([[1.,0.,0.]],device=env.device).expand(4,-1))
        tangent = quat_apply(quat,torch.tensor([[0.,1.,0.]],device=env.device).expand(4,-1))
        state = env.shuttle.data.default_root_state.clone()
        state[:,:3] = tool+.25*normal
        state[1::2,:3] += .5*tangent[1::2]
        state[:,7:10] = -normal*torch.tensor([10.,10.,120.,120.],device=env.device)[:,None]
        env.shuttle.write_root_state_to_sim(state)
        env.previous_ball_position.copy_(state[:,:3])
        env.previous_ball_velocity.copy_(state[:,7:10])
        mass = env.robot.root_physx_view.get_masses().to(env.device)
        inertia = env.robot.root_physx_view.get_inertias().to(env.device).reshape(4,-1,3,3)
        total_mass = mass.sum(1)+cfg.shuttle_mass_kg
        origin = env.scene.env_origins
        def momenta():
            data = env.robot.data
            robot_p = (mass[...,None]*data.body_com_lin_vel_w).sum(1)
            p = robot_p + cfg.shuttle_mass_kg*env.shuttle.data.root_lin_vel_w
            omega_b = quat_apply_inverse(data.body_quat_w,data.body_ang_vel_w)
            own = quat_apply(data.body_quat_w,(inertia@omega_b[...,None]).squeeze(-1))
            angular = (torch.linalg.cross(data.body_com_pos_w-origin[:,None],mass[...,None]*data.body_com_lin_vel_w)+own).sum(1)
            angular += torch.linalg.cross(env.shuttle.data.root_pos_w-origin,cfg.shuttle_mass_kg*env.shuttle.data.root_lin_vel_w)
            ball_inertia = .4*cfg.shuttle_mass_kg*cfg.shuttle_radius_m**2
            angular += ball_inertia*env.shuttle.data.root_ang_vel_w
            return p,angular,robot_p
        p0,l0,r0 = [x.clone() for x in momenta()]
        external = torch.zeros((4,3),device=env.device)
        angular_external = torch.zeros_like(external)
        gravity = torch.tensor(cfg.sim.gravity,device=env.device)
        rows = []
        env._pre_physics_step(zero)
        for step in range(round(.04/args.dt)):
            env._sim_step_counter += 1
            env._apply_action()
            force = env.previous_drag.clone()
            external += (total_mass[:,None]*gravity+force)*args.dt
            torque = torch.linalg.cross(env.robot.data.body_com_pos_w-origin[:,None],mass[...,None]*gravity).sum(1)
            torque += torch.linalg.cross(env.shuttle.data.root_pos_w-origin,cfg.shuttle_mass_kg*gravity+force)
            angular_external += torque*args.dt
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(args.dt)
            env._consume_physics_frame(env._sim_step_counter)
            p,l,r = momenta()
            rows.append(dict(time=(step+1)*args.dt,contact_count=env.contact_events.count.tolist(),
                sweep_misses=env.sweep_misses.tolist(),sweep_unresolved=env.sweep_unresolved.tolist(),
                force=env._latest_racket_force.tolist(),body_contact=env.body_contact_seen.tolist(),
                linear_error=(p-p0-external).norm(dim=-1).tolist(),angular_error=(l-l0-angular_external).norm(dim=-1).tolist(),
                robot_reaction=(r-r0-mass.sum(1)[:,None]*gravity*((step+1)*args.dt)).tolist(),
                ball_velocity=env.shuttle.data.root_lin_vel_w.tolist()))
        result = dict(backend="Isaac Lab / GPU PhysX",dt=args.dt,asset=cfg.robot.spawn.usd_path,
            total_robot_mass=mass.sum(1).tolist(),contact_count=env.contact_events.count.tolist(),
            sweep_misses=env.sweep_misses.tolist(),sweep_unresolved=env.sweep_unresolved.tolist(),rows=rows)
        (args.output/"coupling.json").write_text(json.dumps(result,indent=2)+"\n")
        print("COUPLING "+json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
except BaseException:
    (args.output/"error.txt").write_text(traceback.format_exc())
    raise
finally:
    if env is not None:
        env.close()
    app.close()
