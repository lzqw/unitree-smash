#!/usr/bin/env python3
"""Native finite-mass racket fixture: raw contacts, momentum and flight traces.

This supplements the full-load articulation test; a fixture alone cannot pass G02.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import traceback
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--dt", type=float, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
app = AppLauncher(args).app
try:
    import numpy as np
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObject, RigidObjectCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sensors import ContactSensor, ContactSensorCfg
    from isaaclab.utils import configclass
    from isaacsim.core.utils.stage import get_current_stage
    from quadsmash.physics.native import physics_contract, configure_ccd, ContactEpisodes, sphere_box_gap, drag_force
    physics = physics_contract()
    cases = []
    for kind in ("gravity", "drag", "wind"):
        for speed in (4.,20.,80.):
            cases.append(dict(kind=kind,speed=speed,y=1.))
    for speed in (4.,10.,30.,60.,120.,200.):
        for y in (0.,.08,.129,.14):
            cases.append(dict(kind="impact",speed=speed,y=y,boundary=y>.11))
    for vx in (-3.,-1.,0.,1.,3.):
        cases.append(dict(kind="moving",speed=20.,y=0.,racket_vx=vx))
    for omega in (-40.,-20.,-5.,5.,20.,40.):
        for y in (-.06,.06):
            cases.append(dict(kind="rotating",speed=10.,y=y,omega=omega))
    cases.append(dict(kind="recontact",speed=20.,y=0.))
    n = len(cases)
    material = sim_utils.RigidBodyMaterialCfg(restitution=physics["restitution"],
        static_friction=physics["friction"], dynamic_friction=physics["friction"], restitution_combine_mode="max")
    rigid = sim_utils.RigidBodyPropertiesCfg(disable_gravity=True,linear_damping=0.,angular_damping=0.,
        max_linear_velocity=200.,max_angular_velocity=math.degrees(200.),sleep_threshold=0.,
        solver_position_iteration_count=8,solver_velocity_iteration_count=2)
    collision = sim_utils.CollisionPropertiesCfg(contact_offset=.0001,rest_offset=0.)
    @configclass
    class SceneCfg(InteractiveSceneCfg):
        ball = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Ball",spawn=sim_utils.SphereCfg(
            radius=physics["collision_radius_m"],mass_props=sim_utils.MassPropertiesCfg(mass=physics["mass_kg"]),
            rigid_props=rigid,collision_props=collision,physics_material=material,activate_contact_sensors=True))
        blade = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Blade",spawn=sim_utils.CuboidCfg(
            size=(.015,.22,.22),mass_props=sim_utils.MassPropertiesCfg(mass=.09),
            rigid_props=rigid,collision_props=collision,physics_material=material))
        second = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Second",spawn=sim_utils.CuboidCfg(
            size=(.015,.22,.22),mass_props=sim_utils.MassPropertiesCfg(mass=.09),
            rigid_props=rigid,collision_props=collision,physics_material=material))
        contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Ball",update_period=0.,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Blade","{ENV_REGEX_NS}/Second"],
            track_contact_points=True,max_contact_data_count_per_prim=16)
    sim_cfg = sim_utils.SimulationCfg(dt=args.dt,device=args.device or "cuda:0",gravity=(0.,0.,0.))
    sim_cfg.physx.enable_ccd = False  # Locked Isaac Sim disables sweep CCD on GPU.
    sim_cfg.physx.bounce_threshold_velocity = .05
    sim = sim_utils.SimulationContext(sim_cfg)
    scene = InteractiveScene(SceneCfg(num_envs=n,env_spacing=5.,replicate_physics=True))
    bodies = [f"/World/envs/env_{i}/{name}" for i in range(n) for name in ("Ball","Blade","Second")]
    configure_ccd(get_current_stage(),bodies,[p+"/geometry/mesh" for p in bodies])
    sim.reset()
    device = sim.device
    ball, blade, second, sensor = scene["ball"],scene["blade"],scene["second"],scene["contact"]
    with torch.inference_mode():
        for obj in (ball,blade,second):
            state = obj.data.default_root_state.clone()
            state[:,:3] = scene.env_origins
            state[:,2] = 5.
            state[:,3:7] = torch.tensor([1.,0.,0.,0.],device=device)
            state[:,7:] = 0.
            if obj is ball:
                state[:,0] += .2
                state[:,1] += torch.tensor([c['y'] for c in cases],device=device)
                state[:,7] = -torch.tensor([c['speed'] for c in cases],device=device)
            elif obj is blade:
                state[:,7] = torch.tensor([c.get('racket_vx',0.) for c in cases],device=device)
                state[:,12] = torch.tensor([c.get('omega',0.) for c in cases],device=device)
            else:
                state[:,0] += 2.
                state[-1,0] -= 1.6  # Second physical blade for repeated free collisions.
            obj.write_root_state_to_sim(state)
        scene.reset()
        event = ContactEpisodes(n,device)
        initial_pos = ball.data.root_pos_w.clone()
        initial_vel = ball.data.root_lin_vel_w.clone()
        initial_momentum = .0052*initial_vel+.09*(blade.data.root_lin_vel_w+second.data.root_lin_vel_w)
        external_impulse = torch.zeros((n,3),device=device)
        maximum_momentum_error = torch.zeros(n,device=device)
        counts_second = ContactEpisodes(n,device)
        contact_rows, trajectory = [], []
        previous_count = event.count.clone()
        first_step_angular_velocity = None
        for step in range(round(.20/args.dt)):
            force = torch.zeros((n,3),device=device)
            force[:3,2] = -.0052*9.81
            force[3:6] = drag_force(ball.data.root_lin_vel_w[3:6],.0052,physics['quadratic_drag_k_per_m'],[0,0,0])
            force[6:9] = drag_force(ball.data.root_lin_vel_w[6:9],.0052,physics['quadratic_drag_k_per_m'],[3,0,0])
            ball.set_external_force_and_torque(force[:,None],torch.zeros_like(force[:,None]),is_global=True)
            external_impulse += force*args.dt
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(args.dt)
            if step == 0:
                first_step_angular_velocity = blade.data.root_ang_vel_w.tolist()
            forces = sensor.data.force_matrix_w[:,0]
            gap = sphere_box_gap(ball.data.root_pos_w,blade.data.root_pos_w,blade.data.root_quat_w,[.0075,.11,.11],.025)
            new,touch = event.update(forces[:,0],gap)
            gap2 = sphere_box_gap(ball.data.root_pos_w,second.data.root_pos_w,second.data.root_quat_w,[.0075,.11,.11],.025)
            new2,touch2 = counts_second.update(forces[:,1],gap2)
            momentum = .0052*ball.data.root_lin_vel_w+.09*(blade.data.root_lin_vel_w+second.data.root_lin_vel_w)
            torch.maximum(maximum_momentum_error,(momentum-initial_momentum-external_impulse).norm(dim=-1),out=maximum_momentum_error)
            for lane in torch.where(touch|touch2)[0].tolist():
                contact_rows.append(dict(lane=lane,time=(step+1)*args.dt,new=bool(new[lane]),new_second=bool(new2[lane]),
                    force=forces[lane].tolist(),gap=float(gap[lane]),ball_velocity=ball.data.root_lin_vel_w[lane].tolist(),
                    blade_velocity=blade.data.root_lin_vel_w[lane].tolist(),blade_angular_velocity=blade.data.root_ang_vel_w[lane].tolist(),
                    ball_position=(ball.data.root_pos_w[lane]-scene.env_origins[lane]).tolist(),
                    blade_position=(blade.data.root_pos_w[lane]-scene.env_origins[lane]).tolist(),
                    blade_quaternion=blade.data.root_quat_w[lane].tolist(),contact_position=sensor.data.contact_pos_w[lane,0].tolist()))
            if (step+1)%round(.01/args.dt)==0:
                trajectory.append(dict(time=(step+1)*args.dt,position=(ball.data.root_pos_w-scene.env_origins).tolist(),
                    velocity=ball.data.root_lin_vel_w.tolist(),blade_position=(blade.data.root_pos_w-scene.env_origins).tolist(),
                    blade_quaternion=blade.data.root_quat_w.tolist(),contact_counts=event.count.tolist()))
        result = dict(backend="Isaac Lab / GPU PhysX",dt=args.dt,policy_dt=.01,cases=cases,
            sweep_ccd_active=False,speculative_ccd_requested=True,initial_position=(initial_pos-scene.env_origins).tolist(),
            initial_velocity=initial_vel.tolist(),final_position=(ball.data.root_pos_w-scene.env_origins).tolist(),
            final_velocity=ball.data.root_lin_vel_w.tolist(),contact_count=event.count.tolist(),
            second_contact_count=counts_second.count.tolist(),max_momentum_error=maximum_momentum_error.tolist(),
            first_step_angular_velocity=first_step_angular_velocity,
            contacts=contact_rows,trajectory=trajectory)
        (args.output/"benchmark.json").write_text(json.dumps(result,indent=2)+"\n")
        print("BENCHMARK "+json.dumps(dict(dt=args.dt,cases=n,contact_counts=result['contact_count'],
            maximum_momentum_error=max(result['max_momentum_error']))),flush=True)
except BaseException:
    (args.output/"error.txt").write_text(traceback.format_exc())
    raise
finally:
    if 'sim' in globals():
        if 'scene' in globals():
            del scene
        sim.clear_all_callbacks()
        sim.clear_instance()
    # Standalone fixtures can deadlock Kit's teardown of the tensor scene.
    # The process owns all its resources; receipts/error.txt are flushed above.
    app.close(skip_cleanup=True)
