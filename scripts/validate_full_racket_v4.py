#!/usr/bin/env python3
"""Native full-racket mass, transform, velocity, standing and court regression."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--num-envs',type=int,default=8)
parser.add_argument('--positive-net-fixture',action='store_true')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np
import torch
from pxr import Usd,UsdPhysics
from isaacsim.core.utils.stage import get_current_stage
from isaaclab.sensors import ContactSensor,ContactSensorCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from quadsmash.tasks.stand.jump_env import JumpEnv,JumpEnvCfg
from quadsmash.tasks.stand.visual_geometry import export_robot_geometry
from quadsmash.physics.full_racket_v4 import load_racket,mass_properties
from quadsmash.physics.court_v4 import load_court,spawn_court_net,spawn_court_lines
from quadsmash.physics.native import quat_rotate


class RegressionEnv(JumpEnv):
    def _setup_scene(self):
        super()._setup_scene()
        stage,court = get_current_stage(),load_court()
        for i in range(self.num_envs):
            spawn_court_net(stage,f'/World/envs/env_{i}/Net',court)
            spawn_court_lines(stage,f'/World/envs/env_{i}/Court',court)
        paths = [str(p.GetPath()).replace('/env_0/','/env_.*/')
                 for p in Usd.PrimRange(stage.GetPrimAtPath('/World/envs/env_0/Robot'),Usd.TraverseInstanceProxies())
                 if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        self.scene.sensors['inverse_net'] = ContactSensor(ContactSensorCfg(
            prim_path='/World/envs/env_.*/Net',filter_prim_paths_expr=paths,
            force_threshold=.1,track_contact_points=True,max_contact_data_count_per_prim=64))


def main():
    args.output.mkdir(parents=True,exist_ok=True)
    racket,court = load_racket(),load_court()
    cfg = JumpEnvCfg()
    cfg.robot.spawn.usd_path = str(ROOT/racket['output_asset'])
    cfg.scene.num_envs,cfg.scene.env_spacing = args.num_envs,court['environment_spacing_m']
    cfg.ground_size_m = (4096.,4096.,.1)
    cfg.sim.device = 'cuda:0'
    cfg.sim.dt,cfg.decimation,cfg.sim.render_interval = .001,10,10
    seconds = .5 if args.positive_net_fixture else 4.
    cfg.episode_length_s = seconds
    cfg.seed = 720101
    env = RegressionEnv(cfg)
    wrapped = RslRlVecEnvWrapper(env,clip_actions=None)
    try:
        observation,_ = wrapped.reset()
        flange_id = env.robot.find_bodies('link6')[0][0]
        tool_id = env.racket_body_id
        if args.positive_net_fixture:
            # Deliberately intersect actual robot/tool with the band or posts
            # at initialization. These nonground fixtures are not policy runs.
            root = env.robot.data.root_state_w.clone()
            target = env.scene.env_origins+torch.tensor([court['net_plane_x_m'],0.,1.15],device=env.device)
            target[args.num_envs//2:,1] += court['post_y_center_abs_m']
            root[:,:3] += target-env.robot.data.body_link_pos_w[:,tool_id]
            with torch.inference_mode():
                env.robot.write_root_state_to_sim(root)
                env.sim.forward()
        properties = mass_properties(racket)
        masses = env.robot.root_physx_view.get_masses()
        observed_mass = float(masses[0,tool_id])
        observed_com = env.robot.data.body_com_pos_b[0,tool_id].cpu().numpy()
        engine_inertia = env.robot.root_physx_view.get_inertias()[0,tool_id].cpu().numpy().reshape(3,3)
        assert abs(observed_mass-properties['mass_kg'])<1e-6
        assert np.max(np.abs(observed_com-properties['center_of_mass_in_body_m']))<1e-6
        assert np.max(np.abs(np.sort(np.linalg.eigvalsh(engine_inertia))-properties['principal_inertia_kg_m2']))<1e-6
        export_robot_geometry(get_current_stage(),env.robot,args.output)
        offset = torch.tensor(racket['string_center_in_flange_m'],device=env.device).expand(args.num_envs,-1)
        maximum_position_error,maximum_velocity_error = 0.,0.
        minimum_root_height,maximum_tilt,maximum_net_force = float('inf'),0.,0.
        net_seen = torch.zeros(args.num_envs,dtype=torch.bool,device=env.device)
        poses,trace = [],[]
        with torch.inference_mode():
            for tick in range(round(seconds/.01)):
                data = env.robot.data
                displacement = quat_rotate(data.body_link_quat_w[:,flange_id],offset)
                expected_position = data.body_link_pos_w[:,flange_id]+displacement
                expected_velocity = data.body_link_lin_vel_w[:,flange_id]+torch.linalg.cross(
                    data.body_link_ang_vel_w[:,flange_id],displacement)
                maximum_position_error = max(maximum_position_error,float((expected_position-data.body_link_pos_w[:,tool_id]).norm(dim=-1).max()))
                maximum_velocity_error = max(maximum_velocity_error,float((expected_velocity-data.body_link_lin_vel_w[:,tool_id]).norm(dim=-1).max()))
                minimum_root_height = min(minimum_root_height,float(data.root_link_pos_w[:,2].min()))
                maximum_tilt = max(maximum_tilt,float(data.projected_gravity_b[:,:2].norm(dim=-1).max()))
                forces = env.scene.sensors['inverse_net'].data.force_matrix_w
                maximum_net_force = max(maximum_net_force,float(forces.norm(dim=-1).sum((1,2)).max()))
                net_seen |= forces.norm(dim=-1).sum((1,2))>.1
                if tick%2==0:
                    poses.append(torch.cat((data.body_link_pos_w[0]-env.scene.env_origins[0],data.body_link_quat_w[0]),-1).cpu().numpy())
                    trace.append(dict(time_s=tick*.01,tool_position_m=(data.body_link_pos_w[0,tool_id]-env.scene.env_origins[0]).cpu().tolist(),tool_velocity_mps=data.body_link_lin_vel_w[0,tool_id].cpu().tolist(),root_position_m=(data.root_link_pos_w[0]-env.scene.env_origins[0]).cpu().tolist()))
                action = torch.zeros((args.num_envs,18),device=env.device)
                if tick>=100:
                    action[:,12:] = .3*torch.sin(torch.tensor((tick-100)*.01,device=env.device))
                observation,_,_,_ = wrapped.step(action)
        np.save(args.output/'link_poses.npy',np.stack(poses))
        (args.output/'trajectory.json').write_text(json.dumps(trace)+'\n')
        result = dict(kind='Native mechanical regression only, not a new policy acceptance result',
            asset=racket['output_asset'],asset_sha256=hashlib.sha256((ROOT/racket['output_asset']).read_bytes()).hexdigest(),
            total_mass_kg=float(masses[0].sum()),tool_mass_kg=observed_mass,tool_com_in_body_m=observed_com.tolist(),
            engine_inertia_kg_m2=engine_inertia.tolist(),expected_mass_properties=properties,
            maximum_flange_to_hit_position_error_m=maximum_position_error,
            maximum_hit_velocity_formula_error_mps=maximum_velocity_error,
            minimum_root_height_m=minimum_root_height,maximum_projected_gravity_xy=maximum_tilt,
            negative_fixture_maximum_net_force_n=maximum_net_force,inverse_net_sensor_shape=list(forces.shape),
            environments=args.num_envs,seconds_per_environment=seconds,ground_initialization=not args.positive_net_fixture,
            positive_net_fixture=args.positive_net_fixture,net_contact_seen_per_environment=net_seen.cpu().tolist(),
            court=court,formal_acceptance=False,old_passes_inherited=False)
        (args.output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
        if args.positive_net_fixture:
            assert net_seen.all(), 'Band/post positive fixtures must trigger in every environment'
        else:
            assert maximum_position_error<.002 and maximum_velocity_error<.05
            assert minimum_root_height>.20 and maximum_tilt<.35
            assert maximum_net_force<.1
        (args.output/'completed.json').write_text(json.dumps(dict(env_steps=args.num_envs*round(seconds/.01)))+'\n')
        print(json.dumps(result),flush=True)
    finally:
        env.close()


try:
    main()
except BaseException:
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'error.txt').write_text(traceback.format_exc())
    raise
finally:
    app.close(skip_cleanup=True)
