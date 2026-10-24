#!/usr/bin/env python3
"""Short native geometry/rule fixtures; fixed root, nonground, no policy claim."""
import argparse,json,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--variant',choices=['near-net-contact','empty-aabb-corner','prehit-invasion'],required=True)
AppLauncher.add_app_launcher_args(p);args=p.parse_args();app=AppLauncher(args).app
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from quadsmash.control.full_racket_geometry_v4 import FullRacketGeometryV4
from quadsmash.physics.full_racket_v4 import RacketPlaneSupport,load_racket
from quadsmash.physics.native import quat_rotate
from quadsmash.tasks.stand.full_racket_env_v4 import FullRacketEnvV4,full_racket_cfg
from quadsmash.usd_dependencies_v4 import audit_composed_asset


class NearNetFixtureEnv(FullRacketEnvV4):
    def _setup_scene(self):
        self.cfg.robot.spawn.articulation_props.fix_root_link=True
        super()._setup_scene()


def main():
    args.output.mkdir(parents=True,exist_ok=True)
    geometry=FullRacketGeometryV4()
    arm=json.loads((ROOT/'configs/full_racket_reset_v4.json').read_text())['arm_joint_positions_rad']
    point,rotation=geometry.forward(arm,base_position=(0,0,0))
    if args.variant=='near-net-contact':
        wanted=Rotation.from_euler('y',.05)
        contact=np.array([1.491,0,1.85])
        center=contact-wanted.apply([.0006,0,.08])
    else:
        wanted=Rotation.from_quat([0.,.5,-.5,np.sqrt(.5)])
        center=np.array([0.,0.,3.])
        xyzw=wanted.as_quat()
        support=RacketPlaneSupport(load_racket(),dtype=torch.float64)
        extent=float(support.maximum_x(torch.zeros((1,3),dtype=torch.float64),
            torch.tensor(xyzw[[3,0,1,2]][None],dtype=torch.float64))[0])
        center[0]=1.5+(.008 if args.variant=='prehit-invasion' else -.008)-extent
    base=wanted*Rotation.from_matrix(rotation).inv()
    root=center-base.apply(point);xyzw=base.as_quat()
    cfg=full_racket_cfg(1);cfg.seed=725603;cfg.episode_length_s=.1
    cfg.robot.spawn.articulation_props.fix_root_link=True
    cfg.robot.init_state.pos=tuple(root);cfg.robot.init_state.rot=tuple(xyzw[[3,0,1,2]])
    env=NearNetFixtureEnv(cfg);wrapped=RslRlVecEnvWrapper(env,clip_actions=None)
    try:
        with torch.inference_mode():
            observation,_=wrapped.reset()
            dependencies=audit_composed_asset(ROOT,cfg.robot.spawn.usd_path)
            data=env.robot.data
            tool=data.body_link_pos_w[:,env.racket_body_id]-env.scene.env_origins
            quaternion=data.body_link_quat_w[:,env.racket_body_id]
            if np.linalg.norm(tool[0].cpu().numpy()-center)>2e-5:
                raise ValueError('Native fixed-root initialization differs from planned fixture pose')
            normal=quat_rotate(quaternion,torch.tensor([[1.,0,0]],device=env.device))
            old_world=quat_rotate(quaternion[:,None],env.racket_points)+tool[:,None]
            maximum=env.racket_plane_support.maximum_x(tool,quaternion)
            initial=dict(tool_position_m=tool.cpu().tolist(),tool_quaternion_wxyz=quaternion.cpu().tolist(),
                native_racket_maximum_x_m=maximum.cpu().tolist(),old_aabb_proxy_maximum_x_m=old_world[...,0].amax(1).cpu().tolist())
            if args.variant=='near-net-contact':
                contact=tool+quat_rotate(quaternion,torch.tensor([[.0006,0,.08]],device=env.device))
                ball=contact+.029*normal;velocity=-8*normal
                assert float(maximum[0])<1.5 and float(ball[0,0])>1.5
            else:
                ball=torch.tensor([[4.,1.,3.]],device=env.device);velocity=torch.zeros_like(ball)
                assert float(old_world[...,2].min())>1.55
                if args.variant=='empty-aabb-corner':
                    assert float(maximum[0])<1.5<float(old_world[...,0].max())
                else:assert float(maximum[0])>1.5
            env.initialize_rally_launch(torch.tensor([0],device=env.device),ball,velocity)
            default=data.default_joint_pos[0,env.joint_ids];limits=data.joint_pos_limits[0,env.joint_ids]
            action=(2*(default-limits[:,0])/(limits[:,1]-limits[:,0])-1)[None]
            observation,_,_,_=wrapped.step(action)
            flags={name:bool(getattr(env.rally,name)[0]) for name in env.rally.flags}
            numerical=int((env.sweep_misses+env.sweep_unresolved)[0])
            result=dict(scope='One actual10ms native fixture, fixed root/nonground. Geometry/rules only, not a jump or policy acceptance.',
                variant=args.variant,initial=initial,flags=flags,actual_contacts=env.contact_records,
                contact_point_m=env.rally.contact_point[0].cpu().tolist(),
                crossing_point_m=env.rally.crossing_point[0].cpu().tolist(),
                contact_time_s=float(env.rally.contact_time[0]),crossing_time_s=float(env.rally.crossing_time[0]),
                final_native_racket_maximum_x_m=env.racket_plane_support.maximum_x(
                    data.body_link_pos_w[:,env.racket_body_id]-env.scene.env_origins,
                    data.body_link_quat_w[:,env.racket_body_id]).cpu().tolist(),
                numerical_failures=numerical,asset_dependencies=dependencies,
                native_plane_query_calls=env.racket_plane_query_calls,native_plane_events=env.racket_plane_records,
                control_ticks=1,physics_dt_s=cfg.sim.dt,formal_acceptance=False,fixed_base=True,ground_initialization=False)
            (args.output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
            assert numerical==0
            assert not flags['robot_net_fault'] and not flags['engineering_body_crossing']
            if args.variant=='near-net-contact':
                assert flags['had_contact'] and flags['legal_initial_contact'] and flags['string_hit']
                assert env.contact_records and env.contact_records[0]['ball_position_m'][0]>1.5
                assert float(env.rally.contact_point[0,0])<1.5
                assert flags['crossed_net'] and not bool(env.rally.faulted()[0])
            elif args.variant=='empty-aabb-corner':
                assert not flags['racket_invasion_fault'] and not flags['had_contact']
                assert result['final_native_racket_maximum_x_m'][0]<=1.5
            else:
                assert flags['racket_invasion_fault'] and not flags['had_contact']
                assert env.racket_plane_query_calls>0 and env.racket_plane_records
            (args.output/'completed.json').write_text(json.dumps(dict(env_steps=1))+'\n')
            print(json.dumps(result),flush=True)
    finally:env.close()


try:main()
except BaseException:
    args.output.mkdir(parents=True,exist_ok=True);(args.output/'error.txt').write_text(traceback.format_exc());raise
finally:app.close(skip_cleanup=True)
