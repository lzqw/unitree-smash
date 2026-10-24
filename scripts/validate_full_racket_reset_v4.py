#!/usr/bin/env python3
"""Compare legacy/default and upright racket reset without hiding self contact."""
import argparse,json,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
AppLauncher.add_app_launcher_args(p);args=p.parse_args();app=AppLauncher(args).app
import torch
from pxr import Usd,UsdPhysics
from isaacsim.core.utils.stage import get_current_stage
from isaaclab.sensors import ContactSensor,ContactSensorCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from quadsmash.tasks.stand.jump_env import JumpEnv,JumpEnvCfg
from quadsmash.physics.full_racket_v4 import load_racket,mass_properties


class ResetEnv(JumpEnv):
    def _setup_scene(self):
        super()._setup_scene()
        paths=[str(p.GetPath()).replace('/env_0/','/env_.*/')
            for p in Usd.PrimRange(get_current_stage().GetPrimAtPath('/World/envs/env_0/Robot'),Usd.TraverseInstanceProxies())
            if p.HasAPI(UsdPhysics.RigidBodyAPI) and p.GetName()!='Racket']
        self.other_body_paths=paths
        self.scene.sensors['racket_self']=ContactSensor(ContactSensorCfg(
            prim_path='/World/envs/env_.*/Robot/Racket',filter_prim_paths_expr=paths,
            force_threshold=.1,track_contact_points=True,max_contact_data_count_per_prim=256))

    def _pre_physics_step(self,actions):
        self._record_action(actions)
        limits=self.robot.data.joint_pos_limits[:,self.joint_ids]
        self.targets.copy_(self.robot.data.default_joint_pos)
        self.targets[:,self.joint_ids]=limits[...,0]+.5*(self.last_action+1)*(limits[...,1]-limits[...,0])


def main():
    args.output.mkdir(parents=True,exist_ok=True)
    cfg=JumpEnvCfg();cfg.scene.num_envs=2;cfg.scene.env_spacing=16.
    cfg.robot.spawn.usd_path=str(ROOT/load_racket()['output_asset'])
    cfg.sim.device='cuda:0';cfg.sim.dt=.001;cfg.decimation=cfg.sim.render_interval=10
    cfg.episode_length_s=2.
    env=ResetEnv(cfg);wrapper=RslRlVecEnvWrapper(env,clip_actions=None)
    try:
        with torch.inference_mode():
            obs,_=wrapper.reset();q=env.robot.data.default_joint_pos.clone()
            upright=[.0786861032,1.20102334,-2.585356,-.156469598,-1.056915283,.13533029]
            q[1,env.joint_ids[12:]]=torch.tensor(upright,device=env.device)
            env.robot.write_joint_state_to_sim(q,torch.zeros_like(q));env.sim.forward()
            limits=env.robot.data.joint_pos_limits[:,env.joint_ids]
            action=2*(q[:,env.joint_ids]-limits[...,0])/(limits[...,1]-limits[...,0])-1
            maximum_force=torch.zeros((2,len(env.other_body_paths)),device=env.device)
            cached_before_first_step=env.scene.sensors['racket_self'].data.force_matrix_w[:,0].norm(dim=-1).cpu().tolist()
            minimum_height=torch.full((2,),100.,device=env.device);maximum_height=torch.zeros_like(minimum_height)
            maximum_tilt=torch.zeros_like(minimum_height);trace=[]
            for tick in range(200):
                data=env.robot.data
                if tick<199:
                    force=env.scene.sensors['racket_self'].data.force_matrix_w[:,0].norm(dim=-1)
                    if tick>0:maximum_force=torch.maximum(maximum_force,force)
                    minimum_height=torch.minimum(minimum_height,data.root_pos_w[:,2])
                    maximum_height=torch.maximum(maximum_height,data.root_pos_w[:,2])
                    maximum_tilt=torch.maximum(maximum_tilt,data.projected_gravity_b[:,:2].norm(dim=-1))
                    trace.append(dict(time_s=tick*.01,root_position_m=(data.root_pos_w-env.scene.env_origins).cpu().tolist(),self_forces_n=force.cpu().tolist()))
                obs,_,_,_=wrapper.step(action)
            props=mass_properties(load_racket());tool_id=env.racket_body_id
            actual_com=env.robot.data.body_com_pos_b[0,tool_id]
            inertia=env.robot.root_physx_view.get_inertias()[0,tool_id].reshape(3,3)
            expected=torch.tensor(props['principal_inertia_kg_m2'])
            assert torch.allclose(torch.linalg.eigvalsh(inertia.cpu()),expected,atol=1e-6)
            assert torch.allclose(actual_com.cpu(),torch.tensor(props['center_of_mass_in_body_m']),atol=1e-6)
            result=dict(kind='Ground reset self-collision regression; not learned-policy acceptance',
                lane_names=['legacy_downward_default','upright_full_racket'],upright_arm_rad=upright,
                maximum_self_contact_force_by_body_n=maximum_force.cpu().tolist(),body_paths=env.other_body_paths,
                cached_before_first_step_forces_n=cached_before_first_step,
                minimum_root_height_m=minimum_height.cpu().tolist(),maximum_root_height_m=maximum_height.cpu().tolist(),
                maximum_tilt_xy=maximum_tilt.cpu().tolist(),full_tool_com_inertia_match=True,
                formal_acceptance=False,ground_initialization=True)
            (args.output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
            (args.output/'trace.json').write_text(json.dumps(trace)+'\n')
            assert maximum_force[1].max()<1.,'Upright reset must not overlap another robot body'
            assert minimum_height[1]>.20 and maximum_height[1]<.40 and maximum_tilt[1]<.35
            (args.output/'completed.json').write_text(json.dumps(dict(env_steps=400))+'\n')
            print(json.dumps(result),flush=True)
    finally:env.close()


try:main()
except BaseException:
    args.output.mkdir(parents=True,exist_ok=True);(args.output/'error.txt').write_text(traceback.format_exc());raise
finally:app.close(skip_cleanup=True)
