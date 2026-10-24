"""Ground-start no-ball curriculum on the actual full-racket asset/court."""
import torch
import isaaclab.sim as sim_utils
from pxr import Usd,UsdPhysics
from isaaclab.sensors import ContactSensor,ContactSensorCfg
from isaacsim.core.utils.stage import get_current_stage
from .jump_env import JumpEnv,JumpEnvCfg
from .coupled_env import ROOT,CoupledEnv
from .visual_geometry import collision_proxy_corners
from quadsmash.physics.full_racket_v4 import load_racket,apply_neutral_arm_reset
from quadsmash.physics.court_v4 import load_court,spawn_court_net,spawn_court_lines
from quadsmash.physics.native import quat_rotate,configure_ccd,physics_contract

CONTACT_CONTRACT='full-racket-ball-material-parity-v4'


def reach_cfg(num_envs,physics_dt=.001):
    cfg=JumpEnvCfg();court=load_court()
    cfg.robot.spawn.usd_path=str(ROOT/load_racket()['output_asset'])
    cfg.scene.num_envs,cfg.scene.env_spacing=num_envs,court['environment_spacing_m']
    cfg.ground_size_m=(4096.,4096.,.1)
    cfg.sim.device='cuda:0';cfg.sim.dt=physics_dt
    cfg.sim.physx.enable_ccd=False;cfg.sim.physx.bounce_threshold_velocity=.05
    cfg.decimation=cfg.sim.render_interval=round(.01/physics_dt)
    cfg.episode_length_s=6.;cfg.event_goal_enabled=True
    cfg.event_goal_position=(1.10,0.,1.80);cfg.event_goal_normal=(1.,0.,0.)
    cfg.event_goal_time_s=2.75
    return apply_neutral_arm_reset(cfg)


class FullRacketReachEnvV4(JumpEnv):
    def __init__(self,cfg,**kwargs):
        self._reach_ready=False;self.reach_records=[]
        super().__init__(cfg,**kwargs)
        self.net_seen=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        self.body_crossed=self.net_seen.clone()
        self.maximum_net_force=torch.zeros(self.num_envs,device=self.device)
        ids,points,self.proxy_paths=collision_proxy_corners(get_current_stage(),self.robot)
        self.proxy_ids=torch.tensor(ids,device=self.device)
        self.proxy_points=torch.tensor(points,dtype=torch.float32,device=self.device)[None]
        self.metric_stride=max(1,round(.001/self.physics_dt))
        self._reach_ready=True

    def _setup_scene(self):
        super()._setup_scene()
        stage=get_current_stage();court=load_court()
        physics=physics_contract()
        material=sim_utils.RigidBodyMaterialCfg(restitution=physics['restitution'],
            static_friction=physics['friction'],dynamic_friction=physics['friction'],restitution_combine_mode='max')
        material.func('/World/ShuttleContactMaterial',material)
        configure_ccd(stage,[f'/World/envs/env_{i}/Robot/Racket' for i in range(self.num_envs)],
            [f'/World/envs/env_{i}/Robot/Racket/Collider' for i in range(self.num_envs)],'/World/ShuttleContactMaterial')
        from isaaclab.sim.utils import bind_physics_material
        for i in range(self.num_envs):
            path=f'/World/envs/env_{i}/Net'
            colliders=spawn_court_net(stage,path,court)
            spawn_court_lines(stage,f'/World/envs/env_{i}/Court',court)
            net_material=sim_utils.RigidBodyMaterialCfg(restitution=.1)
            material_path=path+'/geometry/material'
            net_material.func(material_path,net_material)
            for collider in colliders:bind_physics_material(collider,material_path)
        paths=[str(p.GetPath()).replace('/env_0/','/env_.*/')
               for p in Usd.PrimRange(stage.GetPrimAtPath('/World/envs/env_0/Robot'),Usd.TraverseInstanceProxies())
               if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        self.scene.sensors['inverse_net']=ContactSensor(ContactSensorCfg(
            prim_path='/World/envs/env_.*/Net',filter_prim_paths_expr=paths,
            force_threshold=.1,max_contact_data_count_per_prim=128))

    def _pre_physics_step(self,actions):
        self.action_delta.copy_((actions.clamp(-1,1)-self.last_action).square().sum(1))
        self._record_action(actions)
        limits=self.robot.data.joint_pos_limits[:,self.joint_ids]
        self.targets.copy_(self.robot.data.default_joint_pos)
        self.targets[:,self.joint_ids]=limits[...,0]+.5*(self.last_action+1)*(limits[...,1]-limits[...,0])

    def _apply_action(self):
        # Match the formal BallEnv's1ms jump/recovery metric sampling while
        # preserving the chosen native integration step and every net/plane
        # force sample. This lets the no-ball curriculum adapt at62.5us too.
        CoupledEnv._apply_action(self)
        frame=self._sim_step_counter-1
        if frame==0 or frame%getattr(self,'metric_stride',1)==0:
            self._consume_jump_frame(frame)
        if self._reach_ready:
            force=self.scene.sensors['inverse_net'].data.force_matrix_w.norm(dim=-1).sum((1,2))
            self.net_seen|=force>.1
            self.maximum_net_force.copy_(torch.maximum(self.maximum_net_force,force))
            data=self.robot.data
            points=quat_rotate(data.body_link_quat_w[:,self.proxy_ids,None,:],self.proxy_points)+data.body_link_pos_w[:,self.proxy_ids,None,:]-self.scene.env_origins[:,None,None,:]
            self.body_crossed|=(points[...,0]>1.5).any((1,2))

    def _reset_idx(self,ids):
        if self._reach_ready:
            for i in ids[self._episode_has_stepped[ids]].cpu().tolist():
                self.reach_records.append(dict(env_id=i,robot_net_contact=bool(self.net_seen[i]),
                    engineering_body_crossing=bool(self.body_crossed[i]),maximum_net_force_n=float(self.maximum_net_force[i])))
            self.net_seen[ids]=False;self.body_crossed[ids]=False;self.maximum_net_force[ids]=0.
        super()._reset_idx(ids)
