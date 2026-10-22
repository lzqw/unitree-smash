"""Full-load jump with substep COM, flight and continuous recovery metrics."""
from __future__ import annotations

import json
import torch
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi
from .coupled_env import CoupledEnv, CoupledEnvCfg, LEG_JOINTS, ROOT
from . import jump_metrics

PROTOCOL=json.loads((ROOT/'configs/evaluation_protocol_v2.json').read_text())

@configclass
class JumpEnvCfg(CoupledEnvCfg):
    episode_length_s=4.0
    is_finite_horizon=False
    body_ground_contact=ContactSensorCfg(prim_path='/World/envs/env_.*/Robot/Go2/(base|.*_hip|.*_thigh|.*_calf)',
        filter_prim_paths_expr=['/World/Ground/geometry/mesh'],force_threshold=1.)
    arm_ground_contact=ContactSensorCfg(prim_path='/World/envs/env_.*/Robot/Piper/.*',
        filter_prim_paths_expr=['/World/Ground/geometry/mesh'],force_threshold=1.)
    tool_ground_contact=ContactSensorCfg(prim_path='/World/envs/env_.*/Robot/(MountPlate|Racket)',
        filter_prim_paths_expr=['/World/Ground/geometry/mesh'],force_threshold=1.)

class JumpEnv(CoupledEnv):
    cfg: JumpEnvCfg

    def _setup_scene(self):
        super()._setup_scene()
        for name in ('body_ground_contact','arm_ground_contact','tool_ground_contact'):
            self.scene.sensors[name]=ContactSensor(getattr(self.cfg,name))

    def __init__(self,cfg,**kwargs):
        cfg.robot.spawn.articulation_props.fix_root_link=False
        super().__init__(cfg,**kwargs)
        self.leg_ids,_=self.robot.find_joints(list(LEG_JOINTS),preserve_order=True)
        self.body_ground_contact=self.scene.sensors['body_ground_contact']
        self.ground_sensors=[self.scene.sensors[name] for name in ('body_ground_contact','arm_ground_contact','tool_ground_contact')]
        self.body_masses=self.robot.root_physx_view.get_masses().to(self.device)
        self.mass_weights=self.body_masses/self.body_masses.sum(1,keepdim=True)
        jump_metrics.initialize(self,self.num_envs,self.device)
        for name in ('jump_clock','reset_com_height','max_height','max_leg_torque','max_arm_torque','max_joint_speed',
                     'action_delta','action_clip_count','target_limit_count','saturated_joint_steps'):
            setattr(self,name,torch.zeros(self.num_envs,device=self.device))
        self.com_position=torch.zeros((self.num_envs,3),device=self.device)
        self.com_velocity=torch.zeros_like(self.com_position)
        self.clear_flight=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        self.stable_recovery=torch.zeros_like(self.clear_flight)
        self._jump_physics_frame=-1
        self.completed_episodes=self.successful_episodes=self.failed_episodes=0
        self._metrics_started=False
        self._episode_has_stepped=torch.zeros_like(self.clear_flight)
        self.episode_records=[]

    def _phase(self):
        return self.episode_length_buf.float()*self.step_dt

    def _feet_in_contact(self):
        return self.observed_foot_contact

    def _pre_physics_step(self,actions):
        self.action_delta.copy_((actions.clamp(-1.,1.)-self.last_action).square().sum(1))
        self._record_action(actions)
        self.action_clip_count += (actions.abs()>1.).sum(1)
        phase=self._phase().unsqueeze(1)
        leg_scale=torch.where(phase<.8,.25,.55)
        scale=torch.cat((leg_scale.expand(-1,12),torch.full((self.num_envs,6),.12,device=self.device)),1)
        self.targets.copy_(self.robot.data.default_joint_pos)
        self.targets[:,self.joint_ids] += scale*self.last_action
        limits=self.robot.data.joint_pos_limits
        self.target_limit_count += ((self.targets<limits[:,:,0])|(self.targets>limits[:,:,1])).sum(1)
        self.targets.clamp_(min=limits[:,:,0],max=limits[:,:,1])

    def _apply_action(self):
        super()._apply_action()
        self._consume_jump_frame(self._sim_step_counter-1)

    def _consume_jump_frame(self,frame):
        if frame<=self._jump_physics_frame:return
        self._jump_physics_frame=frame
        if frame==0:return
        self._observe_physics_frame(frame)
        self.jump_clock += self.physics_dt
        data=self.robot.data
        self.com_position.copy_((self.mass_weights[:,:,None]*data.body_com_pos_w).sum(1))
        self.com_velocity.copy_((self.mass_weights[:,:,None]*data.body_com_lin_vel_w).sum(1))
        roll,pitch,_=euler_xyz_from_quat(data.root_quat_w)
        # All robot links except feet; one shared static ground filter per view.
        body_force=sum(sensor.data.force_matrix_w.norm(dim=-1).sum((1,2)) for sensor in self.ground_sensors)
        clear,stable=jump_metrics.advance(self,dt=self.physics_dt,time=self.jump_clock,
            com_height=self.com_position[:,2],foot_contact=self.observed_foot_contact,
            foot_clearance=data.body_pos_w[:,self.foot_ids,2].min(1).values,
            roll_pitch=torch.stack((wrap_to_pi(roll),wrap_to_pi(pitch)),1),
            horizontal_speed=data.root_link_lin_vel_w[:,:2].norm(dim=-1),
            angular_speed=data.root_ang_vel_w.norm(dim=-1),joint_speed=data.joint_vel.abs().max(1).values,
            body_ground_force=body_force,protocol=PROTOCOL)
        self.clear_flight.copy_(clear);self.stable_recovery.copy_(stable)
        torch.maximum(self.max_height,data.root_pos_w[:,2],out=self.max_height)
        torque=data.applied_torque[:,self.joint_ids].abs()
        torch.maximum(self.max_leg_torque,torque[:,:12].max(1).values,out=self.max_leg_torque)
        torch.maximum(self.max_arm_torque,torque[:,12:].max(1).values,out=self.max_arm_torque)
        torch.maximum(self.max_joint_speed,data.joint_vel.abs().max(1).values,out=self.max_joint_speed)
        self.saturated_joint_steps += (torque[:,:12]>=23.5*.99).sum(1)

    @property
    def jump_success(self):
        return jump_metrics.success(self) & (self.robot.data.root_pos_w[:,2]>=.12)

    def _get_dones(self):
        self._consume_jump_frame(self._sim_step_counter)
        return self.robot.data.root_pos_w[:,2]<.12,self.episode_length_buf>=self.max_episode_length

    def _get_rewards(self):
        self._episode_has_stepped[:]=True
        phase=self._phase()
        # A positive-velocity integral rewards repeated small oscillations.
        # Track one timed full-COM preparation/apex trajectory instead. This
        # is dense training shaping only; acceptance still uses true takeoff.
        target_com=self.reset_com_height-.06*torch.exp(-((phase-.93)/.12).square()) \
            +.14*torch.exp(-((phase-1.25)/.19).square())
        height_tracking=torch.exp(-((self.com_position[:,2]-target_com)/.06).square())*((phase>=.65)&(phase<=1.7))
        flight=self.clear_flight.float()*(self.flight_apex_rise/.08).clamp(0.,1.)
        recovery=self.stable_recovery.float()
        # Shaping is time-integrated; each actual jump/success bonus occurs once.
        first_jump=self.flight_qualified & (~self.rewarded_jump)
        first_success=self.jump_success & (~self.rewarded_success)
        first_body=self.body_ground_seen & (~self.rewarded_body_contact)
        self.rewarded_jump |= first_jump
        self.rewarded_success |= first_success
        self.rewarded_body_contact |= first_body
        data=self.robot.data
        arm_error=(data.joint_pos[:,self.joint_ids[12:]]-data.default_joint_pos[:,self.joint_ids[12:]]).square().sum(1)
        settling=(phase>1.8).float()*self.landed_once
        settle_cost=.1*data.root_ang_vel_w.square().sum(1)+.03*data.joint_vel.square().sum(1)
        tilt=data.projected_gravity_b[:,:2].square().sum(1)
        supported=self.observed_foot_contact.any(1)&(~self.body_ground_seen)
        prepare_quality=(phase<.8)*supported*torch.exp(-4.*tilt)
        recovery_quality=self.landed_once*supported*torch.exp(-4.*tilt
            -2.*data.root_link_lin_vel_w[:,:2].square().sum(1)-settle_cost)
        return (8.*height_tracking+3.*flight+3.*recovery+.3*prepare_quality+2.*recovery_quality-.03*arm_error
            -.00002*data.applied_torque.square().sum(1)-.005*self.action_delta-settling*settle_cost)*self.step_dt \
            +2.*first_jump.float()+5.*first_success.float()-4.*first_body.float()-2.*self.reset_terminated.float()

    @torch.inference_mode()
    def _reset_idx(self,env_ids):
        complete=env_ids[self._episode_has_stepped[env_ids]] if self._metrics_started and env_ids.numel() else env_ids[:0]
        if complete.numel():
            success=self.jump_success
            self.completed_episodes+=int(complete.numel())
            self.successful_episodes+=int(success[complete].sum())
            self.failed_episodes+=int((~success[complete]).sum())
            names={'max_height_m':'max_height','total_com_rise_m':'max_com_rise','airborne_s':'airborne_time',
                'recovery_s':'recovery_time','takeoff_time_s':'takeoff_time','landing_time_s':'landing_time',
                'maximum_body_ground_force_n':'maximum_body_ground_force','max_leg_torque_nm':'max_leg_torque',
                'max_arm_torque_nm':'max_arm_torque','max_joint_speed_rad_s':'max_joint_speed',
                'clipped_action_values':'action_clip_count','limited_joint_targets':'target_limit_count',
                'saturated_leg_joint_substeps':'saturated_joint_steps'}
            values=torch.stack([getattr(self,v)[complete] for v in names.values()],1).cpu().tolist()
            flags=torch.stack((self.valid_jump_landed[complete],self.rewarded_jump[complete],self.body_ground_seen[complete],
                self.recovery_seen[complete],success[complete],self.reset_terminated[complete]),1).cpu().tolist()
            for i,numbers,booleans in zip(complete.cpu().tolist(),values,flags):
                self.episode_records.append({'env_id':i,**dict(zip(names,numbers)),
                    **dict(zip(('qualified_jump_landed','qualified_flight_seen','body_landing','stable_recovery','success','terminated'),booleans))})
        super()._reset_idx(env_ids)
        self.reset_com_height[env_ids]=(self.mass_weights[env_ids]*self.robot.data.body_com_pos_w[env_ids,:,2]).sum(1)
        jump_metrics.reset(self,env_ids)
        for name in ('jump_clock','max_height','max_leg_torque','max_arm_torque','max_joint_speed',
                     'action_delta','action_clip_count','target_limit_count','saturated_joint_steps',
                     'com_position','com_velocity','clear_flight','stable_recovery','_episode_has_stepped'):
            getattr(self,name)[env_ids]=0
        self._metrics_started=True
