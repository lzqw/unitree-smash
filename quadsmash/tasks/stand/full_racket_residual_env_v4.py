"""Full-model TRAIN residual learning; original physics and rally metrics."""
import torch
from .full_racket_train_env_v4 import RallyTrainingEnvV4
from quadsmash.physics.native import quat_rotate
from quadsmash.control.full_racket_residual_reward_v4 import ResidualEventRewardV4
from quadsmash.control.measured_joint_placement_reward_v4 import MeasuredJointPlacementRewardV4
from quadsmash.control.measured_string_alignment_reward_v4 import MeasuredStringAlignmentRewardV4,alignment_settings,attack_aligned_target


class FullRacketResidualTrainingEnvV4(RallyTrainingEnvV4):
    action_family='bounded-residual-joint-targets-full-racket-v4'

    def __init__(self,*args,joint_placement_reward=None,live_alignment_reward=None,**kwargs):
        self.residual_rewards=None
        self.joint_placement_reward=None
        self.live_alignment_reward=None
        alignment=alignment_settings(live_alignment_reward)
        if alignment is not None and joint_placement_reward is None:
            raise ValueError('Measured string alignment requires explicit measured joint placement too')
        super().__init__(*args,**kwargs)
        n,device=self.num_envs,self.device
        self.residual_rewards=ResidualEventRewardV4(n,device)
        if joint_placement_reward is not None:
            self.joint_placement_reward=MeasuredJointPlacementRewardV4(
                n,device,self.rally.court,joint_placement_reward)
        if alignment is not None:
            self.live_alignment_reward=MeasuredStringAlignmentRewardV4(n,device,alignment,
                semiaxes_yz_m=self.racket_config['assumptions']['strings_semiaxes_yz_m'],
                contact_plane_offset_m=self.cfg.shuttle_radius_m+self.racket_config['assumptions']['string_thickness_m']/2)
        self.reward_target=torch.zeros((n,3),device=device)
        self.reward_event_time=torch.full((n,),2.75,device=device)
        self.reward_plan_valid=torch.zeros(n,dtype=torch.bool,device=device)
        self.residual_raw=torch.zeros((n,18),device=device)
        self.reward_velocity=torch.zeros(3,device=device)
        self.reward_normal=torch.tensor([1.,0.,0.],device=device)
        self.reward_offset=torch.tensor([.0256,0.,0. if alignment is not None else .1],device=device)
        self.last_residual_reward=torch.zeros(n,device=device)

    def _get_rewards(self):
        if self.residual_rewards is None:
            return super()._get_rewards()
        self._episode_has_stepped[:]=True
        # The existing jump recorder reads these historical event buffers.
        self.rewarded_jump|=self.flight_qualified
        self.rewarded_success|=self.jump_success
        self.rewarded_body_contact|=self.body_ground_seen
        rally=self.rally
        air=rally.legal_initial_contact&rally.string_hit&rally.airborne_string_hit
        robot_fault=rally.robot_rule_faulted()
        v=rally.outgoing_velocity
        attack=(air&rally.released&torch.isfinite(v).all(1)&(v.norm(dim=1)>=8.)&(v[:,0]>0)&(v[:,2]<=-1.))
        flags=dict(flight=self.flight_qualified,air=air&(~robot_fault),attack=attack&(~robot_fault),
            crossed=air&rally.crossed_net&(~robot_fault),target=air&rally.landed_target&(~rally.faulted()),
            contact_recovery=air&self.jump_success&(~rally.faulted()),
            # Pay the full chain only at the actual episode boundary, so a
            # later fall cannot keep an earlier full-success training bonus.
            complete=rally.complete(self.jump_success)&self.reset_time_outs&(~self.reset_terminated),
            robot_fault=robot_fault,body_fault=self.body_ground_seen)
        if self.live_alignment_reward is not None:
            # Original v1/v2 pay20 for any legal air target hit. In opt-in v3
            # that event also requires the unchanged actual attack predicate.
            # Partial speed feedback below is named shaping, never an attack.
            flags['target']=attack_aligned_target(air,attack,rally.landed_target,rally.faulted())
        d=self.robot.data;clock=self.episode_length_buf*self.step_dt
        point=d.body_link_pos_w[:,self.racket_body_id]-self.scene.env_origins+quat_rotate(
            d.body_link_quat_w[:,self.racket_body_id],self.reward_offset.expand(self.num_envs,-1))
        normal=quat_rotate(d.body_link_quat_w[:,self.racket_body_id],
            torch.tensor([1.,0.,0.],device=self.device).expand(self.num_envs,-1))
        target=self.reward_target+self.reward_velocity*(clock-self.reward_event_time).clamp(-.15,.1)[:,None]
        active=(self.reward_plan_valid&rally.in_play&(~rally.had_contact)&(~self.body_ground_seen)
                &(clock>=self.reward_event_time-.25)&(clock<=self.reward_event_time+.03))
        position_term=torch.exp(-((point-target).norm(dim=1)/.08).square())
        normal_term=.25*torch.exp(-((normal-self.reward_normal).norm(dim=1)/.2).square())
        dense=((position_term+normal_term) if self.live_alignment_reward is None else normal_term)*active
        cost=.00002*d.applied_torque.square().sum(1)+.005*self.action_delta+.02*self.residual_raw.square().sum(1)
        reward=self.residual_rewards.step(flags,dense,cost,self.step_dt)
        if self.joint_placement_reward is not None:
            reward=reward+self.joint_placement_reward.step(ground_seen=rally.ground_seen,
                legal_air=air,attack=attack,crossed=rally.crossed_net,
                faulted=rally.faulted(),first_landing=rally.landing_point)
        if self.live_alignment_reward is not None:
            tool=d.body_link_pos_w[:,self.racket_body_id]
            rotation=d.body_link_quat_w[:,self.racket_body_id]
            relative=self.shuttle.data.root_pos_w-tool
            inverse=torch.cat((rotation[:,:1],-rotation[:,1:]),1)
            relative_tool=quat_rotate(inverse,relative)
            velocity=self.shuttle.data.root_lin_vel_w-d.body_link_lin_vel_w[:,self.racket_body_id]
            normal_rate=torch.linalg.cross(d.body_link_ang_vel_w[:,self.racket_body_id],normal)
            distance_rate=(normal*velocity).sum(1)+(normal_rate*relative).sum(1)
            reward=reward+self.live_alignment_reward.step(clock_s=clock,
                ball_relative_tool_m=relative_tool,plane_distance_rate_mps=distance_rate,
                ball_in_play=rally.in_play&self.ball_observation_enabled,had_contact=rally.had_contact,
                ground_seen=rally.ground_seen,body_contact=self.body_ground_seen,faulted=rally.faulted(),
                legal_air=air,released=rally.released,frame_hit=rally.frame_hit,
                outgoing_velocity_mps=v,terminated=self.reset_terminated|self.reset_time_outs)
        self.last_residual_reward=reward.detach().clone()
        return reward

    def _reset_idx(self,ids):
        super()._reset_idx(ids)
        if self.residual_rewards is not None:self.residual_rewards.reset(ids)
        if self.joint_placement_reward is not None:self.joint_placement_reward.reset(ids)
        if self.live_alignment_reward is not None:self.live_alignment_reward.reset(ids)
