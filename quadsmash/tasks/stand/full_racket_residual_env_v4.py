"""Full-model TRAIN residual learning; original physics and rally metrics."""
import torch
from .full_racket_train_env_v4 import RallyTrainingEnvV4
from quadsmash.physics.native import quat_rotate
from quadsmash.control.full_racket_residual_reward_v4 import ResidualEventRewardV4


class FullRacketResidualTrainingEnvV4(RallyTrainingEnvV4):
    action_family='bounded-residual-joint-targets-full-racket-v4'

    def __init__(self,*args,**kwargs):
        self.residual_rewards=None
        super().__init__(*args,**kwargs)
        n,device=self.num_envs,self.device
        self.residual_rewards=ResidualEventRewardV4(n,device)
        self.reward_target=torch.zeros((n,3),device=device)
        self.reward_event_time=torch.full((n,),2.75,device=device)
        self.reward_plan_valid=torch.zeros(n,dtype=torch.bool,device=device)
        self.residual_raw=torch.zeros((n,18),device=device)
        self.reward_velocity=torch.zeros(3,device=device)
        self.reward_normal=torch.tensor([1.,0.,0.],device=device)
        self.reward_offset=torch.tensor([.0256,0.,.1],device=device)
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
        dense=(position_term+normal_term)*active
        cost=.00002*d.applied_torque.square().sum(1)+.005*self.action_delta+.02*self.residual_raw.square().sum(1)
        reward=self.residual_rewards.step(flags,dense,cost,self.step_dt)
        self.last_residual_reward=reward.detach().clone()
        return reward

    def _reset_idx(self,ids):
        super()._reset_idx(ids)
        if self.residual_rewards is not None:self.residual_rewards.reset(ids)
