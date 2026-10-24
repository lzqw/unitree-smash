"""Actual ground-reset disturbances shared by native smoke and formal panels."""
import torch
from .full_racket_env_v4 import FullRacketEnvV4


class FrozenPanelEnv(FullRacketEnvV4):
    def __init__(self,*arguments,reset_noise,**keywords):
        self._evaluation_ready=False;self.reset_noise=reset_noise;self.initial_states=[]
        super().__init__(*arguments,**keywords);self._evaluation_ready=True

    @torch.inference_mode()
    def _reset_idx(self,ids):
        super()._reset_idx(ids)
        if not self._evaluation_ready:return
        data=self.robot.data;device=self.device
        indices=ids.cpu().tolist();noise=self.reset_noise
        q=data.default_joint_pos[ids].clone();qd=data.default_joint_vel[ids].clone()
        q[:,self.joint_ids]+=torch.tensor(noise['joint_position_rad'][indices],dtype=q.dtype,device=device)
        qd[:,self.joint_ids]+=torch.tensor(noise['joint_velocity_rad_s'][indices],dtype=qd.dtype,device=device)
        limits=data.joint_pos_limits[ids];q.clamp_(min=limits[...,0],max=limits[...,1])
        root=data.default_root_state[ids].clone();root[:,:3]+=self.scene.env_origins[ids]
        root[:,:2]+=torch.tensor(noise['root_xy_m'][indices],dtype=root.dtype,device=device)
        yaw=torch.tensor(noise['root_yaw_rad'][indices],dtype=root.dtype,device=device)
        root[:,3:7]=torch.stack(((yaw/2).cos(),torch.zeros_like(yaw),torch.zeros_like(yaw),(yaw/2).sin()),1)
        self.robot.write_root_state_to_sim(root,env_ids=ids)
        self.robot.write_joint_state_to_sim(q,qd,env_ids=ids);self.sim.forward();self.policy_history.reset(ids)
        self.reset_com_height[ids]=(self.mass_weights[ids]*data.body_com_pos_w[ids,:,2]).sum(1)
        for j,i in enumerate(indices):
            if int(self.episode_index[i])==0:
                self.initial_states.append(dict(env_id=i,root_state_local=torch.cat((root[j,:3]-self.scene.env_origins[i],root[j,3:])).cpu().tolist(),
                    joint_position_rad=q[j,self.joint_ids].cpu().tolist(),joint_velocity_rad_s=qd[j,self.joint_ids].cpu().tolist()))
