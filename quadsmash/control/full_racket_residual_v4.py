"""Public-observation residual commands on the unchanged full-racket robot."""
import io
import torch
from .full_racket_arm_target_rate_v4 import ArmTargetRateV4
from .full_racket_rally_controller_v4 import FullRacketRallyControllerV4

SCHEMA='full-racket-residual-ppo-v4'


class ResidualActionMixerV4:
    def __init__(self,count,device,joint_scale_rad):
        self.scale=torch.as_tensor(joint_scale_rad,dtype=torch.float32,device=device)
        if self.scale.shape!=(18,) or not torch.isfinite(self.scale).all() or (self.scale<=0).any() or (self.scale>.25).any():
            raise ValueError('Require18 positive finite residual scales no larger than.25rad')
        self.arm_rate=ArmTargetRateV4(count,device,5.)
        self.nonzero_seen=torch.zeros(count,dtype=torch.bool,device=device)
        self.last_requested=None;self.last_actual=None

    def reset(self,ids=None):
        self.arm_rate.reset(ids)
        if ids is None:self.nonzero_seen.zero_()
        else:self.nonzero_seen[ids]=False

    @torch.inference_mode()
    def action(self,base,residual,public,default,limits):
        if base.shape!=residual.shape or base.shape!=(len(self.nonzero_seen),18):
            raise ValueError('Require one18D base and residual command per lane')
        if not torch.isfinite(base).all() or not torch.isfinite(residual).all():
            raise ValueError('Finite residual/base commands required')
        residual=residual.clamp(-1.,1.)
        self.nonzero_seen|=(residual!=0).any(1)
        # Joint command changes are converted through the same original
        # joint range. Every resulting command still enters native actuators.
        requested=(base+2*residual*self.scale/(limits[:,1]-limits[:,0])).clamp(-1.,1.)
        desired=limits[:,0]+.5*(requested+1)*(limits[:,1]-limits[:,0])
        actual_arm=(default[12:][None]+public[:,21:27]).clamp(limits[12:,0],limits[12:,1])
        bounded=self.arm_rate(desired[:,12:],actual_arm,public[:,775])
        action=requested.clone()
        action[:,12:]=(2*(bounded-limits[12:,0])/(limits[12:,1]-limits[12:,0])-1).clamp(-1.,1.)
        # A fresh, identically zero residual is exactly the existing policy.
        # After a nonzero command the slew state must still be honored.
        action=torch.where(self.nonzero_seen[:,None],action,base)
        self.last_requested=requested.detach().clone();self.last_actual=action.detach().clone()
        return action


class FullRacketResidualControllerV4:
    def __init__(self,checkpoint,count,device):
        if checkpoint.get('schema')!=SCHEMA:
            raise ValueError('Unknown residual checkpoint')
        if checkpoint['asset_sha256']!=checkpoint['base_controller']['asset_sha256'] or not torch.equal(checkpoint['parameters'],checkpoint['base_controller']['parameters']):
            raise ValueError('Residual base/controller contract mismatch')
        if checkpoint['base_controller']['seed'].get('arm_target_rate_limit_rad_s')!=5.:
            raise ValueError('Residual baseline must use the same5rad/s arm target limiter')
        self.base=FullRacketRallyControllerV4(checkpoint['base_controller'],count,device)
        self.actor=torch.jit.load(io.BytesIO(checkpoint['residual_actor_jit']),map_location=device).eval()
        self.mixer=ResidualActionMixerV4(count,device,checkpoint['residual_joint_scale_rad'])

    @property
    def logs(self):return self.base.logs

    def reset(self,ids=None):
        self.base.reset(ids);self.mixer.reset(ids)

    @torch.inference_mode()
    def action(self,observation,default,limits):
        residual=self.actor(observation['policy'])
        base=self.base.action(observation,default,limits)
        return self.mixer.action(base,residual,observation['policy'],default,limits)


def make_rally_controller(checkpoint,count,device):
    if checkpoint.get('schema')==SCHEMA:
        return FullRacketResidualControllerV4(checkpoint,count,device)
    if 'residual_actor_jit' in checkpoint:
        raise ValueError('Residual actor requires its explicit controller schema')
    return FullRacketRallyControllerV4(checkpoint,count,device)
