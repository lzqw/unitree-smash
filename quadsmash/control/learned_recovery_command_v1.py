"""Learned native-position recovery head with causal, per-episode state.

The frozen base controller remains outside this module. A new normalized delta
coordinate spans the original joint range; this is not the legacy .4 rad action.
No runtime optimizer, outcome label or case identifier is an input.
"""
import torch
from torch import nn

SCHEMA='learned-native-recovery-command-v1'
INPUT_SIZE=831  # public794 + base18 + actual previous18 + recovery elapsed1


def command_features(public,base,previous,elapsed):
    return torch.cat((public,base,previous,elapsed[:,None]),dim=-1)


def compose_native_delta(base,delta,previous,active,limits,arm_rate=10.,dt=.01):
    candidate=torch.clamp(base+2.*delta,-1.,1.)
    step=2.*arm_rate*dt/(limits[12:,1]-limits[12:,0])
    arm=torch.maximum(torch.minimum(candidate[:,12:],previous[:,12:]+step),previous[:,12:]-step)
    candidate=torch.cat((candidate[:,:12],torch.clamp(arm,-1.,1.)),dim=1)
    return torch.where(active[:,None],candidate,base)


class RecoveryCommandHead(nn.Module):
    def __init__(self,mean,scale,hidden=128):
        super().__init__()
        mean=torch.as_tensor(mean,dtype=torch.float32);scale=torch.as_tensor(scale,dtype=torch.float32)
        if (mean.shape!=(INPUT_SIZE,) or scale.shape!=mean.shape or
                not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or torch.any(scale<=0)):
            raise ValueError('Fixed finite feature normalization required')
        self.register_buffer('feature_mean',mean.clone());self.register_buffer('feature_scale',scale.clone())
        self.hidden=int(hidden)
        self.network=nn.Sequential(nn.Linear(INPUT_SIZE,self.hidden),nn.ELU(),
            nn.Linear(self.hidden,self.hidden),nn.ELU(),nn.Linear(self.hidden,18))
        nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)

    def forward(self,features):
        normalized=torch.clamp((features-self.feature_mean)/self.feature_scale,-8.,8.)
        return torch.tanh(self.network(normalized))

    @classmethod
    def from_checkpoint(cls,checkpoint,device='cpu'):
        if checkpoint.get('schema')!=SCHEMA:raise ValueError('Wrong recovery command schema')
        state=checkpoint['head_state_dict']
        model=cls(state['feature_mean'],state['feature_scale'],checkpoint['hidden'])
        model.load_state_dict(state,strict=True)
        if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
            raise ValueError('Nonfinite recovery checkpoint')
        return model.to(device).eval()


def admit_native_recovery_head(checkpoint,base_sha,protocol_sha,limits,control_dt):
    """Reject different base/interface versions before any learned execution."""
    if (checkpoint.get('base_policy_sha256')!=base_sha or
            checkpoint.get('protocol_sha256')!=protocol_sha):
        raise ValueError('Recovery head belongs to a different frozen base or protocol')
    declared=torch.as_tensor(checkpoint.get('joint_limits_rad',[]),dtype=torch.float64)
    actual=torch.as_tensor(limits,dtype=torch.float64).cpu()
    if declared.shape!=(18,2) or not torch.allclose(declared,actual,atol=1e-6,rtol=0.):
        raise ValueError('Recovery head native joint limits differ')
    if (checkpoint.get('control_dt_s')!=control_dt or control_dt!=.01 or
            checkpoint.get('arm_target_rate_rad_s')!=10.):
        raise ValueError('Recovery head command period or slew differs')
    return RecoveryCommandHead.from_checkpoint(checkpoint)


class NativeRecoveryCommandAdapter:
    def __init__(self,head,count,device,limits,*,arm_rate=10.,dt=.01):
        self.head=head.to(device).eval();self.count=int(count);self.device=device
        self.limits=torch.as_tensor(limits,device=device,dtype=torch.float32)
        if (self.limits.shape!=(18,2) or not torch.isfinite(self.limits).all() or
                torch.any(self.limits[:,0]>=self.limits[:,1]) or arm_rate<=0 or dt<=0):
            raise ValueError('Original native limits and command period required')
        self.arm_rate=float(arm_rate);self.dt=float(dt)
        self.previous=torch.zeros((count,18),device=device)
        self.last_clock=torch.zeros(count,device=device)
        self.previous_known=torch.zeros(count,dtype=torch.bool,device=device)
        self.observed_hit=torch.zeros(count,dtype=torch.bool,device=device)
        self.active=torch.zeros(count,dtype=torch.bool,device=device)
        self.onset=torch.zeros(count,device=device)
        self.last_delta=torch.zeros((count,18),device=device)

    def reset(self,mask=None):
        if mask is None:mask=torch.ones(self.count,dtype=torch.bool,device=self.device)
        self.previous[mask]=0.;self.last_clock[mask]=0.;self.previous_known[mask]=False
        self.observed_hit[mask]=False;self.active[mask]=False;self.onset[mask]=0.;self.last_delta[mask]=0.

    @torch.inference_mode()
    def action(self,public,base):
        if (public.shape!=(self.count,794) or base.shape!=(self.count,18) or
                not torch.isfinite(public).all() or not torch.isfinite(base).all() or
                torch.any(abs(base)>1.+1e-6)):
            raise ValueError('Finite original public inputs and native base commands required')
        clock=public[:,775]
        reset=self.previous_known & (clock<self.last_clock-1e-6)
        self.reset(reset)
        if torch.any((~self.previous_known)&(abs(clock)>1e-6)):
            raise ValueError('Recovery adapter must start from the real ground reset')
        if torch.any(self.previous_known & (abs(clock-self.last_clock-self.dt)>2e-6)):
            raise ValueError('Recovery adapter requires exactly one call per native control step')
        entered=(~self.active)&self.observed_hit&(public[:,774]>.5)&self.previous_known
        self.onset[entered]=clock[entered];self.active|=entered
        self.observed_hit|=public[:,773]>.5
        self.last_delta.zero_();output=base.clone()
        if self.active.any():
            lanes=self.active
            features=command_features(public[lanes],base[lanes],self.previous[lanes],clock[lanes]-self.onset[lanes])
            self.last_delta[lanes]=self.head(features)
            output=compose_native_delta(base,self.last_delta,self.previous,self.active,self.limits,self.arm_rate,self.dt)
        self.previous.copy_(output);self.previous_known[:]=True;self.last_clock.copy_(clock)
        return output
