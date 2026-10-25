"""Pinned, learned Go2 flat-walk actor adapted to the native public observation.

The original 50 Hz actor is held for two 100 Hz control steps. The full-load
simulation and its joint/actuator limits are unchanged. No simulator state is
read here; the position command and proprioception are public observations.
"""
import hashlib
import json
from pathlib import Path
import torch


class Go2WalkTransfer:
    def __init__(self, manifest, num_envs, device):
        spec=json.loads(Path(manifest).read_text())
        path=Path(spec['policy_path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=spec['policy_sha256']:
            raise ValueError('Transferred walking actor hash mismatch')
        self.actor=torch.jit.load(str(path),map_location=device).eval()
        self.last_raw=torch.zeros((num_envs,12),device=device)
        self.reached=torch.zeros(num_envs,dtype=torch.bool,device=device)
        self.tick=0

    def reset(self, ids=None):
        if ids is None:
            self.last_raw.zero_()
            self.reached.zero_()
            self.tick=0
        else:
            self.last_raw[ids]=0.
            self.reached[ids]=False

    def action(self, observation, parameters, default, limits, *, heading_error=None,
               heading_kp=None, maximum_time=3.4, heading_readiness=True):
        obs=observation['policy']
        t=obs[:,775]
        speed,kp,kd=parameters[:,:3].unbind(1)
        error=obs[:,760]
        ready=(error.abs()<.04)&(obs[:,:2].norm(dim=1)<.12)
        if heading_error is not None and heading_readiness:
            ready &= (heading_kp<=0)|((heading_error.abs()<.03)&(obs[:,761].abs()<.04))
        self.reached|=ready
        delay=(parameters[:,14] if parameters.shape[1]==15 else
               parameters[:,10] if parameters.shape[1]>=11 else .15)
        ramp=((t-delay)/.35).clamp(0,1)
        commands=torch.zeros((len(t),3),device=t.device)
        commands[:,0]=(speed*torch.tanh(kp*error-kd*obs[:,0])).clamp(-.4,1.)*ramp
        commands[:,1]=(1.5*obs[:,761]-.3*obs[:,1]).clamp(-.3,.3)*ramp
        if heading_error is not None:
            commands[:,2]=(heading_kp*heading_error).clamp(-.6,.6)*ramp
        commands[self.reached]=0
        if self.tick%2==0:
            source=torch.cat((obs[:,:9],commands,obs[:,9:21],obs[:,27:39],self.last_raw),1)
            self.last_raw.copy_(self.actor(source))
        self.tick+=1
        q=default[None,:].repeat(len(t),1)
        gain=parameters[:,7:8] if parameters.shape[1]>=11 else 1.
        q[:,:12]+=.25*self.last_raw*gain
        if parameters.shape[1]>=11:
            q[:,4:8]+=parameters[:,3:4]
            q[:,4:6]+=parameters[:,4:5]
            q[:,6:8]-=parameters[:,4:5]
            q[:,8:12]+=parameters[:,5:6]
            q[:,:4]+=parameters[:,6:7]*obs[:,7:8]
            blend=((t-.1)/.45).clamp(0,1)
            if parameters.shape[1]==15:
                q[:,12:]=torch.lerp(default[12:][None,:],parameters[:,8:14],blend[:,None])
            else:
                q[:,13]=torch.lerp(default[13].expand_as(t),parameters[:,8],blend)
                q[:,14]=torch.lerp(default[14].expand_as(t),parameters[:,9],blend)
                if parameters.shape[1]>=12:
                    q[:,16]=torch.lerp(default[16].expand_as(t),parameters[:,11],blend)
        rest=q.clone()
        rest[:,:12]=default[:12]
        q=torch.where((self.reached|(t>=maximum_time))[:,None],rest,q)
        return (2*(q-limits[:,0])/(limits[:,1]-limits[:,0])-1).clamp(-1,1)
