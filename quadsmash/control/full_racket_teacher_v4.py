"""CEM-adapted whole-body controller; public observations and fixed commands only."""
import torch
from .go2_walk_transfer import Go2WalkTransfer
from .go2_standing_transfer import Go2StandingTransfer
from .piecewise_affine import PiecewiseAffinePolicy

LOW=[.35,.45,1.10,1.55,.85,-.15,-.15,.30]+[-.25]*6+[-.10,0.,.8,.7]
HIGH=[.85,.90,1.60,2.,1.15,.15,.15,1.50]+[.25]*6+[.12,1.5,1.7,1.2]
INITIAL=[.49,.70,1.35,1.72,1.,0.,0.,1.]+[0.]*6+[.04,.7,1.45,1.]


class FullRacketTeacherV4:
    def __init__(self,backbone,seed,manifest,count,device):
        self.walk=Go2WalkTransfer(manifest,count,device)
        self.standing=Go2StandingTransfer(manifest,count,device)
        self.strike=PiecewiseAffinePolicy(backbone['strike_policy_checkpoint']).to(device).eval()
        self.walk_seed=backbone['walk_parameters'].to(device)
        self.standing_seed=backbone['recovery_parameters'].to(device)
        self.arm=torch.tensor(seed['joint_position_rad'],dtype=torch.float32,device=device)
        self.arm_velocity=torch.tensor(seed['joint_velocity_seed_rad_s'],dtype=torch.float32,device=device)
        self.pitch_basis=torch.tensor(seed['pitch_compensation_basis_rad_per_rad'],dtype=torch.float32,device=device)

    def reset(self):
        self.walk.reset();self.standing.reset()

    @torch.inference_mode()
    def action(self,observation,parameters,default,limits,event_time=None):
        obs=observation['policy'];time=obs[:,775]
        event_time=torch.full_like(time,2.75) if event_time is None else event_time
        local=(time-parameters[:,2]-(event_time-2.75))*parameters[:,4]
        walk_obs=obs.clone()
        # Convert the public known head command to the public walking waypoint.
        # Command normal is world +X in the body frame; gravity supplies -Z.
        forward,up=obs[:,763:766],-obs[:,6:9]
        walk_obs[:,760:763]-=(1.10-parameters[:,1:2])*forward+1.50*up
        walk_parameters=self.walk_seed[None].repeat(len(obs),1)
        walk_parameters[:,0]=parameters[:,0]
        walk_parameters[:,7]*=parameters[:,17]
        walk=self.walk.action({'policy':walk_obs},walk_parameters,default,limits,maximum_time=3.)
        strike_obs=obs.clone();strike_obs[:,775]=local.clamp_min(0.)
        strike=self.strike(strike_obs)
        q=limits[:,0]+.5*(strike+1)*(limits[:,1]-limits[:,0])
        push=((local-.75)/.15).clamp(0,1)*((1.60-local)/.2).clamp(0,1)
        q[:,4:8]+=parameters[:,5:6]*push[:,None]
        q[:,8:12]+=parameters[:,6:7]*push[:,None]
        arm_hit=self.arm[None]+parameters[:,8:14]
        velocity=self.arm_velocity[None]*parameters[:,7:8]
        center=event_time-parameters[:,14]
        arm_pre=arm_hit-.12*velocity
        blend=((time-.20)/(parameters[:,16]-.20)).clamp(0,1)
        arm=torch.lerp(default[12:][None],arm_pre,blend[:,None])
        swinging=time>=center-.12
        trajectory=arm_hit+velocity*(time-center).clamp(-.12,.12)[:,None]
        arm=torch.where(swinging[:,None],trajectory,arm)
        pitch=torch.atan2(obs[:,6],-obs[:,8])
        feedback=((time-event_time+.45)/.15).clamp(0,1)*((event_time+.45-time)/.20).clamp(0,1)
        arm+=parameters[:,15:16]*pitch[:,None]*feedback[:,None]*self.pitch_basis[None]
        fold=((time-(center+.12))/.50).clamp(0,1)
        arm=torch.lerp(arm,default[12:][None],fold[:,None])
        q[:,12:]=arm
        strike=(2*(q-limits[:,0])/(limits[:,1]-limits[:,0])-1).clamp(-1,1)
        walking=local<.75
        action=torch.where(walking[:,None],walk,strike)
        # Both walking and strike use the same newly planned real arm target.
        action[:,12:]=strike[:,12:]
        recovery=local>=parameters[:,3]
        standing=self.standing.action(observation,self.standing_seed[None].expand(len(obs),-1),
            default,limits,default[12:][None].expand(len(obs),-1),active=recovery)
        return torch.where(recovery[:,None],standing,action).clamp(-1,1)
