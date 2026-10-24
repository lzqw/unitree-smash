"""CEM-adapted whole-body controller; public observations and fixed commands only."""
import torch
from .go2_walk_transfer import Go2WalkTransfer
from .go2_standing_transfer import Go2StandingTransfer
from .piecewise_affine import PiecewiseAffinePolicy
from .full_racket_kinematics_v4 import FullRacketKinematicsV4
from .full_racket_recovery_blend_v4 import LegRecoveryBlend
from .full_racket_arm_search_v4 import arm_velocity
from .full_racket_preparation_v4 import validate_preparation,preparation_curve,own_side_reference
from .full_racket_joint_preparation_v4 import FullRacketJointPreparation
from .full_racket_face_feedback_v4 import FullRacketFaceFeedback
from .full_racket_recovery_search_v4 import recovery_parameters
from .full_racket_contact_normal_search_v4 import contact_normal_body

LOW=[.35,.45,1.10,1.55,.85,-.15,-.15,.30]+[-.25]*6+[-.10,0.,.8,.7,0.,.04,-.20,-.15,-.15,-.15]
HIGH=[1.,1.20,1.60,2.,1.15,.15,.15,1.50]+[.25]*6+[.12,1.5,1.7,1.2,1.,.15,.10,.15,.15,.15]
INITIAL=[.49,.70,1.35,1.72,1.,0.,0.,1.]+[0.]*6+[.04,.7,1.45,1.,.5,.08,0.,0.,0.,0.]
PARAMETER_NAMES=['walk_speed','walk_waypoint_x','leg_phase_start','recovery_local_time','leg_clock_scale',
    'push_thigh_offset','push_calf_offset','arm_velocity_gain',*[f'arm_joint_{i}_offset' for i in range(1,7)],
    'arm_lead','arm_pitch_compensation_gain','arm_ready_time','walk_actor_gain','cartesian_gain',
    'swing_half_window','walk_thigh_offset','walk_calf_offset','push_front_back_thigh','push_front_back_calf']


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
        self.smooth_recovery_arm=seed.get('smooth_recovery_arm',False)
        self.nominal_pitch=float(seed.get('nominal_projected_pitch_rad',0.))
        self.leg_event_time_gain=float(seed.get('leg_event_time_gain',1.))
        self.leg_recovery_blend=LegRecoveryBlend(count,device,seed.get('leg_recovery_blend_s',0.))
        if not 0.<=self.leg_event_time_gain<=1.:raise ValueError('Leg event time gain must be in[0,1]')
        self.kinematics=FullRacketKinematicsV4().to(device=device,dtype=torch.float32)
        self.task_eye=torch.eye(6,device=device)[None]
        self.face_sign=seed.get('face_sign',1)
        uv=seed.get('string_uv_m',[0.,0.])
        self.contact_offset=torch.tensor([self.face_sign*.0256,*uv],dtype=torch.float32,device=device)
        self.target_world=torch.tensor(seed.get('desired_ball_center_m',[1.10,0.,1.80]),dtype=torch.float32,device=device)
        self.normal_world=torch.tensor(seed.get('outward_contact_normal_world',[.99503719,0.,-.09950372]),dtype=torch.float32,device=device)
        self.velocity_world=torch.tensor(seed.get('desired_point_velocity_mps',[1.5,0.,-.3]),dtype=torch.float32,device=device)
        self.preparation=validate_preparation(seed.get('cartesian_preparation'))
        self.joint_preparation=None
        self.face_feedback=None
        if seed.get('measured_face_feedback') is not None:
            self.face_feedback=FullRacketFaceFeedback(seed['measured_face_feedback']).to(device=device,dtype=torch.float32)
        if seed.get('joint_preparation_guard') is not None:
            if self.preparation is not None:raise ValueError('Use one preparation family per controller')
            self.joint_preparation=FullRacketJointPreparation(seed['joint_preparation_guard']).to(device=device,dtype=torch.float32)

    def reset(self):
        self.walk.reset();self.standing.reset();self.leg_recovery_blend.reset()

    @torch.inference_mode()
    def action(self,observation,parameters,default,limits,event_time=None,target_offset_world=None):
        obs=observation['policy'];time=obs[:,775]
        event_time=torch.full_like(time,2.75) if event_time is None else event_time
        local=(time-parameters[:,2]-self.leg_event_time_gain*(event_time-2.75))*parameters[:,4]
        walk_obs=obs.clone()
        # Convert the public known head command to the public walking waypoint.
        # Command normal is world +X in the body frame; gravity supplies -Z.
        forward,up=obs[:,763:766],-obs[:,6:9]
        walk_obs[:,760:763]-=(1.10-parameters[:,1:2])*forward+1.50*up
        walk_parameters=self.walk_seed[None].repeat(len(obs),1)
        walk_parameters[:,0]=parameters[:,0]
        walk_parameters[:,7]*=parameters[:,17]
        if parameters.shape[1]>=22:
            walk_parameters[:,3]+=parameters[:,20];walk_parameters[:,5]+=parameters[:,21]
        walk=self.walk.action({'policy':walk_obs},walk_parameters,default,limits,maximum_time=3.)
        strike_obs=obs.clone();strike_obs[:,775]=local.clamp_min(0.)
        strike=self.strike(strike_obs)
        q=limits[:,0]+.5*(strike+1)*(limits[:,1]-limits[:,0])
        push=((local-.75)/.15).clamp(0,1)*((1.60-local)/.2).clamp(0,1)
        q[:,4:8]+=parameters[:,5:6]*push[:,None]
        q[:,8:12]+=parameters[:,6:7]*push[:,None]
        if parameters.shape[1]>=24:
            q[:,4:6]+=parameters[:,22:23]*push[:,None];q[:,6:8]-=parameters[:,22:23]*push[:,None]
            q[:,8:10]+=parameters[:,23:24]*push[:,None];q[:,10:12]-=parameters[:,23:24]*push[:,None]
        arm_hit=self.arm[None]+parameters[:,8:14]
        velocity=arm_velocity(parameters,self.arm_velocity)
        center=event_time-parameters[:,14]
        half_window=parameters[:,19] if parameters.shape[1]>=20 else torch.full_like(time,.12)
        arm_pre=arm_hit-half_window[:,None]*velocity
        blend=((time-.20)/(parameters[:,16]-.20)).clamp(0,1)
        arm=torch.lerp(default[12:][None],arm_pre,blend[:,None])
        swinging=time>=center-half_window
        swing_clock=torch.maximum(torch.minimum(time-center,half_window),-half_window)
        trajectory=arm_hit+velocity*swing_clock[:,None]
        arm=torch.where(swinging[:,None],trajectory,arm)
        pitch=torch.atan2(obs[:,6],-obs[:,8])-self.nominal_pitch
        feedback=((time-event_time+.45)/.15).clamp(0,1)*((event_time+.45-time)/.20).clamp(0,1)
        arm+=parameters[:,15:16]*pitch[:,None]*feedback[:,None]*self.pitch_basis[None]
        if parameters.shape[1]>=20:
            # Correct the task in the current body frame from public command,
            # gravity, measured tool pose and actual joint angles. These are
            # bounded position targets; native actuators supply every force.
            actual=default[12:][None]+obs[:,21:27]
            point,_,jp,jn=self.kinematics(actual,self.contact_offset)
            left=torch.linalg.cross(up,forward)
            target=obs[:,760:763].clone()
            known_offset=self.target_world-torch.tensor([1.10,0.,1.80],device=time.device)
            target+=known_offset[0]*forward+known_offset[1]*left+known_offset[2]*up
            if target_offset_world is not None:
                target+=target_offset_world[:,:1]*forward+target_offset_world[:,1:2]*left+target_offset_world[:,2:3]*up
            normal_target=self.face_sign*contact_normal_body(parameters,self.normal_world,forward,left,up)
            desired_velocity=self.velocity_world[0]*forward+self.velocity_world[1]*left+self.velocity_world[2]*up
            if self.preparation is not None:
                desired_velocity=desired_velocity*parameters[:,7:8]
                known=torch.tensor([1.10,0.,1.80],device=time.device)
                offset=torch.tensor(self.preparation['position_world_m'],device=time.device)-known
                ready=obs[:,760:763]+offset[0]*forward+offset[1]*left+offset[2]*up
                task_target,desired_velocity,preparation_progress=preparation_curve(time,event_time,ready,target,desired_velocity,
                    self.preparation['transition_duration_s'])
                task_target,desired_velocity=own_side_reference(task_target,desired_velocity,obs[:,760:763],forward,
                    obs[:,773]>.5,self.preparation['unhit_reference_max_x_m'])
            # A60ms velocity lead compensates finite joint PD tracking lag.
            relative_command=desired_velocity-obs[:,:3]-torch.linalg.cross(obs[:,3:6],point)
            if self.preparation is None:
                error_position=target+desired_velocity*(time-event_time).clamp(-.15,.10)[:,None]+.06*relative_command-point
            else:
                guided,_=own_side_reference(task_target+.06*relative_command,desired_velocity,obs[:,760:763],forward,
                    obs[:,773]>.5,self.preparation['unhit_reference_max_x_m'])
                error_position=guided-point
            error_normal=.3*(normal_target-obs[:,48:51])
            jacobian=torch.cat((jp,.3*jn),1)
            error=torch.cat((error_position,error_normal),1)
            solved=torch.linalg.solve(jacobian@jacobian.transpose(1,2)+.04**2*self.task_eye,error[:,:,None])
            delta=(jacobian.transpose(1,2)@solved).squeeze(-1).clamp(-.4,.4)
            if self.preparation is not None:
                preferred=torch.lerp(default[12:][None],arm_hit,preparation_progress[:,None])
                bias=(preferred-actual).clamp(-.4,.4)
                projected=torch.linalg.solve(jacobian@jacobian.transpose(1,2)+.04**2*self.task_eye,
                    jacobian@bias[:,:,None])
                null=bias-(jacobian.transpose(1,2)@projected).squeeze(-1)
                delta=(delta+self.preparation['nullspace_gain']*null).clamp(-.4,.4)
            servo=actual+delta
            airborne=(obs[:,772]+obs[:,773])>.5
            correction=((time-event_time+.25)/.12).clamp(0,1)*((event_time+.12-time)/.12).clamp(0,1)
            gain=parameters[:,18]*correction*airborne
            if self.preparation is not None:
                preparation_gain=self.preparation['servo_gain']*((time-.20)/.50).clamp(0,1)
                gain=torch.maximum(gain,preparation_gain)
            arm=torch.lerp(arm,servo,gain[:,None])
        fold=((time-(center+half_window))/.50).clamp(0,1)
        arm=torch.lerp(arm,default[12:][None],fold[:,None])
        if self.joint_preparation is not None:
            actual_arm=default[12:][None]+obs[:,21:27]
            arm=self.joint_preparation(arm,obs,limits[12:],self.kinematics,actual_arm)
        if self.face_feedback is not None and bool((feedback>0).any()):
            left=torch.linalg.cross(up,forward)
            normal_target=self.face_sign*contact_normal_body(parameters,self.normal_world,forward,left,up)
            desired=normal_target+self.face_feedback.config['measured_normal_lead_gain']*(normal_target-obs[:,48:51])
            desired=desired/torch.linalg.vector_norm(desired,dim=1,keepdim=True).clamp_min(1e-6)
            reference=arm.clamp(limits[12:,0],limits[12:,1])
            correction=self.face_feedback(reference,desired,limits[12:],self.kinematics,self.contact_offset)
            amount=(parameters[:,15]*feedback).clamp(0,1)
            candidate=reference+amount[:,None]*correction
            if self.joint_preparation is not None:
                candidate=self.joint_preparation(candidate,obs,limits[12:],self.kinematics,actual_arm)
            point,normal,*_=self.kinematics(reference,self.contact_offset)
            new_point,new_normal,*_=self.kinematics(candidate,self.contact_offset)
            cfg=self.face_feedback.config
            keep=(torch.linalg.vector_norm(new_point-point,dim=1)<=cfg['maximum_point_displacement_m'])
            keep&=(candidate-reference).abs().amax(1)<=cfg['maximum_joint_delta_rad']+1e-6
            keep&=(desired-new_normal).square().sum(1)<=(desired-normal).square().sum(1)
            arm=torch.where((keep&(amount>0))[:,None],candidate,arm)
        q[:,12:]=arm
        strike=(2*(q-limits[:,0])/(limits[:,1]-limits[:,0])-1).clamp(-1,1)
        walking=local<.75
        action=torch.where(walking[:,None],walk,strike)
        # Both walking and strike use the same newly planned real arm target.
        action[:,12:]=strike[:,12:]
        recovery=local>=parameters[:,3]
        recovery_arm=arm if self.smooth_recovery_arm else default[12:][None].expand(len(obs),-1)
        standing=self.standing.action(observation,recovery_parameters(parameters,self.standing_seed),
            default,limits,recovery_arm,active=recovery)
        standing=self.leg_recovery_blend.action(time,recovery,obs[:,-18:],standing)
        return torch.where(recovery[:,None],standing,action).clamp(-1,1)
