"""Ball-aware execution of a learned reach controller, with causal prediction."""
import json
from pathlib import Path
import numpy as np
import torch
from .full_racket_teacher_v4 import FullRacketTeacherV4
from .full_racket_geometry_v4 import FullRacketGeometryV4
from .initial_intercept import predict_initial_intercept
from .full_racket_arm_search_v4 import causal_prepose_delta
from .full_racket_relative_prepose_v4 import validate_relative_prepose,relative_prepose_delta
from .full_racket_height_intercept_v4 import validate_height_intercept,predict_height_intercept
from .contact_first_trajectory_search_v1 import (
    ContactFirstSearchConfig, ContactFirstTrajectorySearchV1,
)
from .full_racket_contact_normal_search_v4 import contact_normal_body

ROOT=Path(__file__).resolve().parents[2]


class FullRacketRallyControllerV4:
    def __init__(self,checkpoint,count,device):
        self.checkpoint=checkpoint
        self.teacher=FullRacketTeacherV4(checkpoint['backbone'],checkpoint['seed'],
            ROOT/'configs/g07_go2_walk_transfer.json',count,device)
        self.base=checkpoint['parameters'].to(device)
        self.parameters=self.base[None].repeat(count,1)
        self.event_time=torch.full((count,),2.75,device=device)
        self.planned=torch.zeros(count,dtype=torch.bool,device=device)
        self.physics=json.loads((ROOT/'configs/shuttle_physics_v2.json').read_text())
        model=FullRacketGeometryV4();q=np.array(checkpoint['seed']['joint_position_rad'])
        self.nominal_target=np.array(checkpoint['seed'].get('desired_ball_center_m',[1.10,0.,1.80]))
        self.jacobian=model.contact_jacobian(q,uv=checkpoint['seed'].get('string_uv_m',[0.,0.]),face=checkpoint['seed'].get('face_sign',1),base_position=(0,0,0))
        self.inverse=self.jacobian.T@np.linalg.inv(self.jacobian@self.jacobian.T+.04**2*np.eye(3))
        self.logs=[]
        self.prepose_gain=float(checkpoint['seed'].get('causal_arm_prepose_gain',0.))
        if not 0.<=self.prepose_gain<=1.:raise ValueError('Causal preparation gain must be in[0,1]')
        self.relative_prepose=validate_relative_prepose(checkpoint['seed'].get('relative_contact_prepose'))
        self.height_intercept=validate_height_intercept(checkpoint['seed'].get('height_intercept'))
        self.target_offset_world=torch.zeros((count,3),device=device)
        # Optional per-lane event target installed by an event generator.  A
        # None value preserves the original shared teacher path bit-for-bit.
        self.event_velocity_world=None
        # This planner is an explicit opt-in interface.  The historical
        # action path remains bit-for-bit unchanged when the seed has no
        # contact_first_search block.  A caller can use the current DTG event
        # and measured six-axis Jacobians through plan_contact_first_arm().
        search_config=checkpoint['seed'].get('contact_first_search')
        self.contact_first_planner=None
        self.contact_first_plan_q=None
        self.contact_first_plan_start_time=None
        self.contact_first_plan_contact_time=None
        self.contact_first_plan_valid=None
        self.contact_first_plan_log=[]
        if search_config is not None:
            if not isinstance(search_config,dict):
                raise ValueError('contact_first_search must be a mapping')
            allowed=set(ContactFirstSearchConfig.__dataclass_fields__)
            unknown=set(search_config)-allowed
            if unknown:
                raise ValueError(f'Unknown contact_first_search keys: {sorted(unknown)}')
            self.contact_first_planner=ContactFirstTrajectorySearchV1(
                ContactFirstSearchConfig(**search_config))
            horizon=self.contact_first_planner.config.horizon_steps
            self.contact_first_plan_q=torch.zeros((count,horizon,6),device=device)
            self.contact_first_plan_start_time=torch.full((count,),float('inf'),device=device)
            self.contact_first_plan_contact_time=torch.zeros((count,),device=device)
            self.contact_first_plan_valid=torch.zeros(count,dtype=torch.bool,device=device)

    def reset(self,ids=None):
        self.teacher.reset(ids)
        if ids is None:
            self.parameters.copy_(self.base[None])
            self.event_time.fill_(2.75);self.planned.zero_();self.logs=[]
            self.target_offset_world.zero_()
            self.event_velocity_world=None
            self.teacher.velocity_world_override=None
        else:
            self.parameters[ids]=self.base
            self.event_time[ids]=2.75;self.planned[ids]=False
            self.target_offset_world[ids]=0.
            if self.event_velocity_world is not None:
                self.event_velocity_world[ids]=0.
            if getattr(self,'contact_first_plan_valid',None) is not None:
                self.contact_first_plan_valid[ids]=False
                self.contact_first_plan_start_time[ids]=float('inf')
        if ids is None and getattr(self,'contact_first_plan_valid',None) is not None:
            self.contact_first_plan_valid.zero_()
            self.contact_first_plan_start_time.fill_(float('inf'))
            self.contact_first_plan_contact_time.zero_()
            self.contact_first_plan_log=[]

    def plan_contact_first_arm(self, **kwargs):
        """Plan a causal pre-contact arm trajectory from a DTG event.

        This method is intentionally separate from ``action`` until the
        contact-prefix evaluator is supplied by the full-body teacher.  It
        prevents an unverified kinematic plan from silently replacing the
        existing controller while providing the real Approach integration
        point for CPU and native pre-contact validation.
        """
        if self.contact_first_planner is None:
            raise RuntimeError('contact_first_search is not enabled in this checkpoint')
        return self.contact_first_planner.plan(**kwargs)

    @torch.inference_mode()
    def _install_contact_first_plan(self, ids, public, actual, forward, left, up,
                                    target, normal_target, desired_velocity, event_time,
                                    limits):
        """Build one causal pre-contact arm plan per newly planned lane."""
        if self.contact_first_planner is None:
            return
        limits_np=limits.detach().cpu().numpy()
        q_limits=limits_np[12:]
        cfg=self.contact_first_planner.config
        point,normal,jp,jn=self.teacher.kinematics(actual,self.teacher.contact_offset)
        for local,lane in enumerate(ids.tolist()):
            contact_time=float(event_time[local].item()-public[local,775])
            try:
                def kinematics_fn(q):
                    q_tensor=torch.as_tensor(q,device=actual.device,dtype=actual.dtype)[None]
                    pose=self.teacher.kinematics(q_tensor,self.teacher.contact_offset)
                    return tuple(value[0].detach().cpu().numpy() for value in pose)

                result=self.contact_first_planner.plan(
                    q_rad=actual[local].detach().cpu().numpy(),
                    qd_rad_s=np.zeros(6),
                    position_jacobian=jp[local].detach().cpu().numpy(),
                    normal_jacobian=jn[local].detach().cpu().numpy(),
                    racket_position_world_m=point[local].detach().cpu().numpy(),
                    racket_normal_world=normal[local].detach().cpu().numpy(),
                    target_position_world_m=target[local].detach().cpu().numpy(),
                    target_normal_world=normal_target[local].detach().cpu().numpy(),
                    target_velocity_world_m_s=desired_velocity[local].detach().cpu().numpy(),
                    contact_time_s=contact_time,
                    base_linear_velocity_world_m_s=public[local,:3],
                    base_angular_velocity_world_rad_s=public[local,3:6],
                    racket_offset_world_m=point[local].detach().cpu().numpy(),
                    lower_rad=q_limits[:,0], upper_rad=q_limits[:,1],
                    velocity_limits_rad_s=np.full(6,cfg.arm_velocity_limit_rad_s),
                    kinematics_fn=kinematics_fn,
                )
            except (ValueError, RuntimeError, FloatingPointError) as exc:
                self.contact_first_plan_valid[lane]=False
                self.contact_first_plan_log.append(dict(
                    lane=lane, accepted=False, rejection=type(exc).__name__,
                    error=str(exc), contact_time_s=contact_time))
                continue
            self.contact_first_plan_q[lane]=torch.as_tensor(
                result.q_rad,device=actual.device,dtype=actual.dtype)
            self.contact_first_plan_start_time[lane]=float(public[local,775])
            self.contact_first_plan_contact_time[lane]=result.contact_time_s
            self.contact_first_plan_valid[lane]=True
            self.contact_first_plan_log.append(dict(
                lane=lane, accepted=True, contact_index=result.contact_index,
                contact_time_s=result.contact_time_s, score=result.score,
                position_error_m=result.position_error_m,
                normal_error=result.normal_error,
                velocity_error_m_s=result.velocity_error_m_s,
                metadata=result.metadata))

    @torch.inference_mode()
    def _overlay_contact_first_plan(self, action, public, limits):
        """Use the optional contact prefix only until the planned hit."""
        if self.contact_first_planner is None or not self.contact_first_plan_valid.any():
            return action
        cfg=self.contact_first_planner.config
        time=public[:,775]
        elapsed=time-self.contact_first_plan_start_time
        index=torch.round(elapsed/cfg.dt_s).long().clamp(
            0,cfg.horizon_steps-1)
        plan=self.contact_first_plan_q[
            torch.arange(len(public),device=public.device),index]
        active=self.contact_first_plan_valid & (elapsed>=0.) & (
            elapsed<=self.contact_first_plan_contact_time+cfg.dt_s)
        normalized=2.*(plan-limits[12:,0])/(limits[12:,1]-limits[12:,0])-1.
        action[:,12:]=torch.where(active[:,None],normalized,action[:,12:])
        return action

    @torch.inference_mode()
    def action(self,observation,default,limits):
        obs=observation['policy']
        ids=torch.where((~self.planned)&(obs[:,67]>.5)&(obs[:,775]>=2.0))[0]
        if len(ids):
            public=obs[ids].cpu().numpy()
            # The extra learned parameters choose a legal interception plane
            # and a timing residual. Both are fixed policy parameters, never
            # case labels or future simulator states.
            intercept=self.parameters[ids,24].cpu().numpy() if self.parameters.shape[1]>=25 else self.nominal_target[0]
            if self.height_intercept is None:
                plan=predict_initial_intercept(public,[1.10,0,1.80],[1.,0,0],intercept,self.physics,
                                               dt=.001,max_seconds=1.5)
            else:
                plan=predict_height_intercept(public,[1.10,0,1.80],[1.,0,0],intercept,self.physics,
                    self.height_intercept,dt=.001,max_seconds=1.5)
            arrival=public[:,775]+plan['arrival_time_s']
            # Correct the measured no-ball event offset, learned before this
            # episode, using only the current observation's ball prediction.
            calibration=self.checkpoint.get('event_calibration_s',2.75-self.checkpoint['result']['event_time_s'])
            residual=self.parameters[ids,25].cpu().numpy() if self.parameters.shape[1]>=26 else 0.
            calibrated=arrival+calibration+residual
            valid=plan['valid']&np.isfinite(calibrated)&(calibrated>=2.3)&(calibrated<=3.3)
            self.event_time[ids]=torch.as_tensor(np.where(valid,calibrated,2.75),device=obs.device,dtype=obs.dtype)
            delta=plan['position_m']-np.array(self.checkpoint['result']['virtual_ball_center_m'])
            correction=np.clip(np.nan_to_num(delta)@self.inverse.T,-.30,.30)
            correction[~valid]=0.
            applied_prepose=np.zeros((len(ids),6))
            if self.parameters.shape[1]<20:
                self.parameters[ids,8:14]+=torch.as_tensor(correction,device=obs.device,dtype=obs.dtype)
            else:
                target_delta=plan['position_m']-self.nominal_target
                target_delta[~valid]=0.
                self.target_offset_world[ids]=torch.as_tensor(target_delta,device=obs.device,dtype=obs.dtype)
                if self.prepose_gain:
                    before=self.parameters[ids,8:14].clone()
                    raw_hit=self.teacher.arm[None]+before
                    bounded_hit=raw_hit.clamp(limits[12:,0],limits[12:,1])
                    _,_,position_jacobian,normal_jacobian=self.teacher.kinematics(bounded_hit,self.teacher.contact_offset)
                    lower_hit=torch.maximum(torch.maximum(limits[12:,0],self.teacher.arm[None]-.25),raw_hit-.30)
                    upper_hit=torch.minimum(torch.minimum(limits[12:,1],self.teacher.arm[None]+.25),raw_hit+.30)
                    bounds=((lower_hit-bounded_hit).cpu().numpy(),(upper_hit-bounded_hit).cpu().numpy())
                    if self.relative_prepose is None:
                        prepose=causal_prepose_delta(target_delta,public,position_jacobian.cpu().numpy(),normal_jacobian.cpu().numpy(),valid,self.prepose_gain,*bounds)
                    else:
                        prepose=relative_prepose_delta(plan['position_m'],public,self.relative_prepose['reference_target_body_m'],
                            position_jacobian.cpu().numpy(),normal_jacobian.cpu().numpy(),valid,self.prepose_gain,*bounds)
                    proposed=bounded_hit+torch.as_tensor(prepose,device=obs.device,dtype=obs.dtype)-self.teacher.arm[None]
                    # Invalid forecasts retain the original controller path.
                    apply=valid if self.relative_prepose is None else valid & np.any(prepose!=0.,axis=1)
                    # A zero relative correction also preserves an existing
                    # raw target above a position bound; the action decoder
                    # still clamps it as before. Replacing it with the bound
                    # here would change the measured reference trajectory.
                    after=torch.where(torch.as_tensor(apply,device=obs.device)[:,None],proposed,before)
                    self.parameters[ids,8:14]=after
                    applied_prepose=(after-before).cpu().numpy()
            self.planned[ids]=True
            if self.contact_first_planner is not None:
                # Reuse exactly the same body-frame target, face normal and
                # desired velocity that FullRacketTeacherV4 will use below.
                # The planner therefore receives the DTG event once and does
                # not read future simulator state.
                forward=obs[ids,763:766]
                up=-obs[ids,6:9]
                left=torch.linalg.cross(up,forward)
                target_body=obs[ids,760:763].clone()
                known_offset=self.teacher.target_world-obs.new_tensor([1.10,0.,1.80])
                target_body+=known_offset[0]*forward+known_offset[1]*left+known_offset[2]*up
                target_body+=self.target_offset_world[ids,:1]*forward
                target_body+=self.target_offset_world[ids,1:2]*left
                target_body+=self.target_offset_world[ids,2:3]*up
                normal_target=contact_normal_body(
                    self.parameters[ids],self.teacher.normal_world,forward,left,up)
                velocity_world=self.teacher.velocity_world
                if self.event_velocity_world is not None:
                    velocity_world=self.event_velocity_world[ids]
                if velocity_world.ndim==1:
                    velocity_world=velocity_world[None].expand(len(ids),-1)
                desired_velocity=(velocity_world[:,0:1]*forward+
                                  velocity_world[:,1:2]*left+
                                  velocity_world[:,2:3]*up)
                actual=default[12:][None]+obs[ids,21:27]
                self._install_contact_first_plan(
                    ids,public,actual,forward,left,up,target_body,
                    normal_target,desired_velocity,self.event_time[ids],limits)
            self.logs.append(dict(input='Current public794D observation, fixed known command and learned no-ball calibration only',
                env_ids=ids.cpu().tolist(),observation_clock_s=public[:,775].tolist(),
                used_public_observations=public.tolist(),calibrated_event_time_s=calibrated.tolist(),
                arm_joint_correction_rad=correction.tolist(),causal_prepose_gain=self.prepose_gain,
                relative_contact_prepose=self.relative_prepose,
                height_intercept=self.height_intercept,
                causal_prepose_joint_delta_rad=applied_prepose.tolist(),**{k:v.tolist() for k,v in plan.items()}))
        action=self.teacher.action(observation,self.parameters,default,limits,event_time=self.event_time,
                                   target_offset_world=self.target_offset_world)
        return self._overlay_contact_first_plan(action,obs,limits)
