"""Ball-aware execution of a learned reach controller, with causal prediction."""
import json
from pathlib import Path
import numpy as np
import torch
from .full_racket_teacher_v4 import FullRacketTeacherV4
from .full_racket_geometry_v4 import FullRacketGeometryV4
from .initial_intercept import predict_initial_intercept, propagate_ball_state
from .full_racket_arm_search_v4 import causal_prepose_delta
from .full_racket_relative_prepose_v4 import validate_relative_prepose,relative_prepose_delta
from .full_racket_height_intercept_v4 import validate_height_intercept,predict_height_intercept
from .contact_first_trajectory_search_v1 import (
    ContactFirstSearchConfig, ContactFirstTrajectorySearchV1,
    rotation_current_from_future_from_rotvec,
)
from .full_racket_contact_normal_search_v4 import contact_normal_body

ROOT=Path(__file__).resolve().parents[2]


def public_world_basis_in_body(public):
    """Columns are the fixed world x/y/z axes expressed in the current body.

    The locked rally command normal is world +x; gravity supplies world -z.
    This convention is specific to the existing rally observation contract.
    """
    forward, up = public[:, 763:766], -public[:, 6:9]
    return torch.stack((forward, torch.linalg.cross(up, forward), up), dim=-1)


def planning_world_pose(public, command_world):
    """World-from-current-body rotation and origin from the fixed rally command."""
    forward, up = public[763:766], -public[6:9]
    rotation = np.stack((forward, np.cross(up, forward), up), axis=0)
    if not np.allclose(rotation.T@rotation,np.eye(3),atol=2e-5):
        raise ValueError('Invalid public planning basis')
    return rotation, np.asarray(command_world)-rotation@public[760:763]



def causal_shuttle_target_in_planning_frame(physics, initial_position_world,
                                            initial_velocity_world,
                                            reference_incoming_velocity_world,
                                            world_from_planning, planning_origin_world,
                                            desired_racket_velocity_planning,
                                            duration_s):
    """One candidate-time shuttle target, in the *fixed* planning frame.

    The current intercept predictor supplies the causal shuttle state. Its
    nominal intercept delta is already present in the propagated position,
    so the legacy rally-controller prepose displacement MUST NOT be applied
    again here. Intentional racket geometry offsets belong in the contact FK,
    not in a second translation of the ball.
    """
    basis = np.asarray(world_from_planning, dtype=np.float64)
    origin = np.asarray(planning_origin_world, dtype=np.float64)
    initial_position = np.asarray(initial_position_world, dtype=np.float64)
    initial_velocity = np.asarray(initial_velocity_world, dtype=np.float64)
    incoming_reference = np.asarray(reference_incoming_velocity_world, dtype=np.float64)
    racket_velocity = np.asarray(desired_racket_velocity_planning, dtype=np.float64)
    if (basis.shape != (3, 3) or any(value.shape != (3,) for value in
            (origin, initial_position, initial_velocity, incoming_reference, racket_velocity)) or
            not all(np.isfinite(value).all() for value in
                    (basis, origin, initial_position, initial_velocity,
                     incoming_reference, racket_velocity))):
        raise ValueError('Finite shuttle and planning-frame inputs required')
    if not np.allclose(basis.T @ basis, np.eye(3), atol=2e-5):
        raise ValueError('World-from-planning basis must be orthonormal')
    ball_position, ball_velocity = propagate_ball_state(
        initial_position, initial_velocity, duration_s, physics)
    return dict(
        # Retain planner's historical field names; their explicit contract
        # is the fixed planning frame, not a changing future-base frame.
        position_world_m=basis.T @ (ball_position - origin),
        incoming_ball_velocity_world_m_s=basis.T @ ball_velocity,
        velocity_world_m_s=racket_velocity + basis.T @ (
            ball_velocity - incoming_reference),
    )


def transported_body_acceleration(current, previous, current_basis,
                                  previous_basis, delta_time, usable):
    """Inertial finite-difference acceleration expressed in the current body.

    Subtracting body components directly invents acceleration during rotation.
    First transport the previous vector B_old -> W -> B_now. No future state
    or angular-velocity integration is needed for this measured-state estimate.
    """
    previous_world = torch.bmm(previous_basis.transpose(1, 2), previous[:, :, None])
    previous_here = torch.bmm(current_basis, previous_world).squeeze(-1)
    safe_dt = torch.where(usable, delta_time, torch.ones_like(delta_time))
    acceleration = (current - previous_here) / safe_dt[:, None]
    return torch.where(usable[:, None], acceleration, torch.zeros_like(current))


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
        self.contact_first_evaluator=None
        self.contact_first_evaluator_factory=None
        # Anchored search is an offline reference generator. Wall time is
        # telemetry, not a trajectory admission criterion. An explicit offline
        # runner opts in; the ordinary native action loop keeps its fallback.
        self.contact_first_offline_diagnostic=False
        if search_config is not None:
            if not isinstance(search_config,dict):
                raise ValueError('contact_first_search must be a mapping')
            allowed=set(ContactFirstSearchConfig.__dataclass_fields__)
            unknown=set(search_config)-allowed
            if unknown:
                raise ValueError(f'Unknown contact_first_search keys: {sorted(unknown)}')
            self.contact_first_planner=ContactFirstTrajectorySearchV1(
                ContactFirstSearchConfig(**search_config))
            from .contact_replan_gate_v1 import ContactReplanGate
            self.contact_replan_gate=ContactReplanGate(count,device)
            horizon=self.contact_first_planner.config.horizon_steps
            self.contact_first_plan_q=torch.zeros((count,horizon,6),device=device)
            self.contact_first_joint_targets=torch.zeros((count,horizon,18),device=device)
            self.contact_first_joint_valid=torch.zeros(count,dtype=torch.bool,device=device)
            self.contact_first_recovery_owned=torch.zeros(count,dtype=torch.bool,device=device)
            self.contact_first_execution_log=[]
            self.contact_first_plan_start_time=torch.full((count,),float('inf'),device=device)
            self.contact_first_plan_contact_time=torch.zeros((count,),device=device)
            self.contact_first_plan_valid=torch.zeros(count,dtype=torch.bool,device=device)
            self.contact_first_last_actual_q=torch.zeros((count,6),device=device)
            self.contact_first_last_time=torch.full((count,),float('nan'),device=device)
            self.contact_first_last_valid=torch.zeros(count,dtype=torch.bool,device=device)
            self.contact_first_last_base_linear=torch.zeros((count,3),device=device)
            self.contact_first_last_base_angular=torch.zeros((count,3),device=device)
            self.contact_first_last_basis=torch.zeros((count,3,3),device=device)

    def reset(self,ids=None):
        self.teacher.reset(ids)
        if hasattr(self,"contact_replan_gate"):self.contact_replan_gate.reset(ids)
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
                if hasattr(self,'contact_first_joint_valid'):
                    self.contact_first_joint_valid[ids]=False
                    self.contact_first_recovery_owned[ids]=False
                self.contact_first_plan_start_time[ids]=float('inf')
                self.contact_first_last_valid[ids]=False
                self.contact_first_last_base_linear[ids]=0.
                self.contact_first_last_base_angular[ids]=0.
                self.contact_first_last_basis[ids]=0.
        if ids is None and getattr(self,'contact_first_plan_valid',None) is not None:
            self.contact_first_plan_valid.zero_()
            self.contact_first_joint_valid.zero_()
            self.contact_first_recovery_owned.zero_()
            self.contact_first_execution_log=[]
            self.contact_first_plan_start_time.fill_(float('inf'))
            self.contact_first_plan_contact_time.zero_()
            self.contact_first_plan_log=[]
            self.contact_first_last_actual_q.zero_()
            self.contact_first_last_time.fill_(float('nan'))
            self.contact_first_last_valid.zero_()
            self.contact_first_last_base_linear.zero_()
            self.contact_first_last_base_angular.zero_()
            self.contact_first_last_basis.zero_()

    def contact_first_inline_search_allowed(self):
        if self.contact_first_planner is None:
            return False
        return (not getattr(self.contact_first_planner.config, 'position_anchored_search', False)
                or getattr(self, 'contact_first_offline_diagnostic', False))

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

    def set_contact_first_evaluator(self, evaluator=None):
        """Attach an optional causal full-body candidate evaluator.

        The callback receives the complete candidate prefix from the planner;
        it must return ``True`` or ``{'accepted': bool, ...}``.  Keeping this
        explicit prevents a kinematic-only plan from being described as a
        whole-body physical result while allowing a native/Pinocchio runner to
        supply the already-audited model without changing this controller.
        """
        if evaluator is not None and not callable(evaluator):
            raise TypeError('Contact-first evaluator must be callable or None')
        if evaluator is not None and getattr(self, 'contact_first_evaluator_factory', None) is not None:
            raise ValueError('Set either a shared evaluator or an evaluator factory')
        self.contact_first_evaluator=evaluator
        return self

    def set_contact_first_evaluator_factory(self, factory=None):
        """Attach a per-lane evaluator factory for live-state admission.

        The factory is called as ``factory(lane, public_row, actual_q,
        actual_qd)`` immediately before that lane's plan is searched.  A
        native caller can therefore construct a fresh evaluator from the
        current articulation state, rather than sharing a stale evaluator
        across lanes.  The default path remains unchanged when unset.
        """
        if factory is not None and not callable(factory):
            raise TypeError('Contact-first evaluator factory must be callable or None')
        if factory is not None and self.contact_first_evaluator is not None:
            raise ValueError('Set either a shared evaluator or an evaluator factory')
        self.contact_first_evaluator_factory=factory
        return self

    @torch.inference_mode()
    def _install_contact_first_plan(self, ids, public, actual, actual_velocity,
                                    base_linear_acceleration, base_angular_acceleration,
                                    forward, left, up,
                                    target, normal_target, desired_velocity, event_time,
                                    limits, target_state_fns=None,
                                    base_state_fns=None, impact_evaluators=None):
        """Build one causal pre-contact arm plan per newly planned lane."""
        if self.contact_first_planner is None:
            return
        limits_np=limits.detach().cpu().numpy()
        q_limits=limits_np[12:]
        cfg=self.contact_first_planner.config
        if not self.contact_first_inline_search_allowed():
            for lane in ids.tolist():
                self.contact_first_plan_log.append(dict(lane=lane,accepted=False,
                    rejection='offline_reference_search_requires_explicit_runner',planning_seconds=0.))
            return
        point,normal,jp,jn=self.teacher.kinematics(actual,self.teacher.contact_offset)
        import time
        for local,lane in enumerate(ids.tolist()):
            if self.contact_first_recovery_owned[lane] or self.contact_first_plan_valid[lane]:
                self.contact_first_plan_log.append(dict(lane=lane,accepted=False,
                    rejection='verified_plan_or_recovery_owns_lane',planning_seconds=0.))
                continue
            planning_started=time.perf_counter()
            contact_time=float(event_time[local].item()-public[local,775])
            try:
                evaluator=self.contact_first_evaluator
                if self.contact_first_evaluator_factory is not None:
                    evaluator=self.contact_first_evaluator_factory(
                        int(lane), public[local].copy(),
                        actual[local].detach().cpu().numpy(),
                        actual_velocity[local].detach().cpu().numpy())
                    if evaluator is not None and not callable(evaluator):
                        raise TypeError('Evaluator factory must return callable or None')
                def kinematics_fn(q):
                    q_tensor=torch.as_tensor(q,device=actual.device,dtype=actual.dtype)[None]
                    pose=self.teacher.kinematics(q_tensor,self.teacher.contact_offset)
                    return tuple(value[0].detach().cpu().numpy() for value in pose)

                def kinematics_at_time_fn(q,duration):
                    event=target_state_fns[local](duration)
                    offset=self.teacher.contact_offset.clone()
                    if 'contact_point_local' in event:
                        offset[1:]=offset.new_tensor(event['contact_point_local'][1:])
                    pose=self.teacher.kinematics(offset.new_tensor(q)[None],offset)
                    return tuple(x[0].detach().cpu().numpy() for x in pose)
                result=self.contact_first_planner.plan(
                    q_rad=actual[local].detach().cpu().numpy(),
                    qd_rad_s=actual_velocity[local].detach().cpu().numpy(),
                    position_jacobian=jp[local].detach().cpu().numpy(),
                    normal_jacobian=jn[local].detach().cpu().numpy(),
                    racket_position_world_m=point[local].detach().cpu().numpy(),
                    racket_normal_world=normal[local].detach().cpu().numpy(),
                    target_position_world_m=target[local].detach().cpu().numpy(),
                    target_normal_world=normal_target[local].detach().cpu().numpy(),
                    target_velocity_world_m_s=desired_velocity[local].detach().cpu().numpy(),
                    contact_time_s=contact_time,
                    base_linear_velocity_world_m_s=public[local,:3],
                    base_linear_acceleration_world_m_s2=base_linear_acceleration[local].detach().cpu().numpy(),
                    base_angular_velocity_world_rad_s=public[local,3:6],
                    base_angular_acceleration_world_rad_s2=base_angular_acceleration[local].detach().cpu().numpy(),
                    racket_offset_world_m=point[local].detach().cpu().numpy(),
                    lower_rad=q_limits[:,0], upper_rad=q_limits[:,1],
                    velocity_limits_rad_s=np.full(6,cfg.arm_velocity_limit_rad_s),
                    evaluator=evaluator,
                    kinematics_fn=kinematics_fn,
                    kinematics_at_time_fn=(kinematics_at_time_fn if cfg.reachable_impact_targets else None),
                    target_state_fn=(None if target_state_fns is None
                                     else target_state_fns[local]),
                    impact_evaluator=(None if impact_evaluators is None else impact_evaluators[local]),
                    base_state_fn=(None if base_state_fns is None
                                   else base_state_fns[local]),
                )
            except (ValueError, RuntimeError, FloatingPointError) as exc:
                # Failed replacement never invalidates a verified prefix.
                self.contact_first_plan_log.append(dict(
                    lane=lane, accepted=False, planning_seconds=time.perf_counter()-planning_started, rejection=type(exc).__name__,
                    error=str(exc), contact_time_s=contact_time,
                    search_diagnostics=json.loads(json.dumps(self.contact_first_planner.last_diagnostics,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item()))))
                continue
            physical=result.metadata.get('evaluator')
            joint_accepted=False
            if cfg.require_joint_prefix:
                if not isinstance(physical,dict) or physical.get('leg_reference')!='causal_frozen_base':
                    raise RuntimeError('Joint execution requires a causal frozen-base leg replay')
                targets=np.asarray(physical['replayed_target_sequence_rad'],dtype=float)
                if (targets.shape!=(result.contact_index,18) or not np.isfinite(targets).all() or
                        np.any(targets < limits_np[:,0]-1e-7) or np.any(targets > limits_np[:,1]+1e-7)):
                    raise ValueError('Incomplete verified joint prefix')
                self.contact_first_joint_targets[lane,:result.contact_index]=torch.as_tensor(
                    targets,device=actual.device,dtype=actual.dtype)
                joint_accepted=True
            self.contact_first_joint_valid[lane]=joint_accepted
            self.contact_first_plan_q[lane]=torch.as_tensor(
                result.q_rad,device=actual.device,dtype=actual.dtype)
            self.contact_first_plan_start_time[lane]=float(public[local,775])
            self.contact_first_plan_contact_time[lane]=result.contact_time_s
            self.contact_first_plan_valid[lane]=True
            self.contact_first_plan_log.append(dict(
                lane=lane, accepted=True, planning_seconds=time.perf_counter()-planning_started, contact_index=result.contact_index,
                contact_time_s=result.contact_time_s, score=result.score,
                position_error_m=result.position_error_m,
                normal_error=result.normal_error,
                velocity_error_m_s=result.velocity_error_m_s,
                physical_evaluator=bool(evaluator is not None),
                search_diagnostics=dict(result.metadata.get(
                    'search_diagnostics', {}))))

    def _contact_first_plan_summaries(self, lanes):
        """Return JSON-safe latest plan outcomes for the causal event log."""
        latest = {}
        for row in reversed(self.contact_first_plan_log):
            lane = int(row['lane'])
            if lane in lanes and lane not in latest:
                latest[lane] = dict(row)
        return [latest[lane] for lane in lanes if lane in latest]

    def set_contact_first_recovery_owned(self, mask):
        # The outer Recovery arbiter remains the final owner after its original trigger.
        self.contact_first_recovery_owned.copy_(mask)

    @torch.inference_mode()
    def _overlay_contact_first_plan(self, action, public, limits):
        """Use the contact prefix, then hand it back continuously."""
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
            elapsed<=self.contact_first_plan_contact_time)
        normalized=2.*(plan-limits[12:,0])/(limits[12:,1]-limits[12:,0])-1.
        active &= ~self.contact_first_recovery_owned
        joint=self.contact_first_joint_valid & active & (elapsed<self.contact_first_plan_contact_time-1e-6)
        arm_only=active & ~self.contact_first_joint_valid
        action[:,12:]=torch.where(arm_only[:,None],normalized,action[:,12:])
        if joint.any():
            lanes=torch.where(joint)[0]
            targets=self.contact_first_joint_targets[lanes,index[lanes]]
            encoded=2.*(targets-limits[:,0])/(limits[:,1]-limits[:,0])-1.
            decoded=limits[:,0]+.5*(encoded.clamp(-1,1)+1)*(limits[:,1]-limits[:,0])
            error=(decoded-targets).abs().amax(1)
            if (error>2e-6).any():raise RuntimeError('Joint prefix final target round-trip exceeds 2e-6 rad')
            action[lanes]=encoded
            self.contact_first_execution_log.append(dict(lanes=lanes.cpu().tolist(),
                node=index[lanes].cpu().tolist(),clock_s=time[lanes].cpu().tolist(),
                target_rad=targets.cpu().tolist(),round_trip_error_rad=error.cpu().tolist(),
                owner='joint_prefix',physical_admission=True))
        if cfg.handoff_duration_s > 0.0:
            handoff=(self.contact_first_plan_valid & ~self.contact_first_joint_valid & ~self.contact_first_recovery_owned &
                     (elapsed>self.contact_first_plan_contact_time) &
                     (elapsed<=self.contact_first_plan_contact_time+cfg.handoff_duration_s))
            blend=((elapsed-self.contact_first_plan_contact_time)/cfg.handoff_duration_s).clamp(0.,1.)
            blended=torch.lerp(normalized,action[:,12:],blend[:,None])
            action[:,12:]=torch.where(handoff[:,None],blended,action[:,12:])
        return action

    @torch.inference_mode()
    def action(self,observation,default,limits):
        obs=observation['policy']
        if self.contact_first_planner is not None:
            self.contact_first_current_public=obs.detach().clone()
        actual_all=default[12:][None]+obs[:,21:27]
        actual_velocity=None
        base_linear_acceleration=None
        if self.contact_first_planner is not None:
            time_all=obs[:,775]
            actual_velocity=torch.zeros_like(actual_all)
            base_linear_acceleration=torch.zeros_like(obs[:,:3])
            base_angular_acceleration=torch.zeros_like(obs[:,3:6])
            previous_valid=self.contact_first_last_valid
            delta_time=time_all-self.contact_first_last_time
            usable=previous_valid & torch.isfinite(delta_time) & (delta_time>1e-6)
            actual_velocity=torch.where(
                usable[:,None],
                (actual_all-self.contact_first_last_actual_q)/delta_time.clamp_min(1e-6)[:,None],
                actual_velocity)
            current_basis=public_world_basis_in_body(obs)
            base_linear_acceleration=transported_body_acceleration(
                obs[:,:3],self.contact_first_last_base_linear,current_basis,
                self.contact_first_last_basis,delta_time,usable)
            base_angular_acceleration=transported_body_acceleration(
                obs[:,3:6],self.contact_first_last_base_angular,current_basis,
                self.contact_first_last_basis,delta_time,usable)
        retry_enabled=(self.contact_first_inline_search_allowed() and self.contact_first_planner.config.bounded_replanning)
        if retry_enabled:
            eligible=self.contact_replan_gate.eligible(obs,self.contact_first_plan_valid,self.contact_first_recovery_owned)
            ids=torch.where(eligible)[0]
            self.contact_replan_gate.record(ids,obs[:,775])
        else:
            ids=torch.where((~self.planned)&(obs[:,67]>.5)&(obs[:,775]>=2.0))[0]
        retry_ids=ids[self.planned[ids]]
        # Retry planning must not accumulate legacy prepose or change the
        # frozen base fallback. Restore those per-lane values after proposal.
        fallback_snapshot=(self.parameters[retry_ids].clone(),self.event_time[retry_ids].clone(),
                           self.target_offset_world[retry_ids].clone())
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
            if self.contact_first_inline_search_allowed():
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
                actual=actual_all[ids]
                # Re-evaluate the causal shuttle state for every candidate
                # contact time.  The planner receives body-frame targets, so
                # transform the propagated world state through the current
                # public basis; no future simulator state is read.
                initial_ball_position=plan['initial_position_m']
                initial_ball_velocity=plan['initial_velocity_mps']
                incoming_ball_velocity=plan['incoming_velocity_mps']
                command_world=np.array([1.10, 0.0, 1.80], dtype=np.float64)
                impact_evaluators=[]
                impact_model=None
                if self.contact_first_planner.config.evaluate_impact_outcome:
                    from .impact_outcome_evaluator_v1 import ImpactOutcomeEvaluator
                    from quadsmash.physics.full_racket_v4 import load_racket
                    from quadsmash.physics.court_v4 import load_court
                    impact_model=ImpactOutcomeEvaluator(self.physics,load_court(),load_racket())
                target_state_fns=[]
                if self.contact_first_planner.config.inverse_impact_targets and impact_model is None:
                    raise ValueError("Inverse impact targets require impact evaluation")
                base_state_fns=[]
                planner_cfg=self.contact_first_planner.config
                for local in range(len(ids)):
                    basis,root=planning_world_pose(public[local],command_world)
                    p0=np.asarray(initial_ball_position[local], dtype=float)
                    v0=np.asarray(initial_ball_velocity[local], dtype=float)
                    v_ref=np.asarray(incoming_ball_velocity[local], dtype=float)
                    desired_body_velocity=desired_velocity[local].detach().cpu().numpy()
                    normal_body=normal_target[local].detach().cpu().numpy()
                    inverse_cache={};base_context={}
                    lane_q=actual[local].detach().cpu().numpy().copy()
                    lane_limits=limits[12:].detach().cpu().numpy().copy()
                    def target_state_fn(duration_s, basis=basis, root=root, inverse_cache=inverse_cache, base_context=base_context, lane_q=lane_q, lane_limits=lane_limits,
                                        p0=p0, v0=v0, v_ref=v_ref, lane_id=int(ids[local]), clock_s=float(public[local,775]),
                                        desired_body_velocity=desired_body_velocity,
                                        normal_body=normal_body):
                        # DO NOT add the legacy plan.position_m - nominal
                        # intercept delta: ball propagation already places
                        # this candidate at the predicted interception point.
                        target=causal_shuttle_target_in_planning_frame(
                            self.physics, p0, v0, v_ref, basis, root,
                            desired_body_velocity, duration_s)
                        if self.contact_first_planner.config.inverse_impact_targets:
                            key=round(float(duration_s),9)
                            if key not in inverse_cache:
                                from .inverse_impact_target_v1 import inverse_impact_target
                                ball,incoming=propagate_ball_state(p0,v0,duration_s,self.physics)
                                local=self.teacher.contact_offset.detach().cpu().numpy().copy()
                                local[0]=self.teacher.face_sign*impact_model.racket['assumptions']['string_thickness_m']/2
                                if self.contact_first_planner.config.reachable_impact_targets:
                                    from .inverse_impact_target_v1 import optimize_reachable_impact
                                    from .contact_pose_reachability_v1 import diagnose_contact_pose
                                    base=base_context['fn'](duration_s)
                                    R=basis@base['rotation_current_from_future']
                                    origin=root+basis@base['translation_current_m']
                                    def kin(q,contact):
                                        offset=self.teacher.contact_offset.clone()
                                        offset[1:]=offset.new_tensor(contact[1:])
                                        pose=self.teacher.kinematics(offset.new_tensor(q)[None],offset)
                                        p,n,jp,jn=[x[0].detach().cpu().numpy().astype(float) for x in pose]
                                        return origin+R@p,R@n,R@jp,R@jn
                                    def base_velocity(point):
                                        return basis@base['linear_velocity_current_m_s']+np.cross(
                                            basis@base['angular_velocity_current_rad_s'],point-origin)
                                    result=optimize_reachable_impact(impact_model,ball,incoming,local,kin,lane_q,
                                        lane_limits[:,0],lane_limits[:,1],np.full(6,self.contact_first_planner.config.arm_velocity_limit_rad_s),base_velocity)
                                    if result.get('normal_world') is not None:
                                        selected=np.asarray(result.get('contact_point_local',local))
                                        result['reachability']=diagnose_contact_pose(lambda q:kin(q,selected),lane_q,
                                            lane_limits[:,0],lane_limits[:,1],np.full(6,self.contact_first_planner.config.arm_velocity_limit_rad_s),
                                            ball,np.asarray(result['normal_world']),np.asarray(result['racket_velocity_world_m_s']),base_velocity)
                                    inverse_cache[key]=result
                                else:
                                    inverse_cache[key]=inverse_impact_target(impact_model,ball,incoming,local)
                                if not hasattr(self,'inverse_impact_audit'):self.inverse_impact_audit=[]
                                self.inverse_impact_audit.append(dict(lane=lane_id,clock_s=clock_s,
                                    duration_s=duration_s,result=inverse_cache[key]))
                            inverse=inverse_cache[key]
                            if not inverse['accepted'] and not self.contact_first_planner.config.position_anchored_search:
                                from .inverse_impact_target_v1 import UnreachableImpactTarget
                                raise UnreachableImpactTarget('predicted_outgoing_unreachable')
                            target['velocity_world_m_s']=basis.T@np.asarray(inverse['racket_velocity_world_m_s'])
                            extras={k:inverse[k] for k in ('contact_point_local','optimized_q_rad','reachability') if k in inverse}
                            original=inverse.get('original_target',inverse)
                            extras['original_requested_racket_velocity_world_m_s']=original['racket_velocity_world_m_s']
                            extras['original_requested_velocity_current_frame_m_s']=(basis.T@np.asarray(original['racket_velocity_world_m_s'])).tolist()
                            extras['optimized_requested_racket_velocity_world_m_s']=inverse['racket_velocity_world_m_s']
                            extras['inverse_accepted_semantics']='analytic_ball_side_only'
                            extras['inverse_robot_kinematic_valid']=inverse.get('robot_contact_kinematic_valid',False)
                            return dict(target,normal_world=basis.T@np.asarray(inverse['normal_world']),**extras)
                        return dict(target, normal_world=normal_body)
                    target_state_fns.append(target_state_fn)
                    impact_fn=None
                    if impact_model is not None:
                        local_contact=self.teacher.contact_offset.detach().cpu().numpy().copy()
                        local_contact[0]=self.teacher.face_sign*impact_model.racket['assumptions']['string_thickness_m']/2
                        def impact_fn(metadata,basis=basis,root=root,p0=p0,v0=v0,local_contact=local_contact):
                            ball, incoming=propagate_ball_state(p0,v0,metadata['contact_time_s'],self.physics)
                            normal=basis@metadata['achieved_normal_current_frame']
                            # Kinematics contact_offset is the ball-center proxy,
                            # radius outward from the physical string surface.
                            proxy=root+basis@metadata['achieved_position_current_frame_m']
                            surface=proxy-self.physics['collision_radius_m']*normal
                            velocity=basis@metadata['achieved_velocity_current_frame_m_s']
                            selected_local=np.asarray(metadata.get('contact_point_local',local_contact))
                            return impact_model.evaluate(ball,incoming,surface,velocity,normal,selected_local,
                                robot_state=metadata.get("impact_robot_state"))
                    impact_evaluators.append(impact_fn)
                    base_linear=np.asarray(public[local,:3], dtype=float)
                    base_angular=np.asarray(public[local,3:6], dtype=float)
                    raw_linear_acceleration=np.asarray(
                        base_linear_acceleration[ids[local]].detach().cpu().numpy(), dtype=float)
                    raw_angular_acceleration=np.asarray(
                        base_angular_acceleration[ids[local]].detach().cpu().numpy(), dtype=float)
                    clipped_linear_acceleration=np.clip(
                        raw_linear_acceleration,
                        -planner_cfg.base_linear_acceleration_limit_m_s2,
                        planner_cfg.base_linear_acceleration_limit_m_s2)
                    clipped_angular_acceleration=np.clip(
                        raw_angular_acceleration,
                        -planner_cfg.base_angular_acceleration_limit_rad_s2,
                        planner_cfg.base_angular_acceleration_limit_rad_s2)
                    def base_state_fn(duration_s, base_linear=base_linear,
                                      base_angular=base_angular,
                                      linear_acceleration=clipped_linear_acceleration,
                                      angular_acceleration=clipped_angular_acceleration):
                        from .base_contact_prediction_v1 import extrapolate_base
                        return extrapolate_base(duration_s,base_linear,base_angular,linear_acceleration,angular_acceleration)
                    base_context['fn']=base_state_fn
                    base_state_fns.append(base_state_fn)
                self._install_contact_first_plan(
                    ids,public,actual,actual_velocity[ids],base_linear_acceleration[ids],
                    base_angular_acceleration[ids],
                    forward,left,up,target_body,
                    normal_target,desired_velocity,self.event_time[ids],limits,
                    target_state_fns,base_state_fns,impact_evaluators)
            self.logs.append(dict(input='Current public794D observation, fixed known command and learned no-ball calibration only',
                env_ids=ids.cpu().tolist(),observation_clock_s=public[:,775].tolist(),
                used_public_observations=public.tolist(),calibrated_event_time_s=calibrated.tolist(),
                arm_joint_correction_rad=correction.tolist(),causal_prepose_gain=self.prepose_gain,
                relative_contact_prepose=self.relative_prepose,
                height_intercept=self.height_intercept,
                causal_prepose_joint_delta_rad=applied_prepose.tolist(),
                contact_first_plan_summaries=self._contact_first_plan_summaries(
                    set(ids.cpu().tolist())),
                **{k:v.tolist() for k,v in plan.items()}))
        if len(retry_ids):
            self.parameters[retry_ids],self.event_time[retry_ids],self.target_offset_world[retry_ids]=fallback_snapshot
        action=self.teacher.action(observation,self.parameters,default,limits,event_time=self.event_time,
                                   target_offset_world=self.target_offset_world)
        action=self._overlay_contact_first_plan(action,obs,limits)
        if self.contact_first_planner is not None:
            self.contact_first_last_actual_q.copy_(actual_all)
            self.contact_first_last_time.copy_(obs[:,775])
            self.contact_first_last_valid.fill_(True)
            self.contact_first_last_base_linear.copy_(obs[:,:3])
            self.contact_first_last_base_angular.copy_(obs[:,3:6])
            self.contact_first_last_basis.copy_(current_basis)
        return action
