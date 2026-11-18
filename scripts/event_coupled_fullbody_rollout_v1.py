"""Production joint 18D rollout over the existing full-body dynamics.

The parameter vector is [leg_position12, arm_position6, arm_rate6]. Every
candidate starts from the supplied actual state/history and is independently
rolled by CoupledContactRollout. No future state injection is possible.
"""
import copy
import hashlib
import numpy as np
from scripts.coupled_contact_prefix_v1 import CoupledContactRollout

class EventCoupledFullBodyRolloutV1:
    parameter_size=24
    def __init__(self, model, interface, predictor, state, commands, arm_reference, metadata, *, impact_builder=None, rollout_class=CoupledContactRollout):
        self.rollout=rollout_class(model,interface,predictor,state,commands,arm_reference,metadata)
        self.parameter_size=int(self.rollout.radii.size)
        self.impact_builder=impact_builder
    def retimed(self,event):
        offset=float(dict(event).get('hit_time_offset_s',0.0)); n0=len(self.rollout.commands)
        n=max(2,min(64,int(round((n0*.01+offset)/.01))))
        old=np.linspace(0.,1.,n0); new=np.linspace(0.,1.,n)
        commands=np.vstack([np.interp(new,old,self.rollout.commands[:,j]) for j in range(18)]).T
        old_ref=np.linspace(0.,1.,n0+1); new_ref=np.linspace(0.,1.,n+1)
        reference=np.vstack([np.interp(new_ref,old_ref,self.rollout.arm_reference[:n0+1,j]) for j in range(6)]).T
        metadata=copy.deepcopy(self.rollout.metadata); metadata['contact_index']=n; metadata['contact_time_s']=float(metadata.get('contact_time_s',n0*.01)+offset)
        return EventCoupledFullBodyRolloutV1(self.rollout.model,self.rollout.interface,self.rollout.predictor,
            self.rollout.initial,commands,reference,metadata,impact_builder=self.impact_builder,rollout_class=type(self.rollout))
    def __call__(self, theta):
        theta=np.asarray(theta,float)
        if theta.shape!=(self.parameter_size,) or not np.isfinite(theta).all(): raise ValueError('24 finite joint parameters required')
        result=self.rollout(theta)
        if not result.get('path_valid') or not result.get('trajectory'):
            return dict(path_valid=False,failure=result.get('failure'),finite_mass_status='EFFECTIVE_MASS_UNVERIFIED')
        e=result['terminal_errors']
        out=dict(path_valid=True,contact_position_error_m=float(e['position_m']),normal_error=float(e['normal']),velocity_error_m_s=float(e['velocity_m_s']),trajectory=result['trajectory'],terminal_errors=e,native_reference_valid=False,teacher_eligible=False,contact_feasible=False,legal_return_predicted=False,attack5_predicted=False,target_landing_predicted=False)
        if self.impact_builder is None:
            out['finite_mass_status']='EFFECTIVE_MASS_UNVERIFIED'
        else:
            # The builder must consume the terminal state from this exact
            # rollout.  The ball and string data are taken from the same
            # immutable metadata snapshot; callers cannot inject a cached
            # impact result or a status string.
            metadata=copy.deepcopy(getattr(self.rollout,'metadata',{}))
            terminal=result['trajectory'][-1]
            if all('final_target_rad' in row for row in result['trajectory']):
                metadata['candidate_command_sequence_hash']=hashlib.sha256(
                    np.asarray([row['final_target_rad'] for row in result['trajectory']],float).tobytes()).hexdigest()
            try:
                contact_point=np.asarray(metadata.get('contact_point_local',np.zeros(3)),float)
                ball_position=np.asarray(metadata.get('ball_position_world',metadata.get('target_position_world_m',np.zeros(3))),float)
                ball_velocity=np.asarray(metadata.get('ball_velocity_world',metadata.get('incoming_velocity_world_m_s',metadata.get('incoming_velocity_planning',np.zeros(3)))),float)
                evidence=self.impact_builder(
                    terminal,
                    ball_position=ball_position,
                    ball_velocity=ball_velocity,
                    contact_point_local=contact_point,
                    contact_time_s=float(metadata.get('contact_time_s',metadata.get('event_time_s',0.0))),
                    support_mode=('support' if bool(np.asarray(terminal.get('contacts',[]),dtype=bool).any()) else 'flight'),
                    candidate_theta=theta.copy(),
                    metadata=metadata,
                )
            except TypeError:
                # Keep a strict compatibility path for the original builder
                # signature, while still passing only same-candidate state.
                evidence=self.impact_builder(terminal,
                    ball_position=ball_position,ball_velocity=ball_velocity,
                    contact_point_local=contact_point,
                    contact_time_s=float(metadata.get('contact_time_s',metadata.get('event_time_s',0.0))),
                    support_mode=('support' if bool(np.asarray(terminal.get('contacts',[]),dtype=bool).any()) else 'flight'))
            if not isinstance(evidence,dict): raise ValueError('Impact builder must return evidence')
            protected={'path_valid','contact_position_error_m','normal_error','velocity_error_m_s',
                       'terminal_errors','trajectory','native_reference_valid','teacher_eligible'}
            if protected.intersection(evidence):
                raise ValueError('Impact evidence cannot overwrite rollout validity or terminal state')
            allowed={'finite_mass_status','finite_mass_verified','contact_feasible',
                     'legal_return_predicted','attack5_predicted','target_landing_predicted',
                     'final_impact','impact_diagnostics','same_candidate_state',
                     'initial_state_hash','command_sequence_hash','terminal_state_hash',
                     'contact_time_s','support_mode','actual_contact_position_world_m',
                     'actual_contact_normal_world','actual_contact_velocity_world_m_s',
                     'mass_model_computed','inverse_effective_mass_kg_inv',
                     'rejection_reason','task_margins','contact_geometry_valid',
                     'candidate_theta_sha256','metadata_contact_index'}
            unknown=set(evidence)-allowed
            if unknown: raise ValueError('Impact evidence contains unapproved fields: '+','.join(sorted(unknown)))
            out.update(evidence)
            out.setdefault('finite_mass_status','EFFECTIVE_MASS_UNVERIFIED')
        out['candidate_theta_sha256']=hashlib.sha256(theta.tobytes()).hexdigest()
        return out


class EventCoupledFullBodyEventRolloutV1:
    """Event layer over the real 24-D rollout.

    Layout is ``[takeoff_offset, push_duration, hit_offset, com_ax,
    com_az, pitch_rate, string_u, string_v, local24]``.  The event variables
    alter the command sequence before the nonlinear rollout is called.  A
    caller may provide ``wbc_event_fn``; it receives the event dictionary and
    must return a 12-D leg correction for each node.  Without that callback a
    conservative causal projection is used, and the result is explicitly
    marked ``wbc_event_source='fallback_reference_projection'``.

    No future state, contact force, or base pose is injected.  The only state
    advanced is the state produced by the existing full-body predictor.
    """
    event_size=8
    parameter_size=32
    def __init__(self, base_rollout, *, wbc_event_fn=None, event_metadata_fn=None, event_bounds=None):
        self.base=base_rollout
        self.local_size=int(getattr(getattr(base_rollout,'rollout',None),'radii',np.zeros(24)).size)
        self.parameter_size=self.event_size+self.local_size
        self.wbc_event_fn=wbc_event_fn
        self.event_metadata_fn=event_metadata_fn
        self.event_bounds=np.asarray(event_bounds if event_bounds is not None else [
            [-.06,.06],[-.05,.05],[-.06,.06],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-.12,.12],[-.12,.12]],float)
        if self.event_bounds.shape!=(8,2) or not np.isfinite(self.event_bounds).all() or np.any(self.event_bounds[:,0]>self.event_bounds[:,1]):
            raise ValueError('Finite event bounds required')
    def _parse(self,theta):
        theta=np.asarray(theta,float)
        if theta.shape!=(self.parameter_size,) or not np.isfinite(theta).all(): raise ValueError('32 finite event/fullbody parameters required')
        event=theta[:8].copy();local=theta[8:].copy()
        if np.any(event<self.event_bounds[:,0]-1e-12) or np.any(event>self.event_bounds[:,1]+1e-12): raise ValueError('Event parameter outside bounds')
        return event,local
    def __call__(self,theta):
        event,local=self._parse(theta)
        takeoff,push,hit,com_x,com_z,pitch,u,v=event
        active_base=self.base.retimed({'hit_time_offset_s':float(hit)}) if hasattr(self.base,'retimed') else self.base
        if self.event_metadata_fn is not None and hasattr(active_base,'rollout'):
            active_base.rollout.metadata.update(self.event_metadata_fn(event,copy.deepcopy(active_base.rollout.metadata)))
        # Event variables are converted to causal command corrections.  This
        # changes the actual 18-D command sequence, rather than metadata only.
        n=len(active_base.rollout.commands)
        s=np.linspace(0.,1.,n)
        push_phase=np.exp(-((s-(.38+takeoff/.12))/.28)**2)
        hit_phase=np.exp(-((s-(.86+hit/.12))/.18)**2)
        event_local=local.copy()
        event_local[:12]+=push_phase[-1]*np.r_[np.full(4,com_x*.02),np.full(4,com_z*.02),np.full(4,pitch*.01)]
        if len(event_local)==24:
            event_local[12:18]+=np.array([u,v,0.,-v,u,0.])*hit_phase[-1]
            event_local[18:24]+=np.array([u,v,-u,-v,u,-v])*hit_phase[-1]*.5
        elif len(event_local)==36:
            event_local[24:30]+=np.array([u,v,0.,-v,u,0.])*hit_phase[-1]
            event_local[30:36]+=np.array([u,v,-u,-v,u,-v])*hit_phase[-1]*.5
        else:
            raise ValueError('Unsupported local rollout parameterization')
        radii=np.asarray(getattr(active_base.rollout,'radii',np.full(len(event_local),np.inf)),float)
        if np.any(np.abs(event_local)>radii+1e-10):
            return dict(path_valid=False,failure='EVENT_LOCAL_PARAMETER_LIMIT',event_parameters=event.tolist(),
                        actual_event_local_parameters=event_local.tolist(),wbc_event_source='fallback_reference_projection')
        if self.wbc_event_fn is not None:
            leg=np.asarray(self.wbc_event_fn(dict(takeoff_time_offset_s=float(takeoff),push_duration_offset_s=float(push),
                hit_time_offset_s=float(hit),com_acceleration_delta_world_m_s2=np.array([com_x,0.,com_z]),
                body_pitch_rate_delta_rad_s=float(pitch),string_contact_uv=np.array([u,v])),
                active_base),float)
            if leg.shape==(12,):
                event_local[:12]+=leg
            elif leg.shape==(n,12):
                active_base.rollout.external_leg_corrections=leg.copy()
            else:
                raise ValueError('WBC event callback must return 12-D or per-node Nx12 correction')
            wbc_source='event_wbc_callback'
        else:
            wbc_source='fallback_reference_projection'
        try:
            result=active_base(event_local)
        except ValueError as exc:
            raise ValueError(f'Event local rollout shape {event_local.shape} expected {active_base.parameter_size}: {exc}') from exc
        if not isinstance(result,dict): raise ValueError('Full-body rollout must return a dictionary')
        result=dict(result)
        result.update(event_parameters=event.tolist(),event_time_offset_s=float(hit),takeoff_time_offset_s=float(takeoff),
                     push_duration_offset_s=float(push),string_contact_uv=[float(u),float(v)],
                     wbc_event_source=wbc_source,actual_event_local_parameters=event_local.tolist())
        return result
