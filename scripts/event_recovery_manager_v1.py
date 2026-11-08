"""Causal lifecycle and geometry for pure-control post-release recovery.

This module selects recovery phases, not actions. It has no TRAIN IDs, fixed
takeover second, student residual, early teacher exit, or learned component.
Native wiring and controller execution are separately required.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class RecoveryPhaseMemory:
    episode: int = -1
    last_time_s: float = -1.
    phase: str = 'PREFIX'
    release_time_s: float | None = None
    first_ground_time_s: float | None = None
    settled_duration_s: float = 0.
    final_stable_duration_s: float = 0.
    ground_contacts: object = None


class EventRecoveryManagerV1:
    def __init__(self):
        self.memory=RecoveryPhaseMemory()

    def update(self,*,episode,time_s,had_racket_contact,released,racket_touching,
               separated_now,ground_normal_force_n,roll_pitch_rad,
               root_velocity_world,root_angular_velocity_world,joint_velocity,
               nonfoot_contact_now=False):
        force=np.asarray(ground_normal_force_n,float)
        angles=np.asarray(roll_pitch_rad,float)
        velocity=np.asarray(root_velocity_world,float)
        omega=np.asarray(root_angular_velocity_world,float)
        jv=np.asarray(joint_velocity,float)
        if (type(episode) is not int or episode<0 or not np.isfinite(time_s) or time_s<0
                or force.shape!=(4,) or angles.shape!=(2,) or velocity.shape!=(3,)
                or omega.shape!=(3,) or jv.shape!=(18,)
                or not all(np.isfinite(x).all() for x in [force,angles,velocity,omega,jv])):
            raise ValueError('Finite current measured state and ground-only forces required')
        m=self.memory
        if episode!=m.episode:
            if m.episode>=0 and episode<m.episode:raise ValueError('Episode moved backwards')
            self.memory=m=RecoveryPhaseMemory(episode=episode,ground_contacts=np.zeros(4,bool))
        if time_s<=m.last_time_s:raise ValueError('Exactly one update per increasing control timestamp required')
        dt=0. if m.last_time_s<0 else float(time_s-m.last_time_s)
        if dt>.020001:raise ValueError('Skipped control samples cannot count as continuous stability')
        # A light contacting foot is retained; 5 N is only a load diagnostic.
        m.last_time_s=float(time_s)
        m.ground_contacts=np.where(m.ground_contacts,force>.5,force>1.)
        support=bool(m.ground_contacts.any())
        previous_phase=m.phase
        if m.phase=='PREFIX' and had_racket_contact and released and separated_now and not racket_touching:
            m.release_time_s=float(time_s)
            m.phase='BUFFER_LOAD' if support else 'AIR_PREPARE'
        if m.phase!='PREFIX':
            if support and m.first_ground_time_s is None:m.first_ground_time_s=float(time_s)
            if not support:
                m.phase='AIR_PREPARE';m.settled_duration_s=0.;m.final_stable_duration_s=0.
            else:
                # This is a soft phase-selection condition, not a success gate
                # or a hard constraint on permissible transient attitude.
                settling=(np.max(abs(angles))<np.deg2rad(30.) and abs(velocity[2])<.5
                    and np.linalg.norm(omega)<2. and not nonfoot_contact_now)
                m.settled_duration_s=m.settled_duration_s+dt if settling else 0.
                m.phase='SUSTAIN' if m.settled_duration_s>=.15 else 'BUFFER_LOAD'
                stable=(np.max(abs(angles))<=np.deg2rad(15.) and np.linalg.norm(velocity[:2])<=.3
                    and abs(velocity[2])<=.1 and np.linalg.norm(omega)<=1.
                    and np.max(abs(jv))<=1. and not nonfoot_contact_now)
                m.final_stable_duration_s=m.final_stable_duration_s+dt if stable else 0.
        return dict(phase=m.phase,changed=previous_phase!=m.phase,
            owns_final_recovery_command=m.phase!='PREFIX',release_time_s=m.release_time_s,
            first_ground_time_s=m.first_ground_time_s,
            ground_contacts=m.ground_contacts.copy(),currently_loaded=force>=5.,
            phase_settled_duration_s=m.settled_duration_s,
            control_sample_stable_duration_s=m.final_stable_duration_s,
            formal_recovery_success_not_inferred=True)


def authored_clearance_state(geometry,state):
    """Lowest authored surface and its instantaneous rate, including the head.

    Exact for the current authored support witness. At changing/tied witnesses
    the derivative is piecewise; native filtered contacts remain authoritative.
    """
    gaps,jacobian=geometry.evaluate(state.q,jacobian=True)
    rates=jacobian@state.v
    if not np.isfinite(gaps).all() or not np.isfinite(rates).all():raise ValueError('Nonfinite surface geometry')
    bodies={}
    for i,shape in enumerate(geometry.shapes):
        name=shape['body']
        if name not in bodies or gaps[i]<bodies[name]['gap_m']:
            bodies[name]=dict(gap_m=float(gaps[i]),rate_m_s=float(rates[i]),shape_index=i)
    if not {'Head_lower','Head_upper'}.issubset(bodies):raise ValueError('Original head geometry required')
    return bodies


class RecoveryCommandArbiterV1:
    """Select one total command; never blend an old residual after takeover."""
    def __init__(self,limits,arm_rate_rad_s=10.):
        self.limits=np.asarray(limits,float)
        if self.limits.shape!=(18,2) or not np.isfinite(self.limits).all() or np.any(self.limits[:,0]>=self.limits[:,1]):
            raise ValueError('Original joint limits required')
        self.arm_rate=float(arm_rate_rad_s)
        if self.arm_rate!=10.:raise ValueError('Keep the current declared arm command slew')

    def select(self,student_target,controller_target,previous_final_target,*,owns_recovery,dt_s=.01):
        previous=np.asarray(previous_final_target,float)
        requested=np.asarray(controller_target if owns_recovery else student_target,float)
        if requested.shape!=(18,) or previous.shape!=(18,) or not np.isfinite(requested).all() or not np.isfinite(previous).all() or dt_s!=.01:
            raise ValueError('Finite total request and actual previous final command required')
        target=np.clip(requested,self.limits[:,0],self.limits[:,1])
        target[12:]=np.clip(target[12:],previous[12:]-dt_s*self.arm_rate,previous[12:]+dt_s*self.arm_rate)
        return target,dict(owner='controller' if owns_recovery else 'frozen_prefix',
            requested_target_rad=requested.copy(),final_target_rad=target.copy(),
            command_mapping_delta_rad=target-requested,
            student_residual_added_after_controller=False,
            measured_joint_position_used_to_bound_target=False)
