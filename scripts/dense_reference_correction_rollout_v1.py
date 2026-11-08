"""Fine physical rollout with a dense nominal command and smooth full18 corrections.

Zero correction reproduces each saved nominal command. All correction knots,
including the first, are free; the original physical interface and path checks
remain unchanged. This function predicts a model, never native execution.
"""
import time
import numpy as np
from scipy.interpolate import PchipInterpolator
from scripts.search_current_native_recovery_v4 import WORK, initialize, knots_for, stable_window_violations
from scripts.audit_actual_handoff_model_v1 import state_from_record
from scripts.audit_settled_stance_model_v2 import ForceRecorder
from scripts.event_recovery_manager_v1 import EventRecoveryManagerV1, RecoveryPhaseMemory
from scripts.predict_self_contact_spheres_v2 import predict_self_contact_spheres


def rollout(key,nodes,substeps,deadline):
    c=WORK['c'];record=WORK['records'][key];state=state_from_record(c.model,record)
    knots=knots_for(record);steps=800-record['tick']
    phase=record['phase'];manager=EventRecoveryManagerV1()
    manager.memory=RecoveryPhaseMemory(episode=0,last_time_s=record['time_s'],phase=phase['phase'],
        release_time_s=phase['release_time_s'],first_ground_time_s=phase['first_ground_time_s'],
        settled_duration_s=phase['phase_settled_duration_s'],
        final_stable_duration_s=phase['control_sample_stable_duration_s'],ground_contacts=state.contacts.copy())
    limits=c.interface.limits;span=np.diff(limits).ravel()
    targets=limits[:,0]+.5*(np.clip(nodes,-1,1)+1)*span
    baseline=np.asarray(record['warmstart_targets'],float)
    if baseline.shape!=(steps+1,18) or not np.isfinite(baseline).all():
        raise ValueError('Full bounded warm-start command grid required')
    knot_ticks=np.rint(knots/.01).astype(int)
    if np.max(abs(knot_ticks*.01-knots))>1e-10:
        raise ValueError('Control knots must align with the command grid')
    curve=PchipInterpolator(knots,targets-baseline[knot_ticks],axis=0)
    previous=state.previous_command_target.copy();rows=[];cost=0.;failure=None;stable=0.;longest=0.;window_violation=0.
    for k in range(steps):
        if time.monotonic()>deadline:failure=dict(step=k,reason='compute_budget');break
        target=np.clip(baseline[k]+curve(k*.01),limits[:,0],limits[:,1])
        recorder=ForceRecorder(c.model)
        try:
            pred=predict_self_contact_spheres(c.model,c.interface,state,target,c.geometry.spheres,
                substeps=substeps,geometry=c.path_geometry,collision_response=recorder)
        except (ValueError,np.linalg.LinAlgError) as error:
            failure=dict(step=k,reason='model_rejected',error=repr(error));break
        bad={n:v for n,v in pred['diagnostics'].items() if
            (n=='nonfoot_ground_violation_m' and v>1e-6) or
            (n in ['joint_position_violation_rad','joint_velocity_violation_rad_s','motor_torque_violation_nm'] and v>3e-4)}
        if bad:
            failure=dict(step=k,reason='physical_path_violation',violations=bad,collision=pred['first_ground_violation']);break
        state=pred['state'];state.previous_command_target=target.copy();force=recorder.last_force
        r=state.root_rotation_world;linear=r@state.v[:3]
        angles=np.array([np.arctan2(r[2,1],r[2,2]),np.arcsin(np.clip(-r[2,0],-1,1))])
        phase=manager.update(episode=0,time_s=float(np.float32(record['tick']+k+1)*np.float32(.01)),
            had_racket_contact=True,released=True,racket_touching=False,separated_now=True,
            ground_normal_force_n=force[:,2],roll_pitch_rad=angles,root_velocity_world=linear,
            root_angular_velocity_world=r@state.v[3:6],joint_velocity=state.joint_velocity,nonfoot_contact_now=False)
        state.contacts=phase['ground_contacts'].copy();state.measured_force_world=force.copy()
        # Only stability accumulated during this newly planned segment counts
        # toward its terminal window; no earlier native window is credited.
        stable=min((k+1)*.01,phase['control_sample_stable_duration_s']);longest=max(longest,stable)
        omega=float(np.linalg.norm(state.v[3:6]));joint_speed=float(np.max(abs(state.joint_velocity)))
        cost+=.01*(20.*np.sum(angles**2)+10.*np.sum(linear**2)+5.*omega**2+.05*joint_speed**2)
        cost+=.001*np.sum(((target-previous)/span)**2);previous=target.copy()
        # Search guidance only; the existing manager remains the unchanged
        # pass/fail authority. Penalize violations throughout the last 0.8 s.
        violations=stable_window_violations(angles,linear,omega,joint_speed,state.contacts)
        if k>=steps-80:window_violation+=float(violations@violations)/80.
        rows.append(dict(step=k,q=state.q.copy(),v=state.v.copy(),target=target.copy(),
            contacts=state.contacts.copy(),ground_force=force.copy(),phase=phase,roll_pitch=angles,
            root_velocity_world=linear,omega=omega,joint_speed=joint_speed,stable_s=stable))
    full=failure is None and len(rows)==steps
    if rows:
        last=rows[-1];cost+=30.*np.sum(last['roll_pitch']**2)+30.*np.sum(last['root_velocity_world']**2)+10.*last['omega']**2+.2*last['joint_speed']**2
    return dict(nodes=nodes,steps=len(rows),required_steps=steps,model_path_complete=full,failure=failure,
        terminal_stable_s=stable,longest_stable_s=longest,cost=float(cost),window_violation=window_violation,
        rank=[0 if full else 1,steps-len(rows),0 if stable>=.8 else 1,window_violation,-stable,float(cost)],
        terminal=rows[-1] if rows else None,trajectory=rows if substeps==160 else None,
        substeps=substeps,native_success=False,valid_expert_labels=False)

