"""Full-command recovery shooting with explicit hip/mount collision response.

Same task cost and original hardware; no extra arm target slew. Physical
predictions remain approximate and are not native recovery demonstrations.
"""
import time
import numpy as np
from scipy.interpolate import PchipInterpolator
from scripts.search_landing_support_tail_v3 import _WORKER
from scripts.check_constrained_recovery_suffix_v1 import state_from_row
from scripts.predict_self_contact_spheres_v2 import predict_self_contact_spheres as predict_position_limited_spheres
from scripts.hip_mount_collision_response_v1 import HipMountCollisionResponseV1
from scripts.foot_separation_path_constraint_v1 import FootSeparationPathConstraintV1
from scripts.event_recovery_manager_v1 import authored_clearance_state
from scripts.feedback_tail_rollout_v1 import terminal_measures


def _rollout(nodes,query,audit,substeps,deadline):
    w=_WORKER;c=w['controller'];interface=c.qp.interface;state=state_from_row(query)
    nodes=np.asarray(nodes,float)
    if nodes.shape!=(5,18) or not np.isfinite(nodes).all():raise ValueError('Five full-command knots required')
    limits=interface.limits
    targets=limits[:,0]+.5*(np.clip(nodes,-1.,1.)+1.)*np.diff(limits).ravel()
    curve=PchipInterpolator([0.,.08,.18,.30,.45,.60],np.vstack([state.previous_command_target,targets]),axis=0)
    geometry=FootSeparationPathConstraintV1(w['geometry'],w['spheres'],margin_m=0.)
    collision_response=HipMountCollisionResponseV1(c.qp.model)
    cost=0.;rows=[];failure=None
    for k in range(60):
        if time.monotonic()>=deadline:failure=dict(step=k,reason='compute_budget');break
        previous=state.previous_command_target.copy()
        target=np.clip(curve(k*.01),limits[:,0],limits[:,1])
        try:
            pred=predict_position_limited_spheres(c.qp.model,interface,state,target,w['spheres'],geometry=geometry,substeps=substeps,collision_response=collision_response)
        except (ValueError,np.linalg.LinAlgError) as e:
            failure=dict(step=k,reason='prediction_rejected',error=repr(e),collision=getattr(geometry,'last_violation',None));break
        bad={name:value for name,value in pred['diagnostics'].items() if
            (name=='nonfoot_ground_violation_m' and value>1e-6) or
            (name in ['joint_position_violation_rad','joint_velocity_violation_rad_s','motor_torque_violation_nm'] and value>3e-4)}
        if bad:failure=dict(step=k,reason='physical_path_violation',violations=bad,first_ground_violation=pred['first_ground_violation']);break
        state=pred['state'];state.previous_command_target=target.copy()
        measures=terminal_measures(c,state)
        clearances=authored_clearance_state(w['geometry'],state)
        # Transient tilt/velocity are costs, not final-standing constraints.
        supported=bool(state.contacts.any())
        cost+=.01*((8. if supported else 2.)*(measures['roll']**2+measures['pitch']**2)
            +2.*measures['omega']**2+2.*measures['horizontal']**2
            +.02*np.sum(((target-previous)/np.diff(limits).ravel())**2))
        for name,item in clearances.items():
            gap=item['gap_m'];rate=item['rate_m_s'];head=name in ['Head_lower','Head_upper']
            approaching=max(0.,-rate-max(0.,gap-.005)/.15)
            cost+=.01*(40. if head else 10.)*(np.exp(-max(gap,0.)/.02)+approaching**2)
        rows.append(dict(step=k,q=state.q.copy(),v=state.v.copy(),contacts=state.contacts.copy(),
            target=target.copy(),head_lower=clearances['Head_lower'],head_upper=clearances['Head_upper'],measures=measures))
    accepted=failure is None and len(rows)==60
    terminal=rows[-1]['measures'] if rows else None
    if terminal:cost+=5.*(terminal['roll']**2+terminal['pitch']**2)+terminal['omega']**2+terminal['horizontal']**2
    return dict(nodes=nodes,horizon_model_feasible=accepted,steps=len(rows),required_steps=60,
        rank=[0 if accepted else 1,60-len(rows),float(cost)],cost=float(cost),failure=failure,
        terminal=terminal,trajectory=rows if substeps==160 else None,substeps=substeps,
        full_episode_certified=False,native_execution=False,valid_expert_labels=False)



def rollout(nodes,query,audit,substeps,deadline):
    result=_rollout(nodes,query,audit,substeps,deadline)
    terminal=result['terminal']
    if terminal:
        margin=terminal['support_margin_m']
        extra=20.*(terminal['horizontal']**2+terminal['vertical_speed']**2)+10.*terminal['omega']**2
        extra+=10.*(terminal['roll']**2+terminal['pitch']**2)+.1*terminal['joint_speed']**2
        extra+=10.*max(0.,-(margin if margin is not None else -.2)/.1)**2
        result['original_cost']=result['cost'];result['cost']+=extra;result['rank'][-1]=result['cost']
    result.update(software_arm_command_slew=None,hip_mount_collision_response=True)
    return result
