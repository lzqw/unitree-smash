"""Put the existing terminal recovery conditions into local sequence QPs.

Linearized inequalities guide the search; the unchanged nonlinear verifier
still decides acceptance. Nominal contact modes and finite differences apply.
"""
import copy
import numpy as np
import pinocchio as pin
import scripts.landing_joint_sequence_proposal_v1 as original
from scripts.microstep_ground_sequence_v1 import generated_source,microstep_clearance_rows


def terminal_values(controller,state):
    r=controller.risk(state);feet=int(np.sum(state.contacts));cfg=controller.settings
    if feet>=3:margin=cfg.terminal_support_margin_m
    elif feet==2:margin=-cfg.two_foot_max_com_distance_m
    elif feet==1:margin=-cfg.one_foot_max_com_distance_m
    else:raise ValueError('No support at terminal node')
    # All rows are upper bounds; the last converts the lower margin bound.
    return (np.array([r['tilt_rad'],r['angular_speed_rad_s'],r['horizontal_speed_mps'],-r['support_margin_m']]),
        np.array([cfg.terminal_tilt_rad,cfg.terminal_angular_speed_rad_s,cfg.terminal_horizontal_speed_mps,-margin]))


def terminal_rows(controller,state,configuration_map,velocity_map):
    value,limit=terminal_values(controller,state);model=controller.qp.model
    jac=np.zeros((4,48));eps=np.r_[np.full(24,1e-6),np.full(24,1e-5)]
    for col,h in enumerate(eps):
        responses=[]
        for sign in [-1.,1.]:
            trial=copy.deepcopy(state)
            if col<24:
                delta=np.zeros(24);delta[col]=sign*h;trial.q=pin.integrate(model.model,state.q,delta)
                trial.root_position_world=trial.q[:3].copy();trial.root_rotation_world=pin.Quaternion(trial.q[3:7]).matrix()
                trial.joint_position=trial.q[model.q_indices].copy()
            else:
                trial.v=state.v.copy();trial.v[col-24]+=sign*h;trial.joint_velocity=trial.v[model.v_indices].copy()
            responses.append(terminal_values(controller,trial)[0])
        jac[:,col]=(responses[1]-responses[0])/(2*h)
    return value,jac@np.vstack((configuration_map,velocity_map)),limit


def make_proposal(joint_velocity_weight=2.):
    source=generated_source(joint_velocity_weight)
    anchor="            terminal = st['tick'] == H-1"
    if source.count(anchor)!=1:raise ValueError('Changed terminal node')
    source=source.replace(anchor,anchor+"\n            if terminal:\n                tv,tm,tl=terminal_rows(controller,nominal_profile[-1]['following'],\n                    st['endpoint_configuration_map'],vm)\n                bound(tm,tv,-np.inf,tl-1e-5,'terminal_recovery',seed=True)")
    anchor="    row_tolerance[kinds_for_tolerance=='authored_ground_clearance_m']=1e-7"
    if source.count(anchor)!=1:raise ValueError('Changed row tolerance declaration')
    source=source.replace(anchor,anchor+"\n    row_tolerance[kinds_for_tolerance=='terminal_recovery']=1e-7")
    namespace=dict(vars(original));namespace.update(microstep_clearance_rows=microstep_clearance_rows,terminal_rows=terminal_rows)
    exec(compile(source,'<terminal-constrained-recovery>','exec'),namespace)
    return namespace['propose_sequence']
