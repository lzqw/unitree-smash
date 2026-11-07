"""Offline substep ground constraints for the existing differentiated model.

This module creates a process-local proposal variant. It does not change the
frozen native queue's sources or relax independent trajectory acceptance.
"""
import inspect
import numpy as np
import pinocchio as pin
import scripts.landing_joint_sequence_proposal_v1 as original


def configuration_from_displacement(model, initial, displacement):
    q=initial.q.copy();q[:3]+=displacement[:3]
    q[3:7]=pin.Quaternion(pin.exp3(displacement[3:6])@initial.root_rotation_world).coeffs()
    q[model.q_indices]=initial.joint_position+displacement[model.v_indices]
    return q


def microstep_clearance_rows(geometry, model, initial, stage):
    """Chain the already differentiated state into each collider's height."""
    db=stage['displacement_bias'];dm=stage['displacement_map']
    q=configuration_from_displacement(model,initial,db)
    # SE(3) uses body-local tangents; displacement stores world translation
    # and log(R R_initial^T). Differentiate only the three rotation coordinates.
    transform=np.eye(model.model.nv)
    transform[:3,:3]=pin.Quaternion(q[3:7]).matrix().T
    for axis in range(3):
        delta=np.zeros(model.model.nv);delta[3+axis]=1e-6
        plus=configuration_from_displacement(model,initial,db+delta)
        minus=configuration_from_displacement(model,initial,db-delta)
        transform[3:6,3+axis]=(pin.difference(model.model,q,plus)[3:6]-
            pin.difference(model.model,q,minus)[3:6])/(2e-6)
    distance,jacobian=geometry.evaluate(q,jacobian=True)
    return distance,jacobian@transform@dm


def generated_source(joint_velocity_weight=2.):
    if not np.isfinite(joint_velocity_weight) or joint_velocity_weight<=0:
        raise ValueError('Positive finite joint velocity weight required')
    source=inspect.getsource(original.propose_sequence)
    def once(old,new):
        nonlocal source
        if source.count(old)!=1:raise ValueError('Frozen proposal changed: '+old[:100])
        source=source.replace(old,new,1)
    once("        if st['micro'] == qp.substeps-1:\n            if geometry is not None:\n                distances,distance_jac=geometry.evaluate(nominal_profile[st['tick']]['following'].q,jacobian=True)\n                bound(distance_jac@st['endpoint_configuration_map'],distances,\n                    geometry.margin_m,np.inf,'authored_ground_clearance_m',seed=True)",
         "        if geometry is not None:\n            distances,distance_map=microstep_clearance_rows(geometry,model,state,st)\n            bound(distance_map,distances,geometry.margin_m,np.inf,\n                  'authored_ground_clearance_m',seed=False)\n        if st['micro'] == qp.substeps-1:")
    once('            task(vm[3:6], vb[3:6], np.zeros(3), 2.*w)',
         '            task(vm[3:6], vb[3:6], np.zeros(3), 2.*w)\n'+
         f'            task(S.T@vm, S.T@vb, np.zeros(18), {float(joint_velocity_weight)!r}*w)')
    # Keep the old variable layout, but force geometry slack to zero. These
    # candidates already have near-feasible seed paths. Infeasibility is reported.
    once('low=np.r_[low,0.];high=np.r_[high,np.inf]',
         'low=np.r_[low,0.];high=np.r_[high,0.]')
    once("result['search_only_geometry_elastic']=True",
         "result['search_only_geometry_elastic']=False\n        result['all_microstep_ground_constraints']=True")
    once('    As = A*scale;',
         "    row_tolerance=np.full(len(low),cfg.residual_tolerance)\n    kinds_for_tolerance=np.asarray(row_kinds)\n    row_tolerance[kinds_for_tolerance=='authored_ground_clearance_m']=1e-7\n    row_tolerance[kinds_for_tolerance=='search_only_clearance_slack_m']=1e-8\n    As = A*scale;")
    once('if np.max(violations)<=cfg.residual_tolerance:break',
         'if np.all(violations<=row_tolerance):break')
    once('candidates=np.flatnonzero(violations>cfg.residual_tolerance)',
         'candidates=np.flatnonzero(violations>row_tolerance)')
    once('if error > cfg.residual_tolerance:',
         'if np.any(np.maximum(low-A@x,A@x-high)>row_tolerance):')
    return source


def make_proposal(joint_velocity_weight=2.):
    namespace=dict(vars(original));namespace['microstep_clearance_rows']=microstep_clearance_rows
    source=generated_source(joint_velocity_weight)
    exec(compile(source,'<microstep-ground-proposal>','exec'),namespace)
    return namespace['propose_sequence']
