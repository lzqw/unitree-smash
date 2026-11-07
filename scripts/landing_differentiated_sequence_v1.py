"""Differentiate the existing interval model about a predicted nominal path.

Central differences include loaded M/h/J changes caused by state perturbations.
Contact schedule remains nominal: this does not differentiate hybrid event
timing or certify native execution. Deliberately CPU/offline, not a fast MPC.
"""
import copy
import numpy as np
import pinocchio as pin
from scripts.implicit_support_prediction_v1 import implicit_velocity_step
from scripts.landing_sequence_linearization_v1 import world_displacement


def interval_outputs(controller, initial, node, perturbation):
    """Return every microstep and next tangent state for a 66D perturbation."""
    qp=controller.qp;model=qp.model;interface=qp.interface
    perturbation=np.asarray(perturbation,dtype=float)
    if perturbation.shape!=(66,) or not np.isfinite(perturbation).all():
        raise ValueError('Finite configuration/velocity/command perturbation required')
    state=node['state']
    q=pin.integrate(model.model,state.q,perturbation[:24])
    v=state.v+perturbation[24:48];command=node['command']+perturbation[48:]
    snap=copy.deepcopy(model.snapshot(q,v));feet=np.flatnonzero(state.contacts)
    jac=snap['foot_jacobian'][feet].reshape(-1,24)
    jdot=(snap['foot_jacobian_dot'][feet]@v).reshape(-1)
    S=np.zeros((24,18));S[model.v_indices,np.arange(18)]=1.
    h=controller.settings.control_dt_s/qp.substeps;factors={};outputs=[]
    for _ in range(qp.substeps):
        step=implicit_velocity_step(snap['mass_matrix'],v,snap['nonlinear'],S,
            interface.kp,interface.kd,command-q[model.q_indices],interface.effort,h,
            jac,h*(-jac@v/qp.settings.contact_velocity_decay_s-jdot),factors=factors)
        v=step['velocity'];q=pin.integrate(model.model,q,h*v)
        # Constrain raw PD torque as in the unsaturated interval QP. Merely
        # clipping an arbitrarily large torque request is not a feasible plan.
        raw_torque=interface.kp*(command-q[model.q_indices])-interface.kd*v[model.v_indices]
        outputs.append(np.r_[v,q[model.q_indices],world_displacement(model,initial,q),
                             step['force'],raw_torque])
    next_delta=np.r_[pin.difference(model.model,node['following'].q,q),v-node['following'].v]
    return np.asarray(outputs),next_delta


def interval_derivatives(controller,initial,node,*,position_eps=1e-6,velocity_eps=1e-5,command_eps=1e-6):
    steps=np.r_[np.full(24,position_eps),np.full(24,velocity_eps),np.full(18,command_eps)]
    if not np.isfinite(steps).all() or np.min(steps)<=0:raise ValueError('Positive finite difference scales required')
    nominal,next_nominal=interval_outputs(controller,initial,node,np.zeros(66))
    if np.max(abs(next_nominal))>1e-6:raise ValueError('Nominal interval was not independently reproduced')
    output_jac=np.empty((*nominal.shape,66));next_jac=np.empty((48,66))
    for column,h in enumerate(steps):
        delta=np.zeros(66);delta[column]=h
        plus,next_plus=interval_outputs(controller,initial,node,delta)
        minus,next_minus=interval_outputs(controller,initial,node,-delta)
        output_jac[:,:,column]=(plus-minus)/(2*h)
        next_jac[:,column]=(next_plus-next_minus)/(2*h)
    if not np.isfinite(output_jac).all() or not np.isfinite(next_jac).all():
        raise ValueError('Nonfinite local derivative')
    return nominal,output_jac,next_jac


def affine_differentiated_sequence(controller,initial,basis,profile):
    """Propagate all knot derivatives through the refreshed interval dynamics."""
    if len(basis)!=len(profile):raise ValueError('Profile length mismatch')
    width=basis.shape[1]*18;state_jac=np.zeros((48,width))
    for tick,(weights,node) in enumerate(zip(basis,profile)):
        nominal,jac,next_jac=interval_derivatives(controller,initial,node)
        command_map=np.kron(weights[None,:],np.eye(18))
        input_map=np.vstack((state_jac,command_map))
        propagated=jac@input_map
        nf=int(node['state'].contacts.sum())*3
        sections=dict(velocity=slice(0,24),joint_position=slice(24,42),
            displacement=slice(42,66),force=slice(66,66+nf),torque=slice(66+nf,84+nf))
        for micro in range(controller.qp.substeps):
            result=dict(tick=tick,micro=micro,command_map=command_map,command_bias=node['command'])
            for name,section in sections.items():
                result[name+'_map']=propagated[micro,section]
                result[name+'_bias']=nominal[micro,section]
            yield result
        state_jac=next_jac@input_map
