"""Time-varying local command model along a causal, constraint-checked rollout.

Knot variables are increments to the complete nominal command sequence, so
zero increments retain its trajectory. Dynamics are frozen only within each
10 ms interval. Geometry derivatives are local approximations, independently
replayed before a proposal can be considered model-feasible. No native entry.
"""
import copy
import numpy as np
import pinocchio as pin
from scipy.linalg import lu_factor, lu_solve
from scripts.condensed_implicit_drive_v1 import condense_implicit_drive
from scripts.frame_point_kinematics_v1 import frame_point_kinematics


def world_displacement(model, initial, q):
    result=np.zeros(model.model.nv)
    result[:3]=q[:3]-initial.q[:3]
    result[3:6]=pin.log3(pin.Quaternion(q[3:7]).matrix()@initial.root_rotation_world.T)
    result[model.v_indices]=q[model.q_indices]-initial.joint_position
    return result


def make_nominal_profile(controller, initial, plan):
    """Only a complete physically accepted rollout can seed this local model.

The rollout may fail terminal recovery. Its constraints and terminal result
remain distinct. Inputs are predicted from the query, never future trace data.
"""
    horizon=plan.get('horizon',[])
    if len(horizon)!=controller.settings.horizon_steps or not all(r['accepted'] for r in horizon):
        return None
    model=controller.qp.model; state=copy.deepcopy(initial); result=[]
    for row in horizon:
        command=controller.admit_command_target(state,np.asarray(row['joint_position_target']))
        snapshot=copy.deepcopy(model.snapshot(state.q,state.v))
        following=controller._integrate(state,row['acceleration'],row)
        following,added=controller._after_prediction(following)
        end_snapshot=copy.deepcopy(model.snapshot(following.q,following.v))
        com_jac,_=controller.qp.com_kinematics(following);com_jac=com_jac.copy()
        world_to_local=np.eye(24)
        world_to_local[:3,:3]=following.root_rotation_world.T
        world_to_local[3:6,3:6]=following.root_rotation_world.T
        frames=frame_point_kinematics(model,following.q,following.v,
            ('FL_thigh','FR_thigh','RL_thigh','RR_thigh'))
        result.append(dict(state=copy.deepcopy(state),following=copy.deepcopy(following),
            snapshot=snapshot,command=command,end_snapshot=end_snapshot,
            com_jac_world=com_jac@world_to_local,world_to_local=world_to_local,
            frames=frames,displacement=world_displacement(model,initial,following.q),
            added_contacts=added))
        state=following
    return result


def affine_trajectory_sequence(controller, initial, basis, profile):
    """Yield microstep affine maps; zero knots follow the full nominal sequence."""
    qp=controller.qp; model=qp.model; interface=qp.interface
    dt=controller.settings.control_dt_s; h=dt/qp.substeps
    nv,na=24,18; width=basis.shape[1]*na
    S=np.zeros((nv,na));S[model.v_indices,np.arange(na)]=1.
    vm=np.zeros((nv,width));vb=initial.v.copy()
    qm=np.zeros((na,width));qb=initial.joint_position.copy()
    dm=np.zeros((nv,width));db=np.zeros(nv)
    if len(profile)!=len(basis):raise ValueError('One nominal state per control interval required')
    for tick,(weights,node) in enumerate(zip(basis,profile)):
        state=node['state'];snap=node['snapshot'];command=node['command']
        feet=np.flatnonzero(state.contacts);jac=snap['foot_jacobian'][feet].reshape(-1,nv)
        jdot=(snap['foot_jacobian_dot'][feet]@state.v).reshape(-1);nc=len(jac)
        M=snap['mass_matrix'];kp=interface.kp;kd=interface.kd
        factor=lu_factor(np.block([[M+h*(S*(kd+h*kp))@S.T,-h*jac.T],
                                  [jac,np.zeros((nc,nc))]]))
        maps=lu_solve(factor,np.c_[np.vstack((M,(1-h/qp.settings.contact_velocity_decay_s)*jac)),
            np.vstack((-h*S*kp,np.zeros((nc,na)))),
            np.vstack((h*S*kp,np.zeros((nc,na)))),np.r_[-h*snap['nonlinear'],-h*jdot]])
        av,aq,au,b=maps[:,:nv],maps[:,nv:nv+na],maps[:,nv+na:nv+2*na],maps[:,-1]
        um=np.kron(weights[None,:],np.eye(na))
        nominal_micro=condense_implicit_drive(M,snap['nonlinear'],S,kp,kd,jac,jdot,
            state.v,state.joint_position,duration_s=dt,substeps=qp.substeps,
            contact_velocity_decay_s=qp.settings.contact_velocity_decay_s)
        nominal_q=state.q.copy()
        transform=np.eye(nv);transform[:3,:3]=state.root_rotation_world
        transform[3:6,3:6]=state.root_rotation_world
        for micro in range(qp.substeps):
            rm=av@vm+aq@qm+au@um;rb=av@vb+aq@qb+au@command+b
            vm,vb=rm[:nv],rb[:nv]
            qm=qm+h*S.T@vm;qb=qb+h*S.T@vb
            dm=dm+h*transform@vm
            nominal_velocity=nominal_micro[micro]['velocity_map']@command+nominal_micro[micro]['velocity_bias']
            nominal_q=pin.integrate(model.model,nominal_q,h*nominal_velocity)
            # Manifold coordinate offset keeps the local approximation exact
            # at zero increments; no force/velocity constraint is overwritten.
            db=world_displacement(model,initial,nominal_q)
            yield dict(tick=tick,micro=micro,command_map=um,command_bias=command,
                velocity_map=vm,velocity_bias=vb,joint_position_map=qm,joint_position_bias=qb,
                displacement_map=dm,displacement_bias=db,force_map=rm[nv:],force_bias=rb[nv:],
                torque_map=kp[:,None]*(um-qm)-kd[:,None]*(S.T@vm),
                torque_bias=kp*(command-qb)-kd*(S.T@vb))
        if max(np.max(abs(vb-node['following'].v)),np.max(abs(qb-node['following'].joint_position)))>1e-5:
            raise ValueError('Zero-increment dynamics did not reproduce the nominal interval')
