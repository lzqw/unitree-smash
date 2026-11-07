"""Affine held-command response over all implicit-PD substeps.

Unsaturated branch, frozen loaded mass/bias/contact Jacobian per 10ms interval.
This is a model building block, not a new native controller or PhysX replica.
The command is constant; every intermediate force/torque/state is retained so
an optimizer can constrain the interval rather than only its endpoint.
"""
import numpy as np
from scipy.linalg import lu_factor,lu_solve


def condense_implicit_drive(mass,bias,selection,kp,kd,jacobian,jdot_v,
                           velocity,joint_position,*,duration_s=.01,substeps=16,
                           contact_velocity_decay_s=.06):
    mass=np.asarray(mass,dtype=float);selection=np.asarray(selection,dtype=float)
    nv,na=selection.shape;nc=len(jacobian)
    arrays=(mass,bias,selection,kp,kd,jacobian,jdot_v,velocity,joint_position)
    if (type(substeps) is not int or substeps<1 or not 0<duration_s<=.02 or
            contact_velocity_decay_s<=0 or any(not np.isfinite(x).all() for x in arrays)):
        raise ValueError('Finite short-interval model required')
    dt=duration_s/substeps
    if (mass.shape!=(nv,nv) or np.shape(bias)!=(nv,) or np.shape(kp)!=(na,) or
            np.shape(kd)!=(na,) or np.shape(joint_position)!=(na,) or
            np.shape(velocity)!=(nv,) or np.shape(jacobian)!=(nc,nv) or np.shape(jdot_v)!=(nc,)):
        raise ValueError('Incompatible loaded model dimensions')
    if np.any(kp<=0) or np.any(kd<0):raise ValueError('Physical PD gains required')
    slope=kd+dt*kp
    effective=mass+dt*(selection*slope)@selection.T
    matrix=np.block([[effective,-dt*jacobian.T],[jacobian,np.zeros((nc,nc))]])
    factor=lu_factor(matrix)
    rhs_v=np.vstack((mass,(1.-dt/contact_velocity_decay_s)*jacobian))
    rhs_q=np.vstack((-dt*selection*kp,np.zeros((nc,na))))
    rhs_u=np.vstack((dt*selection*kp,np.zeros((nc,na))))
    rhs_b=np.r_[-dt*bias,-dt*jdot_v]
    maps=lu_solve(factor,np.c_[rhs_v,rhs_q,rhs_u,rhs_b])
    v_to=maps[:,:nv];q_to=maps[:,nv:nv+na];u_to=maps[:,nv+na:nv+2*na];offset=maps[:,-1]
    vm=np.zeros((nv,na));vb=np.asarray(velocity,dtype=float).copy()
    qm=np.zeros((na,na));qb=np.asarray(joint_position,dtype=float).copy()
    stages=[]
    for index in range(substeps):
        response_map=v_to@vm+q_to@qm+u_to
        response_bias=v_to@vb+q_to@qb+offset
        vm=response_map[:nv];vb=response_bias[:nv]
        fm=response_map[nv:];fb=response_bias[nv:]
        qm=qm+dt*selection.T@vm;qb=qb+dt*selection.T@vb
        tm=kp[:,None]*(np.eye(na)-qm)-kd[:,None]*(selection.T@vm)
        tb=-kp*qb-kd*(selection.T@vb)
        stages.append(dict(time_s=(index+1)*dt,velocity_map=vm.copy(),velocity_bias=vb.copy(),
            joint_position_map=qm.copy(),joint_position_bias=qb.copy(),
            force_map=fm.copy(),force_bias=fb.copy(),torque_map=tm,torque_bias=tb))
    return stages


def interval_constraints(stages,selection,*,effort,velocity_limit,joint_limits,
                         friction,friction_facets=8):
    """Linear inequalities low <= A @ held_command <= high at every substep."""
    na=selection.shape[1];matrices=[];low=[];high=[];labels=[]
    if not stages or friction<=0 or friction_facets<4:raise ValueError('Physical support bounds required')
    def add(matrix,bias,lower,upper,stage,kind):
        matrices.append(matrix);low.extend(np.broadcast_to(lower,len(bias))-bias)
        high.extend(np.broadcast_to(upper,len(bias))-bias)
        labels.extend((stage,kind,i) for i in range(len(bias)))
    for i,s in enumerate(stages):
        add(s['torque_map'],s['torque_bias'],-effort,effort,i,'motor_torque')
        add(selection.T@s['velocity_map'],selection.T@s['velocity_bias'],
            -velocity_limit,velocity_limit,i,'joint_velocity')
        add(s['joint_position_map'],s['joint_position_bias'],joint_limits[:,0],joint_limits[:,1],i,'joint_position')
        for foot in range(len(s['force_bias'])//3):
            fm=s['force_map'][3*foot:3*foot+3];fb=s['force_bias'][3*foot:3*foot+3]
            add(fm[2:3],fb[2:3],0.,np.inf,i,f'normal_force_{foot}')
            angles=np.arange(friction_facets)*2.*np.pi/friction_facets
            cone=np.c_[np.cos(angles),np.sin(angles),np.full(friction_facets,-friction*np.cos(np.pi/friction_facets))]
            add(cone@fm,cone@fb,-np.inf,0.,i,f'friction_{foot}')
    return dict(matrix=np.vstack(matrices).reshape(-1,na),low=np.asarray(low),high=np.asarray(high),labels=labels)
