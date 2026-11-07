"""Bounded Coulomb contact iteration coupled to implicit position PD.

Each contact alternates unilateral normal complementarity and maximum-
dissipation friction on the circular disk. This is a nonconvex contact-law
fixed point, not an always-convergent convex QP or a PhysX replica. Failure to
satisfy the contact law is reported instead of returning a feasible label.
"""
import numpy as np


def metric_disk_minimum(matrix,bias,radius):
    if radius<=0:return np.zeros(2)
    values,vectors=np.linalg.eigh(matrix)
    if values.min()<=0:raise ValueError('Nonpositive tangential effective mass')
    projected=vectors.T@bias
    free=-projected/values
    if np.linalg.norm(free)<=radius:return vectors@free
    low=0.;high=np.linalg.norm(projected)/radius
    for _ in range(40):
        middle=(low+high)/2
        if np.linalg.norm(projected/(values+middle))>radius:low=middle
        else:high=middle
    return vectors@(-projected/(values+high))


def contact_law_residual(matrix,bias,impulse,mu):
    velocity=(matrix@impulse+bias).reshape(-1,3);p=impulse.reshape(-1,3)
    normal=float(np.max(abs(np.minimum(p[:,2],velocity[:,2])),initial=0))
    cone=float(np.max(np.linalg.norm(p[:,:2],axis=1)-mu*p[:,2],initial=0))
    tangent=0.
    for index in range(len(p)):
        block=slice(3*index,3*index+2)
        h=matrix[block,block];b=bias[block]+matrix[block]@impulse-h@p[index,:2]
        expected=metric_disk_minimum(h,b,mu*max(0.,p[index,2]))
        tangent=max(tangent,float(np.max(abs(expected-p[index,:2]))))
    return dict(normal_complementarity=normal,friction_cone=cone,tangent_fixed_point=tangent)


def coulomb_impulses(matrix,bias,mu,*,warm=None,max_iterations=400,tolerance=2e-7):
    matrix=np.asarray(matrix,float);bias=np.asarray(bias,float);n=len(bias)
    if (n%3 or matrix.shape!=(n,n) or not np.isfinite(matrix).all() or not np.isfinite(bias).all()
            or not np.isfinite(mu) or mu<0 or not np.allclose(matrix,matrix.T,atol=1e-10)
            or type(max_iterations) is not int or max_iterations<1 or not 0<tolerance<1e-3):
        raise ValueError('Invalid contact problem')
    if not n:return np.zeros(0),dict(iterations=0,residual={})
    if np.any(np.diag(matrix)<=0):raise ValueError('Nonpositive contact effective mass')
    p=np.zeros(n) if warm is None else np.asarray(warm,float).copy()
    if p.shape!=(n,) or not np.isfinite(p).all():raise ValueError('Invalid impulse warm start')
    for i in range(n//3):
        k=3*i;p[k+2]=max(0.,p[k+2]);norm=np.linalg.norm(p[k:k+2]);limit=mu*p[k+2]
        if norm>limit:p[k:k+2]*=limit/norm
    for iteration in range(max_iterations):
        for i in range(n//3):
            k=3*i;normal=k+2
            p[normal]=max(0.,p[normal]-(matrix[normal]@p+bias[normal])/matrix[normal,normal])
            tang=slice(k,k+2);h=matrix[tang,tang]
            b=bias[tang]+matrix[tang]@p-h@p[tang]
            p[tang]=metric_disk_minimum(h,b,mu*p[normal])
        residual=contact_law_residual(matrix,bias,p,mu)
        if max(residual.values())<=tolerance:
            return p,dict(iterations=iteration+1,residual=residual)
    raise ValueError('Coulomb contact law did not converge: '+str(residual))


def frictional_velocity_step(mass,velocity,bias,selection,kp,kd,target_error,effort,
                            dt,jacobian,normal_target,mu,*,warm=None,max_iterations=400):
    nv=len(velocity);na=selection.shape[1];nc=len(normal_target)
    arrays=[mass,velocity,bias,selection,kp,kd,target_error,effort,jacobian,normal_target]
    if (mass.shape!=(nv,nv) or selection.shape!=(nv,na) or jacobian.shape!=(3*nc,nv)
            or any(np.shape(x)!=(na,) for x in [kp,kd,target_error,effort])
            or np.shape(bias)!=(nv,) or any(not np.isfinite(x).all() for x in arrays)
            or not np.isfinite(dt) or dt<=0 or np.any(kp<=0) or np.any(kd<0) or np.any(effort<=0)):
        raise ValueError('Invalid implicit friction step')
    slope=kd+dt*kp;drive_bias=kp*target_error;saturated=np.zeros(na,np.int8);seen=set()
    for motor_iteration in range(2*na+2):
        key=tuple(saturated.tolist())
        if key in seen:raise ValueError('Motor active set cycled')
        seen.add(key);free=saturated==0
        effective=mass+dt*(selection[:,free]*slope[free])@selection[:,free].T
        drive=drive_bias.copy();drive[~free]=saturated[~free]*effort[~free]
        rhs=mass@velocity-dt*bias+dt*selection@drive
        solved=np.linalg.solve(effective,np.column_stack([rhs,jacobian.T]))
        vfree=solved[:,0];response=solved[:,1:]
        W=jacobian@response;W=(W+W.T)/2
        offset=jacobian@vfree;offset[2::3]-=normal_target
        impulse,contact=coulomb_impulses(W,offset,mu,warm=warm,max_iterations=max_iterations)
        following=vfree+response@impulse
        torque_raw=drive_bias-slope*(selection.T@following)
        new=np.where(torque_raw>effort+1e-8,1,np.where(torque_raw < -effort-1e-8,-1,0)).astype(np.int8)
        if np.array_equal(new,saturated):
            torque=np.clip(torque_raw,-effort,effort)
            residual=mass@(following-velocity)+dt*bias-dt*selection@torque-jacobian.T@impulse
            if np.max(abs(residual),initial=0)>1e-8:raise ValueError('Momentum balance failed')
            return dict(velocity=following,impulse=impulse,torque=torque,contact=contact,
                momentum_residual=float(np.max(abs(residual),initial=0)),motor_iterations=motor_iteration+1)
        saturated=new
    raise ValueError('Motor active set did not converge')
