"""Coupled implicit PD/contact with unilateral joint-position stops.

For the scalar revolute joints, q_next = q + dt*v_next. Position and speed
bounds therefore define one velocity interval per joint. Signed constraint
impulses are solved with foot impulses and motor saturation through the full
mass matrix. Positions and floating-base velocities are never clipped after
integration. This is a zero-restitution time-step model, not a PhysX replica.
"""
import numpy as np

from scripts.frictional_impulse_step_v1 import metric_disk_minimum, contact_law_residual
from scripts.frictional_speed_limited_step_v1 import speed_limited_velocity_step


def joint_velocity_bounds(position, position_limits, speed_limits, dt):
    q=np.asarray(position,float);bounds=np.asarray(position_limits,float);speed=np.asarray(speed_limits,float)
    if (q.ndim!=1 or bounds.shape!=(len(q),2) or speed.shape!=q.shape
            or not all(np.isfinite(x).all() for x in [q,bounds,speed])
            or not np.isfinite(dt) or dt<=0 or np.any(speed<=0) or np.any(bounds[:,0]>=bounds[:,1])):
        raise ValueError('Finite scalar joint positions and original limits required')
    lower=np.maximum(-speed,(bounds[:,0]-q)/dt)
    upper=np.minimum(speed,(bounds[:,1]-q)/dt)
    if np.any(lower>upper):raise ValueError('Initial penetration cannot be corrected within joint speed bounds')
    return lower,upper


def contact_and_box_impulses(matrix,bias,contact_count,lower,upper,mu,*,
                             warm=None,max_iterations=400,tolerance=2e-7):
    n=3*contact_count;lower=np.asarray(lower,float);upper=np.asarray(upper,float);total=n+len(lower)
    if (type(contact_count) is not int or contact_count<0 or lower.ndim!=1 or upper.shape!=lower.shape
            or matrix.shape!=(total,total) or bias.shape!=(total,)
            or not all(np.isfinite(x).all() for x in [matrix,bias,lower,upper])
            or np.any(lower>upper) or not np.allclose(matrix,matrix.T,atol=1e-10)
            or np.any(np.diag(matrix)<=0) or not np.isfinite(mu) or mu<0
            or type(max_iterations) is not int or max_iterations<1 or not 0<tolerance<1e-3):
        raise ValueError('Invalid coupled contact/joint interval problem')
    impulse=np.zeros(total)
    if warm is not None:
        warm=np.asarray(warm,float)
        if warm.shape!=(n,) or not np.isfinite(warm).all():raise ValueError('Invalid foot warm start')
        impulse[:n]=warm
    for foot in range(contact_count):
        k=3*foot;impulse[k+2]=max(0.,impulse[k+2]);norm=np.linalg.norm(impulse[k:k+2])
        if norm>mu*impulse[k+2]:impulse[k:k+2]*=mu*impulse[k+2]/norm
    for iteration in range(max_iterations):
        for foot in range(contact_count):
            k=3*foot;normal=k+2;tangent=slice(k,k+2)
            impulse[normal]=max(0.,impulse[normal]-(matrix[normal]@impulse+bias[normal])/matrix[normal,normal])
            h=matrix[tangent,tangent];b=bias[tangent]+matrix[tangent]@impulse-h@impulse[tangent]
            impulse[tangent]=metric_disk_minimum(h,b,mu*impulse[normal])
        for joint in range(len(lower)):
            k=n+joint;without=bias[k]+matrix[k]@impulse-matrix[k,k]*impulse[k]
            impulse[k]=(np.clip(without,lower[joint],upper[joint])-without)/matrix[k,k]
        contact=contact_law_residual(matrix[:n,:n],bias[:n]+matrix[:n,n:]@impulse[n:],impulse[:n],mu)
        velocity=bias[n:]+matrix[n:]@impulse
        violation=float(max(np.max(lower-velocity,initial=0.),np.max(velocity-upper,initial=0.)))
        without=velocity-np.diag(matrix)[n:]*impulse[n:]
        expected=(np.clip(without,lower,upper)-without)/np.diag(matrix)[n:]
        fixed=float(np.max(abs(expected-impulse[n:]),initial=0.))
        if max([violation,fixed,*contact.values()])<=tolerance:
            return impulse,dict(iterations=iteration+1,contact=contact,box_violation=violation,box_fixed_point=fixed)
    raise ValueError('Coupled contact/joint interval law did not converge: '+str(dict(
        contact=contact,box_violation=violation,box_fixed_point=fixed)))


def position_limited_velocity_step(mass,velocity,bias,selection,kp,kd,target_error,effort,
                                  dt,jacobian,normal_target,mu,velocity_limits,position,position_limits,
                                  *,warm=None,max_iterations=400):
    lower,upper=joint_velocity_bounds(position,position_limits,velocity_limits,dt)
    na=selection.shape[1]
    if lower.shape!=(na,):raise ValueError('Joint bounds and actuator selection differ')
    # Keep the original operation order whenever position stops are inactive.
    original=speed_limited_velocity_step(mass,velocity,bias,selection,kp,kd,target_error,effort,
        dt,jacobian,normal_target,mu,velocity_limits,warm=warm,max_iterations=max_iterations)
    proposed=selection.T@original['velocity']
    if max(np.max(lower-proposed,initial=0.),np.max(proposed-upper,initial=0.))<=2e-7:
        return dict(original,position_limit_active=False,joint_limit_impulse=original['velocity_limit_impulse'].copy(),
            position_limit_impulse=np.zeros(na),joint_box_violation=0.,joint_box_fixed_point=0.)
    nc=len(normal_target);n=3*nc;augmented=np.vstack([jacobian,selection.T])
    slope=kd+dt*kp;drive_bias=kp*target_error;saturated=np.zeros(na,np.int8);seen=set()
    limits=np.asarray(velocity_limits,float)
    for motor_iteration in range(2*na+2):
        key=tuple(saturated.tolist())
        if key in seen:raise ValueError('Position-limited motor active set cycled')
        seen.add(key);free=saturated==0
        effective=mass+dt*(selection[:,free]*slope[free])@selection[:,free].T
        drive=drive_bias.copy();drive[~free]=saturated[~free]*effort[~free]
        rhs=mass@velocity-dt*bias+dt*selection@drive
        solved=np.linalg.solve(effective,np.column_stack([rhs,augmented.T]))
        vfree=solved[:,0];response=solved[:,1:];matrix=augmented@response;matrix=(matrix+matrix.T)/2
        offset=augmented@vfree;offset[2:n:3]-=normal_target
        impulse,law=contact_and_box_impulses(matrix,offset,nc,lower,upper,mu,warm=warm,max_iterations=max_iterations)
        following=vfree+response@impulse;torque_raw=drive_bias-slope*(selection.T@following)
        new=np.where(torque_raw>effort+1e-8,1,np.where(torque_raw < -effort-1e-8,-1,0)).astype(np.int8)
        if np.array_equal(new,saturated):
            torque=np.clip(torque_raw,-effort,effort)
            residual=mass@(following-velocity)+dt*bias-dt*selection@torque-augmented.T@impulse
            if np.max(abs(residual),initial=0.)>1e-8:raise ValueError('Joint-stop coupled momentum balance failed')
            joint_impulse=impulse[n:]
            # Attribution is diagnostic only: equal bounds belong to speed.
            stop=((joint_impulse>0)&(lower>-limits))|((joint_impulse<0)&(upper<limits))
            stop_impulse=np.where(stop,joint_impulse,0.);speed_impulse=np.where(stop,0.,joint_impulse)
            return dict(velocity=following,impulse=impulse[:n],torque=torque,
                joint_limit_impulse=joint_impulse,position_limit_impulse=stop_impulse,
                velocity_limit_impulse=speed_impulse,position_limit_active=bool(np.any(stop_impulse!=0)),
                velocity_limit_active=bool(np.any(speed_impulse!=0)),
                joint_box_violation=law['box_violation'],joint_box_fixed_point=law['box_fixed_point'],
                speed_violation=float(np.max(abs(selection.T@following)-limits,initial=0.)),
                speed_fixed_point=law['box_fixed_point'],
                contact=dict(iterations=law['iterations'],residual=law['contact']),
                momentum_residual=float(np.max(abs(residual),initial=0.)),motor_iterations=motor_iteration+1)
        saturated=new
    raise ValueError('Position-limited motor active set did not converge')
