"""Coupled implicit PD, Coulomb contact and simulator joint-speed constraints.

Speed-limit impulses are internal generalized constraint reactions, separate
from bounded motor torque. This models the declared simulation limit, not a
real actuator performance envelope or PhysX's exact finite iteration order.
"""
import numpy as np

from scripts.frictional_impulse_step_v1 import (
    frictional_velocity_step, metric_disk_minimum, contact_law_residual,
)


def contact_and_speed_impulses(matrix, bias, contact_count, limits, mu, *,
                               warm=None, max_iterations=400, tolerance=2e-7):
    n=3*contact_count;limits=np.asarray(limits,float);total=n+len(limits)
    if (matrix.shape!=(total,total) or bias.shape!=(total,) or np.any(limits<=0)
            or not np.isfinite(matrix).all() or not np.isfinite(bias).all()
            or not np.isfinite(limits).all() or not np.allclose(matrix,matrix.T,atol=1e-10)
            or np.any(np.diag(matrix)<=0) or not np.isfinite(mu) or mu<0
            or type(max_iterations) is not int or max_iterations<1):
        raise ValueError('Invalid coupled contact/speed problem')
    impulse=np.zeros(total)
    if warm is not None:
        warm=np.asarray(warm,float)
        if warm.shape!=(n,) or not np.isfinite(warm).all():raise ValueError('Invalid foot impulse warm start')
        impulse[:n]=warm
    for foot in range(contact_count):
        k=3*foot;impulse[k+2]=max(0.,impulse[k+2]);norm=np.linalg.norm(impulse[k:k+2])
        if norm>mu*impulse[k+2]:impulse[k:k+2]*=mu*impulse[k+2]/norm
    for iteration in range(max_iterations):
        for foot in range(contact_count):
            k=3*foot;normal=k+2;tangent=slice(k,k+2)
            impulse[normal]=max(0.,impulse[normal]-(matrix[normal]@impulse+bias[normal])/matrix[normal,normal])
            h=matrix[tangent,tangent]
            b=bias[tangent]+matrix[tangent]@impulse-h@impulse[tangent]
            impulse[tangent]=metric_disk_minimum(h,b,mu*impulse[normal])
        # Signed joint reaction: positive enforces the lower speed bound,
        # negative enforces the upper. The full response also moves the base.
        for joint,limit in enumerate(limits):
            k=n+joint;without=bias[k]+matrix[k]@impulse-matrix[k,k]*impulse[k]
            impulse[k]=(np.clip(without,-limit,limit)-without)/matrix[k,k]
        contact=contact_law_residual(matrix[:n,:n],bias[:n]+matrix[:n,n:]@impulse[n:],impulse[:n],mu)
        speed=bias[n:]+matrix[n:]@impulse
        violation=float(np.max(abs(speed)-limits,initial=0.))
        without=speed-np.diag(matrix)[n:]*impulse[n:]
        expected=(np.clip(without,-limits,limits)-without)/np.diag(matrix)[n:]
        fixed_point=float(np.max(abs(expected-impulse[n:]),initial=0.))
        if max([violation,fixed_point,*contact.values()])<=tolerance:
            return impulse,dict(iterations=iteration+1,contact=contact,
                speed_violation=violation,speed_fixed_point=fixed_point)
    raise ValueError('Coupled contact/speed law did not converge: '+str(dict(
        contact=contact,speed_violation=violation,speed_fixed_point=fixed_point)))


def speed_limited_velocity_step(mass,velocity,bias,selection,kp,kd,target_error,effort,
                                dt,jacobian,normal_target,mu,velocity_limits,*,warm=None,max_iterations=400):
    limits=np.asarray(velocity_limits,float);na=selection.shape[1]
    if limits.shape!=(na,) or not np.isfinite(limits).all() or np.any(limits<=0):
        raise ValueError('Explicit original simulator velocity limits required')
    # Preserve the old numerical path exactly whenever the new constraints
    # are inactive. This also retains the original input validation.
    original=frictional_velocity_step(mass,velocity,bias,selection,kp,kd,target_error,effort,
        dt,jacobian,normal_target,mu,warm=warm,max_iterations=max_iterations)
    if np.max(abs(selection.T@original['velocity'])-limits,initial=0.)<=2e-7:
        return dict(original,velocity_limit_impulse=np.zeros(na),velocity_limit_active=False,
            speed_violation=float(np.max(abs(selection.T@original['velocity'])-limits,initial=0.)),
            speed_fixed_point=0.)
    nc=len(normal_target);n=3*nc;augmented=np.vstack([jacobian,selection.T])
    slope=kd+dt*kp;drive_bias=kp*target_error;saturated=np.zeros(na,np.int8);seen=set()
    for motor_iteration in range(2*na+2):
        key=tuple(saturated.tolist())
        if key in seen:raise ValueError('Speed-limited motor active set cycled')
        seen.add(key);free=saturated==0
        effective=mass+dt*(selection[:,free]*slope[free])@selection[:,free].T
        drive=drive_bias.copy();drive[~free]=saturated[~free]*effort[~free]
        rhs=mass@velocity-dt*bias+dt*selection@drive
        solved=np.linalg.solve(effective,np.column_stack([rhs,augmented.T]))
        vfree=solved[:,0];response=solved[:,1:];matrix=augmented@response;matrix=(matrix+matrix.T)/2
        offset=augmented@vfree;offset[2:n:3]-=normal_target
        all_impulse,law=contact_and_speed_impulses(matrix,offset,nc,limits,mu,warm=warm,
            max_iterations=max_iterations)
        following=vfree+response@all_impulse;torque_raw=drive_bias-slope*(selection.T@following)
        new=np.where(torque_raw>effort+1e-8,1,np.where(torque_raw < -effort-1e-8,-1,0)).astype(np.int8)
        if np.array_equal(new,saturated):
            torque=np.clip(torque_raw,-effort,effort)
            residual=mass@(following-velocity)+dt*bias-dt*selection@torque-augmented.T@all_impulse
            if np.max(abs(residual),initial=0.)>1e-8:raise ValueError('Coupled momentum balance failed')
            return dict(velocity=following,impulse=all_impulse[:n],torque=torque,
                velocity_limit_impulse=all_impulse[n:],velocity_limit_active=True,
                speed_violation=law['speed_violation'],speed_fixed_point=law['speed_fixed_point'],
                contact=dict(iterations=law['iterations'],residual=law['contact']),
                momentum_residual=float(np.max(abs(residual),initial=0.)),motor_iterations=motor_iteration+1)
        saturated=new
    raise ValueError('Speed-limited motor active set did not converge')
