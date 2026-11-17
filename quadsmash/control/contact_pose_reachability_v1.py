"""Bounded local pose/velocity diagnosis; failure is not global infeasibility."""
import time
import numpy as np
from scipy.optimize import least_squares,lsq_linear


def diagnose_contact_pose(kinematics,q0,lower,upper,speed,position,normal,velocity,
                          base_velocity_fn=None,max_nfev=40):
    started=time.perf_counter();q0=np.clip(q0,lower,upper)
    position=np.asarray(position);normal=np.asarray(normal);normal=normal/np.linalg.norm(normal)
    rows={}
    for label,with_normal in [('position_only',False),('position_normal',True)]:
        def fun(q):
            p,n,jp,jn=kinematics(q)
            return np.r_[(p-position)/.035,(n-normal)/.12] if with_normal else (p-position)/.035
        def jac(q):
            p,n,jp,jn=kinematics(q)
            return np.vstack((jp/.035,jn/.12)) if with_normal else jp/.035
        fit=least_squares(fun,q0,jac=jac,bounds=(lower,upper),max_nfev=max_nfev)
        p,n,jp,jn=kinematics(fit.x)
        base=np.zeros(3) if base_velocity_fn is None else base_velocity_fn(p)
        rate=lsq_linear(np.vstack((jp,.04*np.eye(6))),np.r_[velocity-base,np.zeros(6)],
                        bounds=(-np.asarray(speed),np.asarray(speed)),max_iter=50)
        actual=base+jp@rate.x
        ep=float(np.linalg.norm(p-position));en=float(np.linalg.norm(n-normal));ev=float(np.linalg.norm(actual-velocity))
        rows[label]=dict(q_rad=fit.x.tolist(),qd_rad_s=rate.x.tolist(),position_m=p.tolist(),normal=n.tolist(),
            velocity_m_s=actual.tolist(),position_error_m=ep,normal_vector_error=en,
            normal_angle_rad=float(np.arccos(np.clip(n@normal,-1,1))),velocity_error_m_s=ev,
            joint_limit_margin_rad=np.minimum(fit.x-lower,upper-fit.x).tolist(),
            speed_margin_rad_s=(speed-np.abs(rate.x)).tolist(),
            position_jacobian_singular_values=np.linalg.svd(jp,compute_uv=False).tolist(),
            task_jacobian_singular_values=np.linalg.svd(np.vstack((jp,jn)),compute_uv=False).tolist(),
            solver_success=bool(fit.success),solver_status=int(fit.status),nfev=int(fit.nfev),
            velocity_solver_success=bool(rate.success),
            kinematic_feasible=bool(ep<=.035 and en<=.12 and ev<=.75),
            first_failed_check=('position' if ep>.035 else 'normal' if en>.12 else 'velocity' if ev>.75 else None))
    return dict(target_position_m=position.tolist(),target_normal=normal.tolist(),target_velocity_m_s=np.asarray(velocity).tolist(),
        stages=rows,seconds=time.perf_counter()-started,global_infeasibility_proven=False,dynamics_verified=False)
