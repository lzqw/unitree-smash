"""State-dependent full18 landing sequence initializations, not control proof.

Foot-placement IK proposes extended landing postures; the coupled trajectory
rollout decides whether they are dynamically realizable. These are search
seeds, never substitute trajectories or native state resets. No case IDs.
"""
import numpy as np
from scipy.optimize import least_squares
from scripts.hip_mount_planning_constraints_v1 import HipMountPlanningConstraintsV1


def landing_geometry_seeds(model,state,limits,nominal):
    nominal=np.asarray(nominal,float);limits=np.asarray(limits,float)
    if nominal.shape!=(18,) or limits.shape!=(18,2):raise ValueError('Original full18 posture required')
    foot_ids=[model.names.index(n) for n in model.feet]
    qnom=state.q.copy();qnom[model.q_indices]=nominal
    baseline=model.snapshot(qnom,np.zeros(24))
    feet=baseline['body_positions_world'][foot_ids].copy()
    com=baseline['com_world'];actual=model.snapshot(state.q,state.v)
    velocity=actual['com_velocity_world']
    # Ballistic arrival estimate is only a seed timing, not a contact promise.
    lowest=max(.01,float(feet[:,2].min())-.022)
    vz=float(velocity[2]);arrival=(vz+np.sqrt(vz*vz+2*9.81*lowest))/9.81
    arrival=float(np.clip(arrival,.04,.5))
    nominal_height=max(.10,float(com[2]-feet[:,2].mean()))
    capture=np.sqrt(nominal_height/9.81)*velocity[:2]
    bounds=limits[:12].copy();hip=HipMountPlanningConstraintsV1(model)
    for joint,bound,lower in hip.constraints:
        if lower:bounds[joint,0]=max(bounds[joint,0],bound)
        else:bounds[joint,1]=min(bounds[joint,1],bound)
    postures=[('nominal_leg_extension',nominal[:12].copy(),None)]
    for lowering in [.03,.07]:
        goal=feet.copy()
        goal[:,:2]+=capture
        goal[:,2]=feet[:,2].min()-lowering
        qwork=qnom.copy()
        def residual(joints):
            qwork[model.q_indices[:12]]=joints
            positions=np.asarray(model.poses(qwork)[0])[foot_ids]
            return np.r_[(positions-goal).ravel(),.02*(joints-nominal[:12])]
        fit=least_squares(residual,np.clip(nominal[:12],bounds[:,0]+1e-8,bounds[:,1]-1e-8),
            bounds=(bounds[:,0],bounds[:,1]),max_nfev=60,ftol=1e-8,xtol=1e-8,gtol=1e-8)
        postures.append(('geometric_foot_plane_'+str(lowering),fit.x.copy(),
            dict(ik_converged=bool(fit.success),maximum_foot_coordinate_error_m=float(np.max(abs(residual(fit.x)[:12]))))))
    times=np.array([.08,.18,.30,.45,.60]);output=[]
    for name,leg_goal,ik in postures:
        for arm_mode in ['brake_at_release','return_toward_nominal']:
            targets=np.tile(state.joint_position,(5,1))
            extension=np.clip(times/max(.08,.6*arrival),0,1)
            targets[:,:12]=(1-extension[:,None])*state.joint_position[:12]+extension[:,None]*leg_goal
            # After estimated touchdown move toward compliant nominal posture;
            # actual task still requires contact feedback, tested separately.
            buffer=np.clip((times-arrival)/.20,0,1)
            targets[:,:12]=(1-buffer[:,None])*targets[:,:12]+buffer[:,None]*nominal[:12]
            if arm_mode=='return_toward_nominal':
                blend=1.-np.exp(-times/.35)
                targets[:,12:]=(1-blend[:,None])*state.joint_position[12:]+blend[:,None]*nominal[12:]
            targets=np.clip(targets,limits[:,0],limits[:,1])
            nodes=2*(targets-limits[:,0])/np.diff(limits).ravel()-1.
            output.append((name+'_'+arm_mode,nodes,dict(ballistic_arrival_s=arrival,ik=ik)))
    return output
