"""One bounded offline local command correction through the fine-step model.

No model-fit force or gain changes. Twelve independent leg commands are free;
the six prescribed arm commands remain exact. Finite-difference columns and the
one proposed command are independently integrated from the same real history.
"""
import copy
import numpy as np
from scipy.optimize import lsq_linear
from scripts.contact_first_fullbody_evaluator_v1 import ContactFirstFullBodyEvaluatorV1


def bounded_response_step(command, limits, desired, scales, evaluate, *, radius_rad=.03, epsilon_rad=.001):
    """evaluate returns a finite 6D response and original physical eligibility.

    A failed local derivative cannot grant eligibility. At most 14 model calls,
    one linear solve and one nonlinear proposal; there is no parameter scan.
    """
    command=np.asarray(command,float).copy();limits=np.asarray(limits,float)
    desired=np.asarray(desired,float);scales=np.asarray(scales,float)
    if (command.shape!=(18,) or limits.shape!=(18,2) or desired.shape!=(6,) or scales.shape!=(6,)
        or not all(np.isfinite(x).all() for x in (command,limits,desired,scales))
        or np.any(scales<=0) or np.any(limits[:,0]>=limits[:,1])
        or np.any(command<limits[:,0]-1e-6) or np.any(command>limits[:,1]+1e-6)
        or not 0<epsilon_rad<radius_rad<=.03):raise ValueError('Explicit finite bounded correction inputs required')
    evaluations=[]
    def call(u,kind):
        try:
            result=evaluate(u.copy())
            response=np.asarray(result['response'],float)
            valid=result.get('valid') is True and response.shape==(6,) and np.isfinite(response).all()
            cost=float(np.sum(((response-desired)/scales)**2)) if valid else None
            evaluations.append(dict(kind=kind,command=u.copy(),valid=bool(valid),cost=cost,
                rejection=result.get('rejection'),response=response.copy()))
            return result,response,cost
        except (ValueError,np.linalg.LinAlgError) as error:
            evaluations.append(dict(kind=kind,command=u.copy(),valid=False,rejection=str(error)))
            return None,None,None
    base,y,cost=call(command,'baseline')
    result=dict(accepted=False,command=command.copy(),evaluations=evaluations,
        radius_rad=radius_rad,epsilon_rad=epsilon_rad,desired=desired,scales=scales,
        fullbody_prediction_valid=False,native_execution=False)
    if cost is None:
        result['rejection']='BASELINE_PHYSICALLY_UNVERIFIED';return result
    result.update(cost_before=cost,cost_after=cost)
    J=np.zeros((6,12));usable=[]
    for j in range(12):
        step=min(epsilon_rad,limits[j,1]-command[j])
        if step<epsilon_rad*.5:step=-min(epsilon_rad,command[j]-limits[j,0])
        u=command.copy();u[j]+=step
        trial,yp,cp=call(u,'derivative_'+str(j))
        if cp is not None:
            J[:,j]=(yp-y)/step/scales;usable.append(j)
    if not usable:
        result['rejection']='NO_VALID_LOCAL_RESPONSE';return result
    # Ridge is dimensionless and favors small changes. Hard physical gates are
    # evaluated separately, never represented only by a penalty.
    low=np.maximum(-1.,(limits[:12,0]-command[:12])/radius_rad)
    high=np.minimum(1.,(limits[:12,1]-command[:12])/radius_rad)
    solution=lsq_linear(np.vstack((J*radius_rad,.1*np.eye(12))),
        np.r_[(desired-y)/scales,np.zeros(12)],bounds=(low,high),max_iter=100,tol=1e-10)
    result.update(usable_columns=usable,linear_solver_success=bool(solution.success))
    if not solution.success or not np.isfinite(solution.x).all():
        result['rejection']='LOCAL_SOLVE_FAILED';return result
    proposed=command.copy();proposed[:12]+=radius_rad*solution.x
    result['linear_predicted_response']=y+scales*(J@(proposed[:12]-command[:12]))
    trial,yp,cp=call(proposed,'proposal')
    if cp is not None and cp<cost-1e-10:
        result.update(accepted=True,command=proposed,cost_after=cp,rejection=None)
    else:result['rejection']='NO_VERIFIED_RESPONSE_IMPROVEMENT'
    return result


def correct_push_command(model,interface,predictor,state,command,desired_com_acceleration,
                         desired_angular_momentum_rate,arm_reference):
    before=model.snapshot(state.q,state.v)
    # 1 g and weight * 0.1 m normalize acceleration and moment task residuals.
    scales=np.r_[np.full(3,9.81),np.full(3,model.mass*9.81*.1)]
    def evaluate(target):
        prediction=predictor(model,interface,copy.deepcopy(state),target,
            duration_s=.01,substeps=160,enforce_joint_stops=True)
        d=prediction['diagnostics'];bad=ContactFirstFullBodyEvaluatorV1._violations(d)
        for key,limit in [('nonfoot_ground_violation_m',1e-6),('nonfoot_contact_substeps',0)]:
            if not np.isfinite(d[key]) or d[key]>limit:bad[key]=d[key]
        following=prediction['state'];after=model.snapshot(following.q,following.v)
        tracking=float(np.max(abs(following.joint_position[12:]-arm_reference)))
        if tracking>.25:bad['arm_tracking_error_rad']=tracking
        roundtrip=interface.limits[:,0]+.5*(interface.normalized_action(target)+1)*np.diff(interface.limits,axis=1)[:,0]
        error=float(np.max(abs(roundtrip-target)))
        if error>2e-6:bad['command_roundtrip_rad']=error
        response=np.r_[(after['com_velocity_world']-before['com_velocity_world'])/.01,
            (after['momentum_world'][3:]-before['momentum_world'][3:])/.01]
        return dict(response=response,valid=not bad,rejection=bad or None)
    return bounded_response_step(command,interface.limits,
        np.r_[desired_com_acceleration,desired_angular_momentum_rate],scales,evaluate)
