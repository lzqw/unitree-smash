"""Offline full-prefix leg correction scored at the contact event.

One local 12-parameter solve. Original six arm commands stay fixed. No state
teleportation, contact-force injection, stochastic search or native execution.
"""
import argparse,copy,gzip,hashlib,json,time
from pathlib import Path
import numpy as np
import pinocchio as pin
from scipy.optimize import lsq_linear, minimize, LinearConstraint, Bounds
from scripts.full_prefix_tracking_v1 import ARM_TRACKING_LIMIT_RAD
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1,PublicState
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance
from scripts.experimental_sphere_admission_v1 import ExperimentalSphereAdmissionV1
from scripts.contact_first_fullbody_evaluator_v1 import ContactFirstFullBodyEvaluatorV1
from scripts.audit_prediction_contact_geometry_v2 import racket_contact_state
from quadsmash.control.push_off_window_v1 import PushOffWindowV1
from quadsmash.control.impact_outcome_evaluator_v1 import ImpactOutcomeEvaluator
from quadsmash.physics.full_racket_v4 import load_racket


def refine_terminal(evaluate,*,radius_rad=.03,epsilon_rad=.001,amplitude_bounds=None,parameter_radii=None,linear_constraints=None,backtracking=False,protect_terminal_tasks=False,path_response_fn=None,path_response_reserve=0.):
    if not np.isfinite(path_response_reserve) or path_response_reserve<0 or (path_response_reserve and path_response_fn is None):
        raise ValueError('Nonnegative reserve requires a path response')
    if not 0<epsilon_rad<radius_rad<=.03:raise ValueError('Bounded offline step required')
    radii=np.full(12,radius_rad) if parameter_radii is None else np.asarray(parameter_radii,float).copy()
    if radii.ndim!=1 or not 1<=len(radii)<=24 or not np.isfinite(radii).all() or np.any(radii<=epsilon_rad) or np.any(radii>.1):
        raise ValueError('Explicit bounded parameter radii required')
    count=len(radii)
    bounds=np.c_[-radii,radii] if amplitude_bounds is None else np.asarray(amplitude_bounds,float).copy()
    if bounds.shape!=(count,2) or not np.isfinite(bounds).all() or np.any(bounds[:,0]>0) or np.any(bounds[:,1]<0):
        raise ValueError('Finite amplitude bounds must include unchanged reference')
    bounds[:,0]=np.maximum(bounds[:,0],-radii);bounds[:,1]=np.minimum(bounds[:,1],radii)
    trials=[]
    def call(amplitude,label):
        started=time.perf_counter()
        try:r=evaluate(amplitude.copy())
        except (ValueError,np.linalg.LinAlgError) as error:
            r=dict(path_valid=False,rejection=str(error))
        residual=np.asarray(r.get('normalized_residual',[]),float)
        valid=r.get('path_valid') is True and residual.shape==(9,) and np.isfinite(residual).all()
        row=dict(label=label,amplitude_rad=amplitude.copy(),result=r,seconds=time.perf_counter()-started,
                 valid=bool(valid),cost=float(residual@residual) if valid else None)
        trials.append(row)
        return row
    base=call(np.zeros(count),'baseline')
    out=dict(improved=False,selected_index=0,trials=trials,radius_rad=radius_rad,
        epsilon_rad=epsilon_rad,parameter_radii_rad=radii,amplitude_bounds=bounds,native_execution=False,valid_teacher_reference=False)
    if not base['valid']:out['rejection']='BASELINE_PATH_UNVERIFIED';return out
    def read_path(result):
        values,lower,upper=(np.asarray(x,float).copy() for x in path_response_fn(result))
        if (values.ndim!=1 or not len(values) or lower.shape!=values.shape or upper.shape!=values.shape
                or not all(np.isfinite(x).all() for x in (values,lower,upper)) or np.any(lower>upper)):
            raise ValueError('Finite consistent full-path response bounds required')
        return values,lower,upper
    if path_response_fn is not None:
        path_base,path_low,path_high=read_path(base['result'])
        if np.any(path_base<path_low) or np.any(path_base>path_high):
            out['rejection']='BASELINE_PATH_RESPONSE_OUTSIDE_BOUNDS';return out
        # Tighten only the local planning model, preserving zero-step
        # feasibility even when an existing verified path is near a bound.
        local_low=path_low+np.minimum(path_response_reserve,.5*(path_base-path_low))
        local_high=path_high-np.minimum(path_response_reserve,.5*(path_high-path_base))
        path_jac=np.zeros((len(path_base),count))
        out.update(path_response_reserve=path_response_reserve,path_local_lower=local_low,path_local_upper=local_high)
        out.update(path_response_baseline=path_base,path_response_lower=path_low,path_response_upper=path_high)
    y=np.array(base['result']['normalized_residual']);J=np.zeros((9,count));usable=[]
    for j in range(count):
        room=bounds[j,1] if bounds[j,1]>=-bounds[j,0] else bounds[j,0]
        step=np.sign(room)*min(epsilon_rad,abs(room))
        if abs(step)<1e-8:continue
        a=np.zeros(count);a[j]=step;r=call(a,'derivative_'+str(j))
        if r['valid'] and path_response_fn is not None:
            try:
                values,lower,upper=read_path(r['result'])
                if values.shape!=path_base.shape or not np.array_equal(lower,path_low) or not np.array_equal(upper,path_high):
                    raise ValueError('Path response shape or physical bounds changed')
                path_jac[:,j]=(values-path_base)/step
            except ValueError as error:
                r['valid']=False;r['path_response_rejection']=str(error)
        if r['valid']:
            J[:,j]=(np.asarray(r['result']['normalized_residual'])-y)/step;usable.append(j)
    out['usable_columns']=usable
    if not usable:out['rejection']='NO_VALID_LOCAL_RESPONSE';return out
    active=np.array(usable)
    A=np.vstack((J[:,active]*radii[active],.1*np.eye(len(active))))
    b=np.r_[-y,np.zeros(len(active))]
    low,high=bounds[active,0]/radii[active],bounds[active,1]/radii[active]
    constraints=[]
    if linear_constraints is None and not protect_terminal_tasks and path_response_fn is None:
        sol=lsq_linear(A,b,bounds=(low,high),tol=1e-10,max_iter=100)
    else:
        if linear_constraints is not None:
            C,clow,chigh=(np.asarray(x,float) for x in linear_constraints)
            if C.ndim!=2 or C.shape[1]!=count or clow.shape!=(len(C),) or chigh.shape!=clow.shape or not all(np.isfinite(x).all() for x in (C,clow,chigh)) or np.any(clow>chigh):
                raise ValueError('Finite original combined parameter constraints required')
            constraints.append(LinearConstraint(C[:,active]*radii[active],clow,chigh))
            out['linear_constraint_rows']=len(C)
        if path_response_fn is not None:
            constraints.append(LinearConstraint(path_jac[:,active]*radii[active],local_low-path_base,local_high-path_base))
            out['path_response_constraint_rows']=len(path_base)
        if protect_terminal_tasks:
            # Position, normal and velocity are separate tasks. The local
            # model may reduce their total cost only inside all three bounds.
            D=J[:,active]*radii[active]
            baseline_squared=np.sum(y.reshape(3,3)**2,axis=1)
            def task_margins(z):
                residual=(y+D@z).reshape(3,3)
                return baseline_squared-np.sum(residual**2,axis=1)
            def task_jacobian(z):
                residual=(y+D@z).reshape(3,3)
                return -2*np.einsum('gi,gij->gj',residual,D.reshape(3,3,-1))
            constraints.append(dict(type='ineq',fun=task_margins,jac=task_jacobian))
            out['protected_task_norms']=np.sqrt(baseline_squared)
            out['task_nonregression_tolerance_normalized']=1e-9
        sol=minimize(lambda z:.5*np.sum((A@z-b)**2),np.zeros(len(active)),
            jac=lambda z:A.T@(A@z-b),bounds=Bounds(low,high),
            constraints=constraints,method='SLSQP',
            options=dict(maxiter=100,ftol=1e-10))
    if not sol.success or not np.isfinite(sol.x).all():out['rejection']='LOCAL_SOLVE_FAILED';return out
    a=np.zeros(count);a[active]=radii[active]*sol.x;out['predicted_residual']=y+J@a
    if path_response_fn is not None:out['predicted_path_response']=path_base+path_jac@a
    # Each scaled sequence starts from the original state via evaluate(). A
    # feasible prefix or interpolated endpoint is never an execution certificate.
    scales=(1.,.5,.25) if backtracking else (1.,)
    out['backtracking_scales']=scales
    out['rejection']='NO_VERIFIED_TERMINAL_IMPROVEMENT'
    for scale in scales:
        final=call(a*scale,'proposal' if scale==1. else 'backtrack_'+str(scale))
        if final['valid'] and path_response_fn is not None:
            try:
                values,lower,upper=read_path(final['result'])
                final['path_response_valid']=bool(values.shape==path_base.shape and np.array_equal(lower,path_low)
                    and np.array_equal(upper,path_high) and np.all(values>=path_low) and np.all(values<=path_high))
                final['actual_path_response']=values
                final['predicted_path_response']=path_base+path_jac@(a*scale)
                final['valid']=final['path_response_valid']
            except ValueError as error:
                final['valid']=False;final['path_response_rejection']=str(error)
        nonregression=True
        if protect_terminal_tasks and final['valid']:
            actual=np.linalg.norm(np.asarray(final['result']['normalized_residual']).reshape(3,3),axis=1)
            nonregression=bool(np.all(actual<=out['protected_task_norms']+1e-9))
            final['terminal_task_norms']=actual
            final['terminal_task_nonregression']=nonregression
        if final['valid'] and nonregression and final['cost']<base['cost']-1e-10:
            out.update(improved=True,selected_index=len(trials)-1,rejection=None,accepted_step_scale=scale)
            break
    return out


class TerminalPushRollout:
    def __init__(self,model,interface,predictor,state,commands,arm_reference,metadata):
        self.model=model;self.interface=interface;self.predictor=predictor
        self.initial=copy.deepcopy(state);self.commands=np.asarray(commands,float).copy()
        self.arm_reference=np.asarray(arm_reference,float).copy();self.metadata=copy.deepcopy(metadata)
        n=metadata['contact_index']
        if (self.commands.shape!=(n,18) or self.arm_reference.ndim!=2 or self.arm_reference.shape[1]!=6
            or len(self.arm_reference)<n+1 or not np.isfinite(self.arm_reference).all()
            or not np.isfinite(self.commands).all() or n<2):raise ValueError('Complete fixed prefix required')
        self.arm_reference=self.arm_reference[:n+1].copy()

    def amplitude_bounds(self):
        # Conservative bounds cover every possibly supported node. No use of
        # future measured contacts and no command clipping after optimization.
        weights=np.sin(np.pi*(np.arange(len(self.commands)-1)+1)/len(self.commands))
        low=np.max((self.interface.limits[:12,0]-self.commands[:-1,:12])/weights[:,None],axis=0)
        high=np.min((self.interface.limits[:12,1]-self.commands[:-1,:12])/weights[:,None],axis=0)
        # The existing QP has numerical command residual tolerance. Retain its
        # unchanged reference at zero; never enlarge either correction side.
        return np.c_[np.minimum(low,0.),np.maximum(high,0.)]

    def validate_amplitude(self,amplitude):
        if amplitude.shape!=(12,) or not np.isfinite(amplitude).all() or np.max(abs(amplitude))>.0300000001:
            raise ValueError('Bounded twelve-leg amplitude required')

    def command_at_node(self,amplitude,reference,node,weight):
        command=reference.copy();command[:12]+=weight*amplitude
        return command,self.arm_reference[node+1].copy()

    def __call__(self,amplitude):
        amplitude=np.asarray(amplitude,float)
        self.validate_amplitude(amplitude)
        model=self.model;i=self.interface;m=self.metadata;state=copy.deepcopy(self.initial)
        R=self.initial.root_rotation_world;origin=self.initial.q[:3]
        window=PushOffWindowV1(len(self.commands));trace=[];failure=None
        for node,reference in enumerate(self.commands):
            decision=window.step(node,state.contacts)
            # Deterministic temporal basis spans the declared pre-contact
            # horizon, but correction ends as soon as predicted support ends.
            weight=np.sin(np.pi*(node+1)/len(self.commands)) if decision['enabled'] else 0.
            command,arm_reference=self.command_at_node(amplitude,reference,node,weight)
            if np.any(command<i.limits[:,0]-1e-6) or np.any(command>i.limits[:,1]+1e-6):
                failure=dict(node=node,reason='ORIGINAL_COMMAND_LIMIT');break
            try:
                step=self.predictor(model,i,copy.deepcopy(state),command,duration_s=.01,substeps=160,enforce_joint_stops=True)
                d=step['diagnostics'];bad=ContactFirstFullBodyEvaluatorV1._violations(d)
                if d['nonfoot_ground_violation_m']>1e-6:bad['nonfoot_ground_violation_m']=d['nonfoot_ground_violation_m']
                if d['nonfoot_contact_substeps']>0:bad['nonfoot_contact_substeps']=d['nonfoot_contact_substeps']
                tracking=float(np.max(abs(step['state'].joint_position[12:]-arm_reference)))
                if tracking>ARM_TRACKING_LIMIT_RAD:bad['arm_tracking_error_rad']=tracking
                roundtrip=i.limits[:,0]+.5*(i.normalized_action(command)+1)*np.diff(i.limits,axis=1)[:,0]
                err=float(np.max(abs(roundtrip-command)))
                if err>2e-6:bad['command_roundtrip_rad']=err
                if bad:failure=dict(node=node,reason='PATH_CONSTRAINT',violations=bad);break
                state=step['state']
                trace.append(dict(node=node,q=state.q.copy(),v=state.v.copy(),contacts=state.contacts.copy(),
                    final_target_rad=command,planned_arm_reference_rad=arm_reference.copy(),weight=weight,window=decision,diagnostics=d,
                    command_error_rad=err,arm_tracking_error_rad=tracking))
            except (ValueError,np.linalg.LinAlgError) as error:
                failure=dict(node=node,reason='MODEL_UNCERTAIN',error=str(error));break
        valid=failure is None and len(trace)==len(self.commands)
        result=dict(path_valid=bool(valid),failure=failure,trajectory=trace,steps=len(trace),
            native_execution=False,valid_teacher_reference=False,FULLBODY_PREDICTION_VALID=False)
        if not valid:return result
        p,n,v=racket_contact_state(model,state.q,state.v,np.asarray(m['contact_point_local'])+np.array([m['task_physics']['collision_radius_m'],0,0]))
        ep=R.T@(p-origin)-m['target_position_world_m'];en=R.T@n-m['target_normal_world'];ev=R.T@v-m['retargeted_velocity_current_frame_m_s']
        errors=dict(position_m=float(np.linalg.norm(ep)),normal=float(np.linalg.norm(en)),velocity_m_s=float(np.linalg.norm(ev)))
        # Optimizer scales are original coarse position and terminal normal /
        # speed scales. Final 3 mm admission stays independent and unchanged.
        result.update(normalized_residual=np.r_[ep/.035,en/.12,ev/.75],terminal_errors=errors)
        terminal=errors['position_m']<=.003 and errors['normal']<=.12 and errors['velocity_m_s']<=.75
        result['terminal_contact_passed']=bool(terminal)
        result['final_impact']=None
        if terminal:
            ball=origin+R@np.asarray(m['target_position_world_m']);incoming=R@np.asarray(m['incoming_velocity_planning'])
            impact=ImpactOutcomeEvaluator(m['task_physics'],m['task_court'],load_racket()).evaluate(
                ball,incoming,p-m['task_physics']['collision_radius_m']*n,v,n,m['contact_point_local'])
            result['final_impact']=impact
            result['FULLBODY_PREDICTION_VALID']=all(impact.get(k) is True for k in
                ('contact_feasible','legal_return_predicted','attack5_predicted','target_landing_predicted'))
        return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve attempts')
    root=Path(__file__).resolve().parents[1]
    seedpath=root/'reports/push_off_task_20261118/DIAGNOSED.json'
    commandpath=root/'reports/momentum_push_20261118/RESULTS.json'
    replaypath=root/'reports/experimental_fullbody_gate_20261118/REPLAY.json.gz'
    initial=json.loads(seedpath.read_text())['rows'][0]
    previous=json.loads(commandpath.read_text())['results'][0]
    calls=json.load(gzip.open(replaypath,'rt'))['physical_calls']
    matching=[c for c in calls if c['lane']==previous['lane']]
    if len(matching)!=1:raise ValueError('Unique diagnostic source event required')
    source=matching[0];model=LoadedPinocchioModelV1();q=np.array(initial['initial_q']);v=np.array(initial['initial_v'])
    state=PublicState(q.copy(),v.copy(),pin.Quaternion(q[3:7]).matrix(),q[:3].copy(),q[model.q_indices].copy(),v[model.v_indices].copy(),np.array(initial['baseline_initial_contacts'],bool),0.)
    state.previous_command_target=np.array(initial['previous_final_command'])
    interface=PositionPDInterface(**json.loads((root/'reports/codex_teacher_joint_eval_20261116/controller.json').read_text())['interface'])
    geometry=AuthoredGroundClearance(root/'configs/go2_piper_authored_ground_geometry_v1.json',model,margin_m=0.)
    if geometry.minimum_witness(state.q)['gap_m'] < -1e-6:raise ValueError('Initial nonfoot geometry violation')
    predictor=ExperimentalSphereAdmissionV1(model,geometry,root/'reports/prediction_model_validation_20261118/AUTHORED_SPHERES.json')
    evaluator=TerminalPushRollout(model,interface,predictor,state,[s['final_target_rad'] for s in previous['trajectory']],source['arm_sequence_rad'],source['metadata'])
    count=0
    def evaluate(amplitude):
        nonlocal count
        result=evaluator(amplitude);count+=1
        if count==1:
            if not result['path_valid']:raise ValueError('Baseline replay failed')
            for current,old in zip(result['trajectory'],previous['trajectory']):
                if not np.array_equal(current['q'],old['q']) or not np.array_equal(current['v'],old['v']):
                    raise ValueError('Frozen baseline replay drift')
            result['baseline_exact_replay']=True
        print(json.dumps(dict(call=count,path_valid=result['path_valid'],errors=result.get('terminal_errors'),failure=result.get('failure'))),flush=True)
        return result
    started=time.perf_counter();result=refine_terminal(evaluate,amplitude_bounds=evaluator.amplitude_bounds())
    result.update(seconds=time.perf_counter()-started,input_sha256={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (seedpath,commandpath,replaypath)})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else float(x))+'\n')
    selected=result['trials'][result['selected_index']]
    print(json.dumps(dict(improved=result['improved'],selected=result['selected_index'],errors=selected['result'].get('terminal_errors'),seconds=result['seconds'])))
if __name__=='__main__':main()
