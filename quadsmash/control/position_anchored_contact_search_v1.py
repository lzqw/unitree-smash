"""Lexicographic contact IK: position anchor cannot be traded for face error."""
import time
import numpy as np
from scipy.optimize import least_squares,lsq_linear


def position_anchor(kinematics,seeds,lower,upper,target,max_nfev=48):
    started=time.perf_counter();target=np.asarray(target,dtype=float)
    best=[None,float('inf')]
    def residual(q):
        p=kinematics(q)[0];e=float(np.linalg.norm(p-target))
        if e<best[1]:best[:]=[q.copy(),e]
        return p-target
    # Existing measured/optimized starts only; no random restart expansion.
    starts=[np.clip(np.asarray(q,dtype=float),lower,upper) for q in seeds]
    start=min(starts,key=lambda q:float(np.linalg.norm(residual(q))))
    # Preserve the historical bounded position-only DLS route, not just a
    # trust-region solve that can stall from the measured wrist configuration.
    legacy=start.copy()
    for k in range(15):
        point,_,jp,_=kinematics(legacy)
        step=jp.T@np.linalg.solve(jp@jp.T+.04**2*np.eye(3),target-point)
        cap=.25 if k<10 else .12
        legacy=np.clip(legacy+np.clip(step,-cap,cap),lower,upper)
        residual(legacy)
    start=best[0].copy()
    fit=least_squares(residual,start,jac=lambda q:kinematics(q)[2],bounds=(lower,upper),
                      max_nfev=max_nfev,ftol=1e-10,xtol=1e-10,gtol=1e-10)
    residual(fit.x);q=best[0];p,n,_,_=kinematics(q)
    return dict(q_anchor=q.copy(),position_anchor=p.copy(),normal_anchor=n.copy(),
                position_error_anchor_m=best[1],solver_success=bool(fit.success),nfev=int(fit.nfev),
                position_seconds=time.perf_counter()-started)


def anchored_pose_velocity(kinematics,seeds,lower,upper,speed,position,normal,velocity,
                           base_velocity_fn=None,normal_steps=12,normal_tolerance=.12,velocity_tolerance=.75,velocity_aware=False):
    anchor=position_anchor(kinematics,seeds,lower,upper,position)
    q=anchor['q_anchor'].copy();requested=np.array(normal,dtype=float,copy=True);requested/=np.linalg.norm(requested)
    # A precise anchor gets at most ten micrometres of numerical slack, never
    # the full 35-mm coarse envelope. Coarse anchors cannot be made worse.
    bound=min(.003,anchor['position_error_anchor_m']+1e-5) if anchor['position_error_anchor_m']<=.003 else anchor['position_error_anchor_m']+1e-9
    started=time.perf_counter();trace=[]
    for _ in range(normal_steps):
        p,n,jp,jn=kinematics(q);old=float(np.linalg.norm(n-requested))
        if old<=normal_tolerance:break
        N=np.eye(6)-np.linalg.pinv(jp,rcond=1e-8)@jp
        task=jn@N;delta=N@task.T@np.linalg.solve(task@task.T+.04**2*np.eye(3),requested-n)
        delta*=min(1.,.12/max(np.max(np.abs(delta)),1e-12))
        accepted=False
        for scale in (1.,.5,.25,.125,.0625):
            trial=np.clip(q+scale*delta,lower,upper)
            for correction in range(3):
                tp,_,tj,_=kinematics(trial)
                step=np.linalg.pinv(tj,rcond=1e-8)@(np.asarray(position)-tp)
                trial=np.clip(trial+np.clip(step,-.05,.05),lower,upper)
            tp,tn,_,_=kinematics(trial);ep=float(np.linalg.norm(tp-position));en=float(np.linalg.norm(tn-requested))
            ok=bool(ep<=bound and en<old-1e-8)
            trace.append(dict(position_error_m=ep,normal_vector_error=en,accepted=ok))
            if ok:q=trial;accepted=True;break
        if not accepted:break
    p,n,jp,jn=kinematics(q);ep=float(np.linalg.norm(p-position));en=float(np.linalg.norm(n-requested))
    if ep>bound:
        q=anchor['q_anchor'].copy();p,n,jp,jn=kinematics(q)
        ep=anchor['position_error_anchor_m'];en=float(np.linalg.norm(n-requested))
    refinement=dict(attempted=False,accepted=False,mode='strict_position_first')
    # A 10-micrometre local nullspace budget can stall at joint limits even
    # though a sub-millimetre position-preserving solution remains possible.
    # Run a second, exact-FK constrained solve ONLY after strict refinement
    # fails. The original q_anchor remains unchanged and the native 3-mm
    # contact hard gate is never relaxed.
    if ep<=.003 and en>normal_tolerance and normal_steps>0:
        from .constrained_contact_normal_v1 import refine_contact_normal
        expanded_bound=min(.003,max(bound,.0005))
        refinement.update(attempted=True,mode='bounded_position_constrained_slsqp',
                          previous_normal_error=float(en),position_budget_m=expanded_bound)
        try:
            refined=refine_contact_normal(
                kinematics,anchor['q_anchor'],q,lower,upper,
                position,requested,expanded_bound,max_iterations=32)
            refinement['attempts']=refined['attempts']
            refinement['seconds']=refined['seconds']
            # Do not spend submillimetre contact precision for an
            # insignificant face improvement that still fails the unchanged
            # normal gate. Preserve the exact old anchor in that case.
            improvement=en-refined['normal_vector_error']
            useful=(refined['normal_vector_error']<=normal_tolerance
                    or improvement>=max(.02,.05*en))
            if (useful and improvement>1e-8
                    and refined['position_error_m'] <= expanded_bound+1e-10):
                q=refined['q_rad'].copy()
                p,n,jp,jn=kinematics(q)
                ep=float(np.linalg.norm(p-position))
                en=float(np.linalg.norm(n-requested))
                refinement['accepted']=True
                bound=expanded_bound
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as error:
            # This is optional normal refinement, never a way to bypass
            # failure of original normal/position/physics gates.
            refinement['rejection']=type(error).__name__+': '+str(error)
    normal_seconds=time.perf_counter()-started
    position_ok=ep<=.003;normal_ok=position_ok and en<=normal_tolerance
    qd=np.zeros(6);base=np.zeros(3) if base_velocity_fn is None else base_velocity_fn(p)
    started=time.perf_counter();rate_success=False
    if normal_ok:
        rate=lsq_linear(np.vstack((jp,.001*np.eye(6))),np.r_[np.asarray(velocity)-base,np.zeros(6)],bounds=(-np.asarray(speed),np.asarray(speed)),max_iter=50)
        qd=rate.x;rate_success=bool(rate.success)
    actual=base+jp@qd;ev=float(np.linalg.norm(actual-velocity))
    velocity_refinement=dict(attempted=False)
    from .velocity_aware_contact_pose_v1 import velocity_support
    support=velocity_support(jp,base,velocity,speed) if normal_ok else None
    if velocity_aware and normal_ok and rate_success and ev>velocity_tolerance and ep<=.0005:
        from .velocity_aware_contact_pose_v1 import refine_velocity_pose
        velocity_refinement['attempted']=True
        try:
            refined=refine_velocity_pose(kinematics,q,qd,lower,upper,speed,position,requested,velocity,
                base_velocity_fn,normal_tolerance=normal_tolerance)
            velocity_refinement.update(refined)
            if refined['improved']:
                q,qd=refined['q_rad'],refined['qd_rad_s']
                p,n,actual=refined['position'],refined['normal'],refined['velocity']
                ep,en,ev=refined['position_error_m'],refined['normal_error'],refined['velocity_error_m_s']
                support=refined['velocity_support'];bound=max(bound,.0005)
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as error:
            velocity_refinement['rejection']=str(error)
    return dict(**anchor,velocity_pose_refinement=velocity_refinement,velocity_support=support,
        saturated_velocity_joints=np.flatnonzero(abs(qd)>=np.asarray(speed)-1e-5).tolist(),q_rad=q,qd_rad_s=qd,position=p,normal=n,velocity=actual,
        velocity_solver_executed=normal_ok,velocity_error_status='SOLVED' if normal_ok else 'NOT_SOLVED',
        requested_normal=requested.copy(),requested_velocity=np.asarray(velocity).copy(),
        anchor_normal_error=float(np.linalg.norm(anchor['normal_anchor']-requested)),
        position_error_m=ep,normal_error=en,requested_velocity_error_m_s=ev,
        POSITION_ANCHORED=position_ok,NORMAL_FEASIBLE=normal_ok,VELOCITY_FEASIBLE=bool(normal_ok and rate_success and ev<=velocity_tolerance),
        ANALYTIC_RETURN_VALID=False,rejection=None if normal_ok else 'POSE_TASK_INCOMPATIBLE',
        position_preservation_bound_m=bound,normal_trace=trace,normal_seconds=normal_seconds,
        constrained_normal_refinement=refinement,
        velocity_seconds=time.perf_counter()-started)


def plan_anchored_events(planner,q0,qd0,lower,upper,speed,requested,target_at,base_at,
                         target_metadata,kinematics,kinematics_at_time,evaluator,impact):
    from collections import Counter
    from .contact_first_trajectory_search_v1 import ContactFirstCandidate,_hermite
    from .inverse_impact_target_v1 import UnreachableImpactTarget
    cfg=planner.config;records=[];rejected=Counter();best=None
    if kinematics is None:raise ValueError('Position anchor requires exact FK')
    if evaluator is None or impact is None:raise ValueError('Anchored execution requires full-body and impact evaluators')
    for dt in cfg.contact_time_offsets_s:
        index=round((requested+dt)/cfg.dt_s);duration=index*cfg.dt_s
        if not 1<=index<cfg.horizon_steps:continue
        try:target,normal,velocity=target_at(duration)
        except UnreachableImpactTarget as error:
            records.append(dict(contact_time_s=duration,rejection=str(error)));rejected['analytic_target']+=1;continue
        R,displacement,vbase,omega=base_at(duration)
        fk=(lambda q:kinematics_at_time(q,duration)) if kinematics_at_time is not None else kinematics
        def base_velocity(point):return R.T@(vbase+np.cross(omega,R@point))
        extra=target_metadata.get(round(duration,9),{})
        if cfg.task_constrained_hit and not extra['attack_geometry']['downward_attack_geometrically_possible']:
            reason=extra['attack_geometry']['rejection']
            records.append(dict(contact_time_s=duration,attack_geometry=extra['attack_geometry'],
                ATTACK_GEOMETRY_FEASIBLE=False,rejection=reason,position_solver_executed=False))
            rejected[reason]+=1;continue
        seeds=[q0.copy()]
        if 'optimized_q_rad' in extra:seeds.append(np.asarray(extra['optimized_q_rad']))
        result=anchored_pose_velocity(fk,seeds,lower,upper,speed,R.T@(target-displacement),R.T@normal,R.T@velocity,base_velocity,normal_tolerance=cfg.normal_tolerance,velocity_tolerance=cfg.velocity_tolerance_m_s,velocity_aware=cfg.velocity_aware_pose)
        record=dict(contact_time_s=duration,contact_point_local=extra.get('contact_point_local'),
            original_requested_racket_velocity_world_m_s=extra.get('original_requested_racket_velocity_world_m_s'),
            optimized_requested_racket_velocity_world_m_s=extra.get('optimized_requested_racket_velocity_world_m_s'),
            original_requested_velocity_error_m_s=(float(np.linalg.norm(R@result['velocity']-
                np.asarray(extra['original_requested_velocity_current_frame_m_s'])))
                if result['velocity_solver_executed'] and 'original_requested_velocity_current_frame_m_s' in extra else None),
            rotation_current_from_future=R.tolist(),**result)
        records.append(record)
        adapted_task_rate=False
        if cfg.task_constrained_hit:
            geometry=extra['attack_geometry']
            record.update(attack_geometry=geometry,ATTACK_GEOMETRY_FEASIBLE=geometry['downward_attack_geometrically_possible'],
                initial_ANALYTIC_GOAL_VALID=extra['analytic_goal_valid'])
            if not geometry['downward_attack_geometrically_possible']:
                record['rejection']=geometry['rejection'];rejected[geometry['rejection']]+=1;continue
            if result['position_error_anchor_m']>.0005:
                record['rejection']='POSITION_ANCHOR_OUTSIDE_FINE_BUDGET';rejected[record['rejection']]+=1;continue
            from .task_constrained_hit_search_v2 import optimize_task_hit
            def contact_fk(q,contact):return kinematics_at_time(q,duration,contact)
            trial=optimize_task_hit(contact_fk,result['q_rad'],result['qd_rad_s'],lower,upper,speed,
                R.T@(target-displacement),extra['contact_point_local'],extra['task_string_strip'],
                base_velocity,R,displacement,dict(extra,contact_time_s=duration),impact,
                extra['task_court'],extra['incoming_velocity_planning'])
            record['task_joint_search']=trial
            if not trial['accepted']:
                record['rejection']='TASK_JOINT_INFEASIBLE';rejected[record['rejection']]+=1;continue
            chosen=trial['selected'];adapted_task_rate=True
            for name in ('q_rad','qd_rad_s','position','normal','velocity','position_error_m','normal_error'):
                result[name]=chosen[name]
            # Keep initial requested normal/speed and immutable anchor in the
            # record; new target is explicitly task-derived, never renamed.
            normal=chosen['task_target_normal_current']
            extra=dict(extra,contact_point_local=chosen['contact_point_local'])
            result['NORMAL_FEASIBLE']=True
            result['requested_velocity_error_m_s']=float(np.linalg.norm(R@result['velocity']-velocity))
            record.update(initial_pose_normal_error=record['normal_error'],
                original_requested_normal_error=float(np.linalg.norm(result['normal']-result['requested_normal'])),
                original_ideal_velocity_feasible=record['VELOCITY_FEASIBLE'],
                q_rad=result['q_rad'],qd_rad_s=result['qd_rad_s'],position=result['position'],
                normal=result['normal'],velocity=result['velocity'],position_error_m=result['position_error_m'],
                normal_error=result['normal_error'],requested_velocity_error_m_s=result['requested_velocity_error_m_s'],
                rejection=None,TASK_FEASIBLE_RACKET_VELOCITY=True,ANALYTIC_GOAL_VALID=True,
                POSITION_FEASIBLE=True,NORMAL_FEASIBLE=True,
                task_target_normal_current=normal,task_contact_point_local=chosen['contact_point_local'])
        if not result['NORMAL_FEASIBLE']:
            rejected['POSE_TASK_INCOMPATIBLE']+=1;continue
        # A fixed desired racket speed is only one possible way to achieve
        # the original ball-side task. The optional rate search attempts a
        # DIFFERENT analytic task-valid impact inside the same ±5-rad/s box.
        # It cannot relabel the original requested-velocity error as zero,
        # and it must still pass the unchanged trajectory/18D/final impact
        # admission checks below.
        if (not result['VELOCITY_FEASIBLE']
                and not cfg.task_constrained_hit
                and cfg.impact_feasible_rate_retarget
                and result['NORMAL_FEASIBLE']
                and 'desired_outgoing_velocity_world_m_s' in extra
                and 'contact_point_local' in extra):
            from .impact_feasible_rate_v1 import optimize_impact_feasible_rate
            _,_,arm_jacobian,_=fk(result['q_rad'])
            meta_probe=dict(extra,contact_time_s=duration,
                achieved_position_current_frame_m=displacement+R@result['position'],
                achieved_normal_current_frame=R@result['normal'],
                achieved_velocity_current_frame_m_s=R@result['velocity'])
            trial=optimize_impact_feasible_rate(
                arm_jacobian,base_velocity(result['position']),
                result['qd_rad_s'],speed,R,meta_probe,impact,
                extra['desired_outgoing_velocity_world_m_s'])
            record['impact_feasible_rate_diagnostic']=trial
            if trial['accepted']:
                adapted_task_rate=True
                result['qd_rad_s']=trial['qd_rad_s'].copy()
                result['velocity']=R.T@trial['achieved_velocity_current_frame_m_s']
                result['requested_velocity_error_m_s']=float(np.linalg.norm(
                    result['velocity']-R.T@velocity))
                record.update(qd_rad_s=result['qd_rad_s'].copy(),
                    velocity=result['velocity'].copy(),
                    requested_velocity_error_m_s=result['requested_velocity_error_m_s'],
                    TASK_FEASIBLE_RATE=True,
                    ORIGINAL_REQUEST_VELOCITY_FEASIBLE=False)
                if 'original_requested_velocity_current_frame_m_s' in extra:
                    record['original_requested_velocity_error_m_s']=float(np.linalg.norm(
                        R@result['velocity']-
                        np.asarray(extra['original_requested_velocity_current_frame_m_s'])))
        if not (result['VELOCITY_FEASIBLE'] or adapted_task_rate):
            rejected['REQUESTED_VELOCITY_UNREACHABLE']+=1
            record['rejection']='REQUESTED_VELOCITY_UNREACHABLE'
            continue
        q1,qd1=result['q_rad'],result['qd_rad_s']
        q,qd=_hermite(q0,qd0,q1,qd1,duration,np.arange(index+1)*cfg.dt_s)
        if (np.any(q<lower-1e-9) or np.any(q>upper+1e-9) or np.any(np.abs(qd)>speed+1e-9) or
                np.any(np.abs(np.diff(q,axis=0))>speed*cfg.dt_s+1e-9)):
            rejected['trajectory_joint_limits']+=1;record['rejection']='trajectory_joint_limits';continue
        qfull=np.repeat(q1[None],cfg.horizon_steps,axis=0);dfull=np.zeros_like(qfull)
        qfull[:index+1]=q;dfull[:index+1]=qd
        metadata=dict(extra,method='position_anchored_contact_search_v1',contact_index=index,contact_time_s=duration,
            target_position_world_m=target,target_normal_world=normal,target_velocity_world_m_s=velocity,
            achieved_position_current_frame_m=displacement+R@result['position'],
            achieved_normal_current_frame=R@result['normal'],achieved_velocity_current_frame_m_s=R@result['velocity'],
            position_tolerance_m=.003,normal_tolerance=cfg.normal_tolerance,velocity_tolerance_m_s=cfg.velocity_tolerance_m_s,
            position_anchor=record,velocity_projection=False,
            task_feasible_velocity_retarget=adapted_task_rate,
            retargeted_velocity_current_frame_m_s=(
                R@result['velocity'] if adapted_task_rate else None),
            original_request_velocity_feasible=bool(result['VELOCITY_FEASIBLE']),
            velocity_error_from_requested_m_s=result['requested_velocity_error_m_s'])
        started=time.perf_counter();out=impact(metadata);record['impact_seconds']=time.perf_counter()-started
        record['impact_outcome']=out
        valid=lambda value:bool(value.get('legal_return_predicted') and value.get('attack5_predicted') and value.get('target_landing_predicted'))
        if not valid(out):
            rejected['ANALYTIC_RETURN_INVALID']+=1;record['rejection']='ANALYTIC_RETURN_INVALID';continue
        record['ANALYTIC_RETURN_VALID']=True
        started=time.perf_counter();physical=evaluator(qfull,dfull,index,metadata);record['dynamics_seconds']=time.perf_counter()-started
        record['physical_result']=physical
        if not isinstance(physical,dict) or not physical.get('accepted'):
            record['rejection']='FULLBODY_PREDICTION_REJECTED'
            rejected['dynamics']+=1;continue
        fields=('actual_racket_position_world_m','actual_racket_normal_world','actual_racket_velocity_world_m_s')
        if not all(k in physical for k in fields):rejected['missing_actual_contact_state']+=1;continue
        final=impact(dict(metadata,impact_robot_state=physical.get('impact_robot_state'),
            achieved_position_current_frame_m=physical[fields[0]],achieved_normal_current_frame=physical[fields[1]],
            achieved_velocity_current_frame_m_s=physical[fields[2]]))
        if not valid(final):rejected['final_impact']+=1;continue
        metadata.update(evaluator=physical,impact_outcome=final)
        if cfg.task_constrained_hit:
            from .task_constrained_impact_goal_v2 import task_margins
            final_task=task_margins(final,extra['task_court'])
            if not final_task['valid']:
                rejected['final_task_margins']+=1;continue
            score=-final_task['gamma']
        else:
            score=float(final['ranking_cost'])+result['requested_velocity_error_m_s']/cfg.velocity_tolerance_m_s
        candidate=ContactFirstCandidate(qfull,dfull,index,duration,score,result['position_error_m'],result['normal_error'],result['requested_velocity_error_m_s'],metadata)
        if best is None or candidate.score<best.score:best=candidate
    planner.last_diagnostics=dict(method='position_anchored_contact_search_v1',anchors=records,rejection_counts=dict(rejected),accepted_candidates=int(best is not None))
    if best is None:raise RuntimeError('No position-anchored candidate passed: '+str(dict(rejected)))
    return best
