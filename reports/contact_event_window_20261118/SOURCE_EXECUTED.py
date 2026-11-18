"""Causal bounded contact-window replay; no native run or reference promotion.

Extend a verified prefix with an explicit hold of its last actual 18D command.
The extension is fully predicted from the original state, never extrapolated
by moving the root or inserting ideal takeoff velocities.
"""
import argparse,copy,gzip,hashlib,json,time
from pathlib import Path
import numpy as np
import pinocchio as pin
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1,PublicState
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance
from scripts.experimental_sphere_admission_v1 import ExperimentalSphereAdmissionV1
from scripts.terminal_push_refinement_v1 import TerminalPushRollout
from scripts.contact_terminal_attribution_v1 import contact_kinematics
from quadsmash.control.position_anchored_contact_search_v1 import anchored_pose_velocity
from quadsmash.control.initial_intercept import observed_initial_ball,propagate_ball_state
from quadsmash.control.contact_reach_bound_v1 import string_strip_reach_bound
from quadsmash.control.full_racket_geometry_v4 import FullRacketGeometryV4
from quadsmash.control.task_constrained_impact_goal_v2 import optimize_task_goal,attack_geometry
from quadsmash.control.impact_outcome_evaluator_v1 import ImpactOutcomeEvaluator
from quadsmash.physics.full_racket_v4 import load_racket


def hold_extension(commands,reference,count):
    commands=np.asarray(commands,float);reference=np.asarray(reference,float)
    if commands.ndim!=2 or commands.shape[1]!=18 or len(commands)<1 or reference.shape!=(len(commands)+1,6):
        raise ValueError('Complete command/reference history required')
    if not np.isfinite(commands).all() or not np.isfinite(reference).all() or not isinstance(count,int) or count<len(commands) or count>len(commands)+4:
        raise ValueError('Finite history and at most four explicit hold ticks required')
    extra=count-len(commands)
    return np.vstack([commands,np.repeat(commands[-1:],extra,axis=0)]),np.vstack([reference,np.repeat(reference[-1:],extra,axis=0)])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prefix',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve previous evidence')
    r=json.loads(a.prefix.read_text())
    if hashlib.sha256(a.prefix.read_bytes()).hexdigest()!=a.prefix.with_suffix('.sha256').read_text().strip():raise ValueError('Prefix hash mismatch')
    for path,digest in r['input_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('Input drift')
    s=r['trials'][r['selected_index']]['result'];initial_path=Path('reports/push_off_task_20261118/DIAGNOSED.json')
    start=json.loads(initial_path.read_text())['rows'][0];lane=start['lane']
    original=Path('reports/experimental_fullbody_gate_20261118/REPLAY.json.gz');source=json.load(gzip.open(original,'rt'))
    matches=[c for c in source['physical_calls'] if c['lane']==lane]
    if len(matches)!=1:raise ValueError('Ambiguous source contact event')
    m=copy.deepcopy(matches[0]['metadata']);n=m['contact_index']
    if not s['path_valid'] or len(s['trajectory'])!=n:raise ValueError('Complete verified prefix required')
    # Obtain the original time candidates from the originating causal plan,
    # not a new timing scan or any success/failure labels.
    windows=[]
    for plan in source['plans']:
        clocks=dict(zip(plan['env_ids'],plan['observation_clock_s']))
        if lane not in clocks or abs(clocks[lane]-start['clock_s'])>1e-7:continue
        for summary in plan['contact_first_plan_summaries']:
            if summary['lane']==lane:windows.append([v['contact_time_s'] for v in summary['search_diagnostics']['anchors']])
    if len(windows)!=1:raise ValueError('Unique frozen contact window required')
    times=windows[0];counts=[round(t/.01) for t in times]
    if any(abs(t-k*.01)>1e-9 for t,k in zip(times,counts)):raise ValueError('Original control grid required')
    commands,reference=hold_extension([t['final_target_rad'] for t in s['trajectory']],r['selected_arm_reference_rad'],max(counts))
    observation_path=Path('reports/joint_loop_round_20261117/attempts/probe01/native/public_observations.npy')
    protected=json.loads(Path('reports/terminal_task_protection_20261118/MANIFEST.json').read_text())['protected_sha256']
    if hashlib.sha256(observation_path.read_bytes()).hexdigest()!=protected[str(observation_path)]:raise ValueError('Observation hash mismatch')
    observed=np.load(observation_path,mmap_mode='r');tick=round(start['clock_s']/.01)
    # Saved stream's index is verified by clock; no future observation is read.
    obs=np.asarray(observed[tick,lane]);assert abs(obs[775]-start['clock_s'])<1e-7
    p0,v0=observed_initial_ball(obs[None],[1.1,0,1.8],[1,0,0])
    model=LoadedPinocchioModelV1();q=np.array(start['initial_q']);v=np.array(start['initial_v']);R0=pin.Quaternion(q[3:7]).matrix()
    state=PublicState(q.copy(),v.copy(),R0,q[:3].copy(),q[model.q_indices].copy(),v[model.v_indices].copy(),np.array(start['baseline_initial_contacts'],bool),0.)
    state.previous_command_target=np.array(start['previous_final_command'])
    interface=PositionPDInterface(**json.loads(Path('reports/codex_teacher_joint_eval_20261116/controller.json').read_text())['interface'])
    geometry=AuthoredGroundClearance('configs/go2_piper_authored_ground_geometry_v1.json',model,margin_m=0.)
    predictor=ExperimentalSphereAdmissionV1(model,geometry,Path('reports/prediction_model_validation_20261118/AUTHORED_SPHERES.json'))
    impact=ImpactOutcomeEvaluator(m['task_physics'],m['task_court'],load_racket());offset=np.array(m['contact_point_local']);offset[0]+=m['task_physics']['collision_radius_m']
    ball,incoming=propagate_ball_state(p0[0],v0[0],max(times),m['task_physics'])
    # The final hold path is not a newly optimized arm reference. Do not reuse
    # an old legal-outgoing claim at the changed event time.
    m.update(contact_index=max(counts),contact_time_s=max(times),target_position_world_m=R0.T@(ball-q[:3]),incoming_velocity_planning=R0.T@incoming)
    started=time.perf_counter();path=TerminalPushRollout(model,interface,predictor,state,commands,reference,m)(np.zeros(12));rows=[]
    if len(path['trajectory'])<n:raise ValueError('Original verified prefix could not be reproduced')
    for old,new in zip(s['trajectory'],path['trajectory'][:n]):
        if not np.array_equal(old['q'],new['q']) or not np.array_equal(old['v'],new['v']):raise ValueError('Original prefix drift')
    for t,k in zip(times,counts):
        row=dict(time_s=t,node_count=k)
        if k>len(path['trajectory']):row.update(status='MODEL_PATH_UNVERIFIED',failure=path['failure']);rows.append(row);continue
        pred=path['trajectory'][k-1];pq=np.array(pred['q']);pv=np.array(pred['v']);R=pin.Quaternion(pq[3:7]).matrix()
        ball,incoming=propagate_ball_state(p0[0],v0[0],t,m['task_physics']);rp,rn,J,Jn=contact_kinematics(model,pq,offset)
        goal=optimize_task_goal(impact,ball,incoming,m['contact_point_local'])
        row.update(status='MODEL_PATH_ONLY',ball_world_m=ball,incoming_world_m_s=incoming,actual_contact_world_m=rp,
            position_error_m=float(np.linalg.norm(rp-ball)),reach_bound=string_strip_reach_bound(FullRacketGeometryV4(),R.T@(ball-pq[:3]),m['task_string_strip'],ball_radius_m=m['task_physics']['collision_radius_m']),
            attack_geometry=attack_geometry(ball,m['task_physics'],m['task_court']),analytic_goal=goal,
            actual_forward_outcome=impact.evaluate(ball,incoming,rp-m['task_physics']['collision_radius_m']*rn,J@pv,rn,m['contact_point_local']))
        if goal['accepted']:
            def kinematics(arm):
                test=pq.copy();test[model.q_indices[12:]]=arm
                point,normal,jp,jn=contact_kinematics(model,test,offset)
                return point,normal,jp[:,model.v_indices[12:]],jn[:,model.v_indices[12:]]
            def base_velocity(point):return R@pv[:3]+np.cross(R@pv[3:6],point-pq[:3])
            pose=anchored_pose_velocity(kinematics,[pq[model.q_indices[12:]],reference[k]],
                interface.limits[12:,0],interface.limits[12:,1],interface.velocity[12:],
                ball,np.array(goal['normal_world']),np.array(goal['racket_velocity_world_m_s']),base_velocity_fn=base_velocity)
            row['conditional_pose_velocity']=pose
            row['conditional_pose_is_executed']=False
            row['conditional_pose_requires_complete_revalidation']=True
        rows.append(row)
    out=dict(seconds=time.perf_counter()-started,rows=rows,path=path,hold_command_rad=commands[n:].copy(),arm_reference_rad=reference,
        prefix_sha256=hashlib.sha256(a.prefix.read_bytes()).hexdigest(),observations_sha256=hashlib.sha256(observation_path.read_bytes()).hexdigest(),
        input_sha256=r['input_sha256'],original_times_s=times,observed_tick=tick,original_prefix_exact=True,
        final_teacher_reference_valid=False,native_execution=False,future_native_states_read=False,
        changed_event_requires_new_arm_reference_and_full_revalidation=True)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else float(x))+'\n')
    print(json.dumps(dict(path_valid=path['path_valid'],failure=path['failure'],rows=[dict(time=r['time_s'],position=r.get('position_error_m'),deficit=r.get('reach_bound',{}).get('conditional_translation_lower_bound_m')) for r in rows])))
if __name__=='__main__':main()
