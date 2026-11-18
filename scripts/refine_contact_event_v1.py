"""One bounded coupled refinement at an original frozen contact event.

A causal hold-tail diagnostic is a seed, never a qualified reference. Every
local response and final proposal is replayed from original q/v/history.
"""
import argparse,copy,gzip,hashlib,json,time
from pathlib import Path
import numpy as np
import pinocchio as pin
from scripts.full_prefix_tracking_v1 import arm_tracking_response,tracking_linearization_reserve
from scripts.coupled_contact_prefix_v1 import CoupledContactRollout
from scripts.terminal_push_refinement_v1 import refine_terminal
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1,PublicState
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance
from scripts.experimental_sphere_admission_v1 import ExperimentalSphereAdmissionV1
from quadsmash.control.initial_intercept import observed_initial_ball,propagate_ball_state
from quadsmash.control.impact_outcome_evaluator_v1 import ImpactOutcomeEvaluator
from quadsmash.control.task_constrained_impact_goal_v2 import task_margins
from quadsmash.physics.full_racket_v4 import load_racket


def retimed_metadata(original,row,initial_q,impact):
    """Recheck the actual forward ball task; accepted labels alone are insufficient."""
    m={k:copy.deepcopy(original[k]) for k in ('task_physics','task_court','task_string_strip','contact_point_local') if k in original}
    m['seed_metadata']=copy.deepcopy(original)
    t=float(row['time_s']);k=round(t/.01)
    if not np.isfinite(t) or k<1 or abs(k*.01-t)>1e-9:raise ValueError('Control-grid event required')
    goal=row['analytic_goal']
    if goal.get('accepted') is not True:raise ValueError('Unverified analytic goal')
    arrays=[np.asarray(row['ball_world_m'],float),np.asarray(row['incoming_world_m_s'],float),
        np.asarray(goal['normal_world'],float),np.asarray(goal['racket_velocity_world_m_s'],float)]
    if any(a.shape!=(3,) or not np.isfinite(a).all() for a in arrays):raise ValueError('Finite event state required')
    ball,incoming,normal,velocity=arrays
    if abs(np.linalg.norm(normal)-1)>1e-8:raise ValueError('Unit actual goal normal required')
    forward=impact.evaluate(ball,incoming,ball-impact.physics['collision_radius_m']*normal,velocity,normal,m['contact_point_local'])
    margins=task_margins(forward,impact.court)
    if not margins['valid']:raise ValueError('Forward task rejected: '+margins['limiting_constraint'])
    initial=np.asarray(initial_q,float)
    if initial.shape!=(25,) or not np.isfinite(initial).all() or abs(np.linalg.norm(initial[3:7])-1)>1e-6:raise ValueError('Original finite root state required')
    R=pin.Quaternion(initial[3:7]).matrix()
    m.update(contact_index=k,contact_time_s=t,target_position_world_m=R.T@(ball-initial[:3]),
        incoming_velocity_planning=R.T@incoming,target_normal_world=R.T@normal,
        target_velocity_world_m_s=R.T@velocity,retargeted_velocity_current_frame_m_s=R.T@velocity,
        analytic_goal_valid=True,initialization_only=False,retimed_forward_goal=forward,
        retimed_task_margins=margins,robot_target_verified=False)
    return m


def continued_event_reference(record,metadata,initial_reference,arm_q_indices):
    """Restore only a complete same-event model prefix, never a partial plan."""
    encode=lambda value:json.dumps(value,sort_keys=True,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else float(x))
    if encode(record['metadata'])!=encode(metadata) or record['event_time_s']!=metadata['contact_time_s']:
        raise ValueError('Continuation event or target changed')
    selected=record['trials'][record['selected_index']]['result'];n=metadata['contact_index']
    arm_tracking_response(selected,arm_q_indices,n)
    trace=copy.deepcopy(selected['trajectory'])
    commands=np.asarray([t['final_target_rad'] for t in trace],float)
    reference=np.asarray(record['selected_arm_reference_rad'],float).copy()
    if commands.shape!=(n,18) or not np.isfinite(commands).all():raise ValueError('Complete finite final commands required')
    if (reference.shape!=(n+1,6) or not np.isfinite(reference).all()
            or not np.array_equal(reference[0],np.asarray(initial_reference))
            or not np.array_equal(reference[1:],np.asarray([t['planned_arm_reference_rad'] for t in trace]))):
        raise ValueError('Continuation reference history changed')
    return trace,commands.copy(),reference


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--event-source',type=Path,required=True);p.add_argument('--event-time',type=float,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--constrain-arm-tracking',action='store_true');p.add_argument('--previous-refinement',type=Path);p.add_argument('--tracking-model-reserve',action='store_true');a=p.parse_args()
    if a.tracking_model_reserve and (a.previous_refinement is None or not a.constrain_arm_tracking):raise ValueError('Tracking reserve requires a constrained verified predecessor')
    if a.output.exists():raise ValueError('Preserve all prior attempts')
    r=json.loads(a.event_source.read_text())
    for path,digest in r['input_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('Input drift')
    if a.event_time not in r['original_times_s']:raise ValueError('Event must belong to the original window')
    rows=[v for v in r['rows'] if abs(v['time_s']-a.event_time)<1e-9]
    if len(rows)!=1 or rows[0]['status']!='MODEL_PATH_ONLY' or not r['path']['path_valid']:raise ValueError('Complete causal seed required')
    row=rows[0];initial_path=Path('reports/push_off_task_20261118/DIAGNOSED.json');initial=json.loads(initial_path.read_text())['rows'][0]
    original_path=Path('reports/experimental_fullbody_gate_20261118/REPLAY.json.gz')
    matches=[c for c in json.load(gzip.open(original_path,'rt'))['physical_calls'] if c['lane']==initial['lane']]
    if len(matches)!=1:raise ValueError('Unique originating event required')
    original=matches[0]['metadata'];model=LoadedPinocchioModelV1();q=np.array(initial['initial_q']);v=np.array(initial['initial_v'])
    obs_path=Path('reports/joint_loop_round_20261117/attempts/probe01/native/public_observations.npy')
    if hashlib.sha256(obs_path.read_bytes()).hexdigest()!=r['observations_sha256']:raise ValueError('Observation source drift')
    obs=np.asarray(np.load(obs_path,mmap_mode='r')[r['observed_tick'],initial['lane']])
    if abs(obs[775]-initial['clock_s'])>1e-7:raise ValueError('Wrong current observation')
    p0,v0=observed_initial_ball(obs[None],[1.1,0,1.8],[1,0,0]);ball,incoming=propagate_ball_state(p0[0],v0[0],a.event_time,original['task_physics'])
    if not np.allclose(ball,row['ball_world_m'],atol=1e-10,rtol=0) or not np.allclose(incoming,row['incoming_world_m_s'],atol=1e-10,rtol=0):raise ValueError('Causal target reconstruction mismatch')
    impact=ImpactOutcomeEvaluator(original['task_physics'],original['task_court'],load_racket());metadata=retimed_metadata(original,row,q,impact);n=metadata['contact_index']
    trace=r['path']['trajectory'][:n]
    if len(trace)!=n or any(t['node']!=k for k,t in enumerate(trace)):raise ValueError('Complete ordered event prefix required')
    commands=np.array([t['final_target_rad'] for t in trace]);reference=np.array(r['arm_reference_rad'])[:n+1]
    if reference.shape!=(n+1,6) or not np.array_equal(reference[1:],np.array([t['planned_arm_reference_rad'] for t in trace])):raise ValueError('Reference history mismatch')
    state=PublicState(q.copy(),v.copy(),pin.Quaternion(q[3:7]).matrix(),q[:3].copy(),q[model.q_indices].copy(),v[model.v_indices].copy(),np.array(initial['baseline_initial_contacts'],bool),0.)
    state.previous_command_target=np.array(initial['previous_final_command'])
    cfg_path=Path('reports/codex_teacher_joint_eval_20261116/controller.json');interface=PositionPDInterface(**json.loads(cfg_path.read_text())['interface'])
    reserve=0.;reserve_audit=None
    input_paths=[a.event_source,initial_path,original_path,obs_path,cfg_path]
    if a.previous_refinement is not None:
        previous=json.loads(a.previous_refinement.read_text())
        if hashlib.sha256(a.previous_refinement.read_bytes()).hexdigest()!=a.previous_refinement.with_suffix('.sha256').read_text().strip():
            raise ValueError('Continuation artifact hash mismatch')
        for path in input_paths:
            if previous['input_sha256'].get(str(path))!=hashlib.sha256(path.read_bytes()).hexdigest():raise ValueError('Continuation core input drift')
        for path,digest in previous['input_sha256'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('Continuation ancestry drift')
        trace,commands,reference=continued_event_reference(previous,metadata,reference[0],model.q_indices[12:])
        if a.tracking_model_reserve:reserve,reserve_audit=tracking_linearization_reserve(previous,model.q_indices[12:])
        input_paths.append(a.previous_refinement)
    geometry=AuthoredGroundClearance('configs/go2_piper_authored_ground_geometry_v1.json',model,margin_m=0.)
    predictor=ExperimentalSphereAdmissionV1(model,geometry,Path('reports/prediction_model_validation_20261118/AUTHORED_SPHERES.json'))
    evaluator=CoupledContactRollout(model,interface,predictor,state,commands,reference,metadata);calls=0
    def evaluate(amplitude):
        nonlocal calls
        result=evaluator(amplitude);calls+=1
        if calls==1:
            if not result['path_valid']:raise ValueError('Seed replay failed')
            for old,new in zip(trace,result['trajectory']):
                if not np.array_equal(old['q'],new['q']) or not np.array_equal(old['v'],new['v']):raise ValueError('Seed state drift')
        print(json.dumps(dict(call=calls,path_valid=result['path_valid'],errors=result.get('terminal_errors'),failure=result.get('failure'))),flush=True)
        return result
    started=time.perf_counter();result=refine_terminal(evaluate,parameter_radii=evaluator.radii,
        amplitude_bounds=evaluator.amplitude_bounds(),linear_constraints=evaluator.combined_arm_constraints(),backtracking=True,protect_terminal_tasks=True,
        path_response_fn=(lambda result:arm_tracking_response(result,model.q_indices[12:],n)) if a.constrain_arm_tracking else None,path_response_reserve=reserve)
    selected=result['trials'][result['selected_index']]['result']
    result.update(tracking_model_reserve_audit=reserve_audit,constrain_arm_tracking=a.constrain_arm_tracking,seconds=time.perf_counter()-started,event_time_s=a.event_time,metadata=metadata,base_arm_reference_rad=reference,
        selected_arm_reference_rad=np.vstack([reference[0],[t['planned_arm_reference_rad'] for t in selected['trajectory']]]),
        original_prefix_exact=True,input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in input_paths},
        native_execution=False,valid_teacher_reference=False)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else float(x))+'\n')
    a.output.with_suffix('.sha256').write_text(hashlib.sha256(a.output.read_bytes()).hexdigest()+'\n')
    print(json.dumps(dict(improved=result['improved'],errors=selected.get('terminal_errors'),seconds=result['seconds'])))
if __name__=='__main__':main()
