"""Run a bounded real CPU event-coupled search from a frozen historical prefix.

This is a development search only.  It uses the existing nonlinear 18-D
predictor and same-candidate impact builder.  It never changes PhysX assets,
formal scoring, or the SuperMimic line.
"""
import argparse,copy,gzip,hashlib,json,time
from pathlib import Path
import numpy as np
import pinocchio as pin
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1,PublicState
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance
from scripts.experimental_sphere_admission_v1 import ExperimentalSphereAdmissionV1
from scripts.event_coupled_fullbody_rollout_v1 import EventCoupledFullBodyRolloutV1,EventCoupledFullBodyEventRolloutV1
from scripts.coupled_contact_prefix_v1 import LegRateCoupledContactRollout
from scripts.strike_state_impact_builder_v1 import StrikeStateImpactBuilderV1
from quadsmash.control.impact_outcome_evaluator_v1 import ImpactOutcomeEvaluator
from quadsmash.control.strict_impact_admission_v1 import StrictImpactAdmissionV1
from quadsmash.physics.full_racket_v4 import load_racket
from quadsmash.physics.court_v4 import load_court
from quadsmash.control.event_coupled_takeoff_strike_v1 import EventCoupledTakeoffStrikePlannerV1
from quadsmash.control.initial_intercept import observed_initial_ball,propagate_ball_state
from scripts.event_wbc_command_generator_v1 import EventWBCCommandGeneratorV1

ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--max-rollouts',type=int,default=6);ap.add_argument('--iterations',type=int,default=1);ap.add_argument('--search-only',action='store_true');ap.add_argument('--output',type=Path,default=ROOT/'reports/event_coupled_production_20261118/RESULTS_v2.json');args=ap.parse_args()
    out=args.output if args.output.is_absolute() else ROOT/args.output
    if out.exists(): raise ValueError('Preserve prior production search')
    initial=json.loads((ROOT/'reports/push_off_task_20261118/DIAGNOSED.json').read_text())['rows'][0]
    prior=json.loads((ROOT/'reports/wider_leg_rate_20261118/RESULTS.json').read_text())
    selected=prior['trials'][prior['selected_index']]['result']
    replay=json.load(gzip.open(ROOT/'reports/experimental_fullbody_gate_20261118/REPLAY.json.gz','rt'))
    calls=[c for c in replay['physical_calls'] if c['lane']==initial['lane']]
    if len(calls)!=1: raise ValueError('Unique historical lane required')
    call=calls[0]; original=call['metadata']; model=LoadedPinocchioModelV1()
    q=np.asarray(initial['initial_q'],float);v=np.asarray(initial['initial_v'],float)
    state=PublicState(q.copy(),v.copy(),pin.Quaternion(q[3:7]).matrix(),q[:3].copy(),q[model.q_indices].copy(),v[model.v_indices].copy(),np.asarray(initial['baseline_initial_contacts'],bool),0.)
    state.previous_command_target=np.asarray(initial['previous_final_command'],float).copy()
    interface=PositionPDInterface(**json.loads((ROOT/'reports/codex_teacher_joint_eval_20261116/controller.json').read_text())['interface'])
    geometry=AuthoredGroundClearance(ROOT/'configs/go2_piper_authored_ground_geometry_v1.json',model,margin_m=0.)
    predictor=ExperimentalSphereAdmissionV1(model,geometry,ROOT/'reports/prediction_model_validation_20261118/AUTHORED_SPHERES.json')
    trace=selected['trajectory']; commands=np.asarray([x['final_target_rad'] for x in trace],float)
    reference=np.asarray(prior['selected_arm_reference_rad'],float)
    metadata=copy.deepcopy(prior['metadata']); metadata['ball_position_world']=q[:3]+pin.Quaternion(q[3:7]).matrix()@np.asarray(metadata['target_position_world_m']); metadata['ball_velocity_world_m_s']=pin.Quaternion(q[3:7]).matrix()@np.asarray(metadata['incoming_velocity_planning'])
    obs_path=ROOT/'reports/joint_loop_round_20261117/attempts/probe01/native/public_observations.npy'; obs=np.asarray(np.load(obs_path,mmap_mode='r')[249,initial['lane']])
    p0,v0=observed_initial_ball(obs[None],[1.1,0,1.8],[1,0,0]); R0=pin.Quaternion(q[3:7]).matrix(); t0=float(metadata['contact_time_s'])
    def event_metadata(event,old):
        _,_,hit,*_=np.asarray(event,float); ball,incoming=propagate_ball_state(p0[0],v0[0],t0+float(hit),original['task_physics'])
        updated=copy.deepcopy(old); updated['ball_position_world']=ball; updated['ball_velocity_world_m_s']=incoming
        updated['target_position_world_m']=R0.T@(ball-q[:3]); updated['incoming_velocity_planning']=R0.T@incoming; updated['contact_time_s']=t0+float(hit)
        updated['contact_index']=max(2,int(round(updated['contact_time_s']/.01)))
        return updated
    impact=ImpactOutcomeEvaluator(metadata['task_physics'],metadata['task_court'],load_racket())
    admission=StrictImpactAdmissionV1(impact)
    builder=StrikeStateImpactBuilderV1(model,admission,initial_state_hash=hashlib.sha256(np.r_[q,v].tobytes()).hexdigest(),command_sequence_hash=hashlib.sha256(commands.tobytes()).hexdigest())
    base=EventCoupledFullBodyRolloutV1(model,interface,predictor,state,commands,reference,metadata,impact_builder=builder,rollout_class=LegRateCoupledContactRollout)
    wbc_generator=EventWBCCommandGeneratorV1(interface)
    event_rollout=EventCoupledFullBodyEventRolloutV1(base,event_metadata_fn=event_metadata,wbc_event_fn=wbc_generator)
    local_bounds=np.asarray(base.rollout.amplitude_bounds(),float)
    bounds=np.vstack([event_rollout.event_bounds,local_bounds])
    planner=EventCoupledTakeoffStrikePlannerV1(event_rollout,bounds=bounds,max_candidates=64)
    planner.probe_order=list(range(8,event_rollout.parameter_size))+list(range(8))
    zero=np.zeros(event_rollout.parameter_size); seed=zero.copy(); seed[8:]=np.asarray(prior['trials'][prior['selected_index']]['amplitude_rad'],float)
    event_seed=zero.copy(); event_seed[2]=.03; event_seed[3]=.2; event_seed[4]=.3; event_seed[5]=.15; event_seed[6]=.04; event_seed[7]=-.04
    event_seed_u=zero.copy(); event_seed_u[6]=.12; event_seed_u[7]=-.12
    event_seed_v=zero.copy(); event_seed_v[6]=-.12; event_seed_v[7]=.12
    event_seed_hit=zero.copy(); event_seed_hit[2]=-.03; event_seed_hit[3]=-.2; event_seed_hit[4]=-.3; event_seed_hit[5]=-.15
    seeds=[seed,event_seed,event_seed_u,event_seed_v,event_seed_hit,zero]
    started=time.perf_counter()
    try:
        result=planner.search(seeds) if args.search_only else planner.optimize(seeds,iterations=args.iterations,step_fraction=.08,max_rollouts=args.max_rollouts)
    except Exception as exc:
        result=dict(best_feasible=None,best_contact=None,best_kinematic=None,trace=[],calls=0,budget=args.max_rollouts,
                    search_exception=type(exc).__name__+': '+str(exc),event_parameter_size=event_rollout.parameter_size,
                    local_parameter_size=event_rollout.local_size,base_parameter_size=base.parameter_size,
                    bounds_shape=list(bounds.shape))
    result['seconds']=time.perf_counter()-started
    result['source']='same frozen historical initial state and command history'; result['native_debug_eligible']=False; result['native_run']=False; result['teacher_eligible']=False
    result['baseline_reference_errors']=selected['terminal_errors']; result['input_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'reports/push_off_task_20261118/DIAGNOSED.json',ROOT/'reports/wide_leg_position_20261118/RESULTS.json',ROOT/'reports/experimental_fullbody_gate_20261118/REPLAY.json.gz')}
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else float(x))+'\n'); (out.with_suffix('.sha256')).write_text(hashlib.sha256(out.read_bytes()).hexdigest()+'\n')
    print(json.dumps({'calls':result['calls'],'seconds':result['seconds'],'best_feasible':result.get('best_feasible') is not None,
                      'best_contact':result.get('best_contact',min((x['result'].get('terminal_errors',{}) for x in result.get('failed',[])),key=lambda e:e.get('position_m',1e9),default=None))},default=lambda x:x.tolist() if isinstance(x,np.ndarray) else float(x)))
if __name__=='__main__': main()
