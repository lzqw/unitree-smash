"""One evidence-derived offline push-off request; no parameter sweep or native run."""
import argparse,copy,gzip,json,hashlib
from pathlib import Path
import numpy as np
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance
from scripts.experimental_sphere_admission_v1 import ExperimentalSphereAdmissionV1
from scripts.push_off_support_qp_v1 import PushOffSupportQPV1


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();output=args.output
    if output.exists():raise ValueError('Preserve previous result')
    source=root/'reports/experimental_fullbody_gate_20261118/REPLAY.json.gz'
    base=root/'reports/action_conditioned_contact_20261118/RESULTS_WITH_BOUND.json'
    replay=json.load(gzip.open(source,'rt'));reference=json.loads(base.read_text())
    observations_path=root/'reports/joint_loop_round_20261117/attempts/probe01/native/public_observations.npy'
    if hashlib.sha256(observations_path.read_bytes()).hexdigest()!=reference['observations_sha256']:raise ValueError('Input drift')
    public=np.load(observations_path,mmap_mode='r');model=LoadedPinocchioModelV1()
    authored=json.loads((root/'reports/G07_full_racket_v4/CPU_REPLAY_AUTHORED_JOINTS_0724.json').read_text())['defaults_before_neutral_arm_override']
    default=np.array([authored[n] for n in model.joint_order]);default[12:]=json.loads((root/'configs/full_racket_reset_v4.json').read_text())['arm_joint_positions_rad']
    interface=PositionPDInterface(**json.loads((root/'reports/codex_teacher_joint_eval_20261116/controller.json').read_text())['interface'])
    geometry=AuthoredGroundClearance(root/'configs/go2_piper_authored_ground_geometry_v1.json',model,margin_m=0.)
    predict=ExperimentalSphereAdmissionV1(model,geometry,root/'reports/prediction_model_validation_20261118/AUTHORED_SPHERES.json')
    rows=[]
    for event in reference['rows']:
        matches=[c for c in replay['physical_calls'] if c['lane']==event['lane'] and abs(c['metadata']['contact_time_s']-event['duration_s'])<1e-9]
        if len(matches)!=1:raise ValueError('Ambiguous source prefix')
        call=matches[0];obs=np.asarray(public[event['current_observation_tick'],event['lane']])
        state=model.state_from_public(obs,default)
        state.previous_command_target=interface.limits[:,0]+.5*(obs[-18:]+1)*np.diff(interface.limits,axis=1)[:,0]
        target=np.asarray(call['result']['replayed_target_sequence_rad'])[0]
        before=model.snapshot(state.q,state.v)
        baseline=predict(model,interface,copy.deepcopy(state),target.copy(),duration_s=.01,substeps=160,enforce_joint_stops=True)
        after=model.snapshot(baseline['state'].q,baseline['state'].v)
        baseline_accel=(baseline['state'].v-state.v)/.01
        baseline_com_accel=(after['com_velocity_world']-before['com_velocity_world'])/.01
        # One requested COM acceleration from the conservative reach deficit.
        # It is a task seed, not a claim that root/COM motion is interchangeable
        # or that this displacement can actually be produced during stance.
        deficit=event['reach_bound']['conditional_translation_lower_bound_m']
        direction=np.asarray(event['target_future_body_m'])-event['reach_bound']['shoulder_body_m']
        direction/=np.linalg.norm(direction)
        R=state.root_rotation_world@np.asarray(event['transport']['rotation_current_from_future'])
        duration=.08;remaining=event['duration_s']
        delta=R@direction*deficit/(duration*(remaining-.5*duration))
        qp=PushOffSupportQPV1(interface)
        plan=qp.propose(state,target,baseline_accel,baseline_com_accel,delta,np.zeros(3),phase='PUSH_OFF')
        zero_request=PushOffSupportQPV1(interface).propose(state,target,baseline_accel,baseline_com_accel,np.zeros(3),np.zeros(3),phase='PUSH_OFF')
        row=dict(lane=event['lane'],clock_s=event['clock_s'],initial_q=state.q,initial_v=state.v,
            previous_final_command=state.previous_command_target,reference=target,
            baseline_acceleration=baseline_accel,baseline_com_acceleration=baseline_com_accel,
            baseline_nonlinear_diagnostics=baseline['diagnostics'],baseline_contact_events=baseline['events'],
            baseline_terminal_contacts=baseline['state'].contacts,baseline_initial_contacts=state.contacts,
            zero_increment_qp=zero_request,
            requested_extra_com_acceleration_world=delta,assumed_request_window_s=duration,
            request_is_unrealized=True,qp=plan,nonlinear_candidate_test='NOT_RUN')
        if plan['accepted']:
            try:
                predicted=predict(model,interface,copy.deepcopy(state),np.asarray(plan['joint_position_target']),duration_s=.01,substeps=160,enforce_joint_stops=True)
                row.update(nonlinear_candidate_test='COMPLETED_ONE_INTERVAL',diagnostics=predicted['diagnostics'],
                    predicted_q=predicted['state'].q,predicted_v=predicted['state'].v)
            except (ValueError,np.linalg.LinAlgError) as error:
                row.update(nonlinear_candidate_test='MODEL_UNCERTAIN',error=str(error))
        rows.append(row)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(scope='one task request, not a complete prefix or native evaluation',rows=rows,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),reference_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),
        native_execution=False,valid_teacher_reference=False,training=False),indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else float(v))+'\n')
    print(json.dumps([dict(lane=r['lane'],accepted=r['qp']['accepted'],status=r['qp'].get('solver_status'),nonlinear=r['nonlinear_candidate_test']) for r in rows]))
if __name__=='__main__':main()
