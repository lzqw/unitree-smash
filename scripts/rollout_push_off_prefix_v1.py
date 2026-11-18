"""One bounded support-window rollout with original-command fallback.

QP proposals are never applied without original nonlinear path checks. No
synthetic takeoff state/contact activation, no search over gains or samples.
"""
import argparse,copy,gzip,json,hashlib,time
from pathlib import Path
import numpy as np
import pinocchio as pin
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1,PublicState
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance
from scripts.experimental_sphere_admission_v1 import ExperimentalSphereAdmissionV1
from scripts.contact_first_fullbody_evaluator_v1 import ContactFirstFullBodyEvaluatorV1
from scripts.push_off_support_qp_v1 import PushOffSupportQPV1
from scripts.audit_prediction_contact_geometry_v2 import racket_contact_state
from quadsmash.control.impact_outcome_evaluator_v1 import ImpactOutcomeEvaluator
from quadsmash.physics.full_racket_v4 import load_racket
from quadsmash.control.push_off_window_v1 import PushOffWindowV1
from scripts.nonlinear_push_correction_v1 import correct_push_command


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--contact-window',action='store_true');p.add_argument('--preserve-momentum-rate',action='store_true');p.add_argument('--nonlinear-push-correction',action='store_true');p.add_argument('--com-scale',type=float,default=1.);a=p.parse_args()
    if not np.isfinite(a.com_scale) or not 0<=a.com_scale<=1:raise ValueError('Bounded COM request scale required')
    if a.nonlinear_push_correction and not a.preserve_momentum_rate:raise ValueError('Explicit momentum task required for correction')
    if a.output.exists():raise ValueError('Preserve prior attempts')
    root=Path(__file__).resolve().parents[1]
    source=root/'reports/push_off_task_20261118/DIAGNOSED.json';seed=json.loads(source.read_text())
    replay=json.load(gzip.open(root/'reports/experimental_fullbody_gate_20261118/REPLAY.json.gz','rt'))
    interface=PositionPDInterface(**json.loads((root/'reports/codex_teacher_joint_eval_20261116/controller.json').read_text())['interface'])
    results=[]
    for request in seed['rows']:
        calls=[c for c in replay['physical_calls'] if c['lane']==request['lane']]
        if len(calls)!=1:raise ValueError('Unique command prefix required')
        call=calls[0];metadata=call['metadata'];commands=np.asarray(call['result']['replayed_target_sequence_rad'])
        model=LoadedPinocchioModelV1();q=np.asarray(request['initial_q']);v=np.asarray(request['initial_v']);R=pin.Quaternion(q[3:7]).matrix()
        state=PublicState(q.copy(),v.copy(),R.copy(),q[:3].copy(),q[model.q_indices].copy(),v[model.v_indices].copy(),np.asarray(request['baseline_initial_contacts'],bool),0.)
        state.previous_command_target=np.asarray(request['previous_final_command']).copy()
        geometry=AuthoredGroundClearance(root/'configs/go2_piper_authored_ground_geometry_v1.json',model,margin_m=0.)
        predictor=ExperimentalSphereAdmissionV1(model,geometry,root/'reports/prediction_model_validation_20261118/AUTHORED_SPHERES.json')
        qp=PushOffSupportQPV1(interface);rows=[];failure=None;adopted=0
        initial_witness=geometry.minimum_witness(state.q)
        if initial_witness['gap_m'] < -1e-6:raise ValueError('Initial nonfoot geometry violation')
        delta=np.asarray(request['requested_extra_com_acceleration_world'])*a.com_scale;start=time.perf_counter()
        window=PushOffWindowV1(len(commands)) if a.contact_window else None
        for node,nominal in enumerate(commands):
            decision=(window.step(node,state.contacts) if window else dict(enabled=node<8 and bool(state.contacts.any()),phase='legacy_fixed_window'))
            before=model.snapshot(state.q,state.v)
            try:
                baseline=predictor(model,interface,copy.deepcopy(state),nominal.copy(),duration_s=.01,substeps=160,enforce_joint_stops=True)
                if decision['enabled']:
                    after=model.snapshot(baseline['state'].q,baseline['state'].v)
                    plan=qp.propose(state,nominal,(baseline['state'].v-state.v)/.01,
                        (after['com_velocity_world']-before['com_velocity_world'])/.01,delta,np.zeros(3),phase='PUSH_OFF',
                        desired_angular_momentum_rate_world=((after['momentum_world'][3:]-before['momentum_world'][3:])/.01 if a.preserve_momentum_rate else None))
                else:plan=dict(accepted=False,rejection='outside_support_window')
                target=np.asarray(plan['joint_position_target']) if plan['accepted'] else nominal.copy()
                correction=None
                if a.nonlinear_push_correction and plan['accepted']:
                    correction=correct_push_command(model,interface,predictor,state,target,
                        (after['com_velocity_world']-before['com_velocity_world'])/.01+delta,
                        (after['momentum_world'][3:]-before['momentum_world'][3:])/.01,
                        np.asarray(call['arm_sequence_rad'])[node+1])
                    if correction['accepted']:target=correction['command'].copy()
                chosen=(predictor(model,interface,copy.deepcopy(state),target,duration_s=.01,substeps=160,enforce_joint_stops=True)
                        if plan['accepted'] else baseline)
                diagnostics=chosen['diagnostics'];bad=ContactFirstFullBodyEvaluatorV1._violations(diagnostics)
                if diagnostics['nonfoot_ground_violation_m']>1e-6:bad['nonfoot_ground_violation_m']=diagnostics['nonfoot_ground_violation_m']
                if diagnostics['nonfoot_contact_substeps']>0:bad['nonfoot_contact_substeps']=diagnostics['nonfoot_contact_substeps']
                roundtrip=interface.limits[:,0]+.5*(interface.normalized_action(target)+1)*np.diff(interface.limits,axis=1)[:,0]
                command_error=float(np.max(abs(roundtrip-target)))
                if command_error>2e-6:bad['command_roundtrip_rad']=command_error
                tracking=float(np.max(abs(chosen['state'].joint_position[12:]-np.asarray(call['arm_sequence_rad'])[node+1])))
                if tracking>.25:bad['arm_tracking_error_rad']=tracking
                if bad:failure=dict(node=node,kind='path_violation',violations=bad);break
                adopted+=int(plan['accepted']);state=chosen['state']
                rows.append(dict(node=node,q=state.q,v=state.v,contacts=state.contacts,final_target_rad=target,
                    elapsed_s=(node+1)*.01,window=decision,qp=plan,nonlinear_correction=correction,diagnostics=diagnostics,command_error_rad=command_error,arm_tracking_error_rad=tracking))
            except (ValueError,np.linalg.LinAlgError) as error:
                failure=dict(node=node,kind='MODEL_UNCERTAIN',error=str(error));break
        point,normal,velocity=racket_contact_state(model,state.q,state.v,np.asarray(metadata['contact_point_local'])+np.array([metadata['task_physics']['collision_radius_m'],0,0]))
        pc=R.T@(point-q[:3]);nc=R.T@normal;vc=R.T@velocity
        errors=dict(position_m=float(np.linalg.norm(pc-metadata['target_position_world_m'])),
            normal=float(np.linalg.norm(nc-metadata['target_normal_world'])),
            velocity_m_s=float(np.linalg.norm(vc-metadata['retargeted_velocity_current_frame_m_s'])))
        path_valid=failure is None and len(rows)==len(commands)
        terminal_pass=path_valid and errors['position_m']<=.003 and errors['normal']<=.12 and errors['velocity_m_s']<=.75
        impact=None
        if terminal_pass:
            ball=q[:3]+R@np.asarray(metadata['target_position_world_m'])
            incoming=R@np.asarray(metadata['incoming_velocity_planning'])
            impact=ImpactOutcomeEvaluator(metadata['task_physics'],metadata['task_court'],load_racket()).evaluate(
                ball,incoming,point-metadata['task_physics']['collision_radius_m']*normal,
                velocity,normal,metadata['contact_point_local'])
        ball_valid=bool(impact and all(impact.get(k) is True for k in
            ('contact_feasible','legal_return_predicted','attack5_predicted','target_landing_predicted')))
        results.append(dict(lane=request['lane'],com_request_scale=a.com_scale,requested_delta_com_acceleration_world=delta,
            preserve_baseline_angular_momentum_rate=a.preserve_momentum_rate,
            nonlinear_push_correction=a.nonlinear_push_correction,
            support_window_ticks=(None if a.contact_window else 8),
            support_window_mode=('causal_contact_latched' if a.contact_window else 'legacy_fixed8'),adopted_qp_commands=adopted,steps=len(rows),required_steps=len(commands),
            path_valid=path_valid,failure=failure,terminal_errors=errors,
            final_root_position=state.q[:3],trajectory=rows,seconds=time.perf_counter()-start,
            FULLBODY_PREDICTION_VALID=bool(terminal_pass and ball_valid),terminal_contact_passed=bool(terminal_pass),
            final_impact=impact,final_impact_revalidation=('COMPLETED' if impact else 'NOT_RUN_TERMINAL_REJECTED'),
            native_execution=False,valid_teacher_reference=False))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        mode='one deterministic requested support correction, no parameter optimization',results=results),indent=2,
        default=lambda v:v.tolist() if isinstance(v,np.ndarray) else float(v))+'\n')
    print(json.dumps([{k:r[k] for k in ('lane','adopted_qp_commands','steps','path_valid','failure','terminal_errors','seconds')} for r in results]))
if __name__=='__main__':main()
