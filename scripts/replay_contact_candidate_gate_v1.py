"""Offline gate replay on recorded causal public states; never a native success test."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from quadsmash.control.full_racket_rally_controller_v4 import FullRacketRallyControllerV4, planning_world_pose
from quadsmash.control.impact_outcome_evaluator_v1 import ImpactOutcomeEvaluator
from scripts.causal_frozen_leg_prefix_v1 import CausalFrozenLegReferenceFactory
from scripts.contact_first_fullbody_evaluator_v1 import ContactFirstFullBodyEvaluatorV1
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1
from scripts.whole_body_support_qp_v1 import PositionPDInterface


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction-substeps',type=int,default=4)
    parser.add_argument('--joint-stops',action='store_true')
    parser.add_argument('--velocity-aware',action='store_true')
    parser.add_argument('--anchored',action='store_true')
    parser.add_argument('--inverse-impact',action='store_true')
    parser.add_argument('--reachable-impact',action='store_true')
    parser.add_argument('--replanning',action='store_true')
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--interface',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.velocity_aware and not args.anchored:raise ValueError("Velocity-aware refinement requires anchored search")
    if args.output.exists():raise ValueError('Preserve prior replay output')
    root=Path(__file__).resolve().parents[1]
    observations=np.load(args.input/'public_observations.npy',mmap_mode='r')
    # Stop at the first recorded planning event, selected without outcome labels.
    old=json.loads((args.input/'causal_plans.json').read_text())
    clock=old[0]['observation_clock_s'][0]
    end=int(np.argmin(np.abs(observations[:,0,775]-clock)))+1
    if args.replanning:end=min(len(observations),266)
    cfg=json.loads(args.interface.read_text())['interface']
    interface=PositionPDInterface(**cfg)
    authored=json.loads((root/'reports/G07_full_racket_v4/CPU_REPLAY_AUTHORED_JOINTS_0724.json').read_text())['defaults_before_neutral_arm_override']
    order=json.loads((root/'configs/go2_piper_racket_landing_model_v1.json').read_text())['joint_order']
    default=np.array([authored[n] for n in order]);default[12:]=json.loads((root/'configs/full_racket_reset_v4.json').read_text())['arm_joint_positions_rad']
    default=torch.tensor(default,dtype=torch.float32);limits=torch.tensor(interface.limits,dtype=torch.float32)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    controller=FullRacketRallyControllerV4(checkpoint,observations.shape[1],'cpu')
    if args.inverse_impact or args.reachable_impact or args.anchored:
        from dataclasses import replace
        controller.contact_first_planner.config=replace(controller.contact_first_planner.config,inverse_impact_targets=True,reachable_impact_targets=args.reachable_impact,bounded_replanning=args.replanning,position_anchored_search=args.anchored,velocity_aware_pose=args.velocity_aware)
    controller.contact_first_offline_diagnostic=bool(args.anchored)
    physical=[];impacts=[];fallback_audit=[];bundles=[]
    physically_accepted_pending=[]  # No export until FINAL chosen-plan impact passes.
    frozen_inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        (args.input/'public_observations.npy',args.checkpoint,args.interface)}
    def factory(lane,public,q,qd):
        model=LoadedPinocchioModelV1();state=model.state_from_public(public,default.numpy())
        rotation,origin=planning_world_pose(public,[1.1,0.,1.8])
        legs=CausalFrozenLegReferenceFactory(controller.teacher,controller.parameters,
            controller.contact_first_current_public,default,limits,controller.event_time,
            controller.target_offset_world,lane,controller.physics,controller.contact_first_planner.config.dt_s)
        def measure(s,metadata):
            arm=torch.tensor(s.joint_position[12:],dtype=default.dtype)[None]
            offset=controller.teacher.contact_offset.clone()
            if 'contact_point_local' in metadata:offset[1:]=offset.new_tensor(metadata['contact_point_local'][1:])
            point,normal,jp,_=controller.teacher.kinematics(arm,offset)
            point,normal,jp=[x[0].numpy().astype(float) for x in (point,normal,jp)]
            R=s.root_rotation_world
            velocity=R@(s.v[:3]+np.cross(s.v[3:6],point)+jp@s.joint_velocity[12:])
            return rotation.T@(s.root_position_world+R@point-origin),rotation.T@R@normal,rotation.T@velocity
        evaluator=ContactFirstFullBodyEvaluatorV1(interface_config=cfg,model=model,
            initial_state=state,leg_reference_factory=legs,contact_prediction='unilateral_release',
            record_state_trace=True,racket_measure_fn=measure,measurement_uses_metadata=True,
            substeps=args.prediction_substeps,enforce_joint_stops=args.joint_stops)
        def evaluate(*a):
            result=evaluator(*a)
            physical.append(dict(lane=lane,result=result,metadata=a[3],arm_sequence_rad=np.asarray(a[0]).tolist()))
            if result.get('accepted'):
                # A FULLBODY pass is only an intermediate candidate; the
                # planner may reject its terminal actual-string/return result.
                # Store this snapshot without writing any reference archive.
                physically_accepted_pending.append(dict(
                    lane=lane,result=result,contact_index=int(a[2]),
                    initial_q=state.q.copy(),initial_v=state.v.copy(),
                    clock_s=float(public[775]),
                    prior_final_target_rad=(interface.limits[:,0]+
                        .5*(public[-18:]+1)*np.diff(interface.limits,axis=1)[:,0]),
                    controller_parameters=controller.parameters[lane].detach().cpu().tolist()))
            return result
        return evaluate
    controller.set_contact_first_evaluator_factory(factory)
    original=ImpactOutcomeEvaluator.evaluate
    def recorded(self,*a,**kw):
        result=original(self,*a,**kw);impacts.append(result.copy());return result
    # Process-local read-only instrumentation, restored even on failure.
    ImpactOutcomeEvaluator.evaluate=recorded
    try:
        with torch.inference_mode():
            for tick in range(end):
                old_planned=controller.planned.clone()
                before=(controller.parameters.clone(),controller.event_time.clone(),controller.target_offset_world.clone())
                controller.action({'policy':torch.tensor(np.array(observations[tick]))},default,limits)
                if old_planned.any():
                    drift=max(float((new[old_planned]-old[old_planned]).abs().max()) for new,old in zip(
                        (controller.parameters,controller.event_time,controller.target_offset_world),before))
                    fallback_audit.append(dict(tick=tick,frozen_fallback_state_drift=drift))
                    if drift:raise RuntimeError('Retry altered frozen fallback parameters')
    finally:ImpactOutcomeEvaluator.evaluate=original
    # Export ONLY a plan actually selected by the controller after FINAL impact
    # validation, matched to the exact physical 18D terminal prefix. Earlier
    # callbacks passing dynamics alone must never become teacher bundles.
    from scripts.offline_reference_bundle_v1 import (
        require_final_trajectory_admission,write_candidate)
    for lane in range(observations.shape[1]):
        if not bool(controller.contact_first_plan_valid[lane].item()):
            continue
        if not bool(controller.contact_first_joint_valid[lane].item()):
            raise ValueError('Accepted offline plan lacks fully-owned 18D prefix')
        selected_log=next((row for row in reversed(controller.contact_first_plan_log)
                           if int(row['lane'])==lane and row.get('accepted') is True),None)
        if selected_log is None:
            raise ValueError('Selected joint prefix lacks final planner admission log')
        count=int(selected_log['contact_index'])
        selected=(controller.contact_first_joint_targets[lane,:count]
                  .detach().cpu().numpy().astype(float))
        matches=[]
        for pending in physically_accepted_pending:
            if pending['lane']!=lane or pending['contact_index']!=count:
                continue
            replayed=np.asarray(pending['result']['replayed_target_sequence_rad'],dtype=float)
            if (replayed.shape==selected.shape and
                    np.max(np.abs(replayed-selected),initial=0.)<=2e-6):
                matches.append(pending)
        if len(matches)!=1:
            raise ValueError('Chosen 18D prefix has no unique physically verified provenance')
        admitted=matches[0]
        require_final_trajectory_admission(selected_log,admitted['result'])
        path=args.output.parent/(args.output.stem+'_candidates')/f'lane{lane}_final_verified'
        write_candidate(path,final_targets_rad=admitted['result']['replayed_target_sequence_rad'],
            prior_final_target_rad=admitted['prior_final_target_rad'],
            state_q=admitted['initial_q'],state_v=admitted['initial_v'],
            control_tick=round(admitted['clock_s']/.01),
            contact_index=count,limits=interface.limits,input_hashes=frozen_inputs,
            controller_parameters=admitted['controller_parameters'],
            prediction_contract=admitted['result']['prediction_contract'],
            cpu_prediction_valid=True,state_source='public_reconstruction')
        bundles.append(str(path))
    result=dict(scope='CPU gate replay; public reconstruction, no native truth or online execution',
        finite_mass_refinement='NOT_EVALUATED in this offline adapter',
        prediction_contract=dict(version='implicit_contact_diagnostic_v2',substeps=args.prediction_substeps,
            integration_dt_s=.01/args.prediction_substeps,joint_stops=args.joint_stops,physx_equivalent=False),
        velocity_aware_pose=args.velocity_aware,future_recorded_actions_read=False,source_ticks=end,training=False,native='NOT_RUN',
        input_sha256=frozen_inputs,offline_reference_bundles=bundles,
        position_anchored_search=args.anchored,inverse_impact_targets=bool(args.inverse_impact or args.anchored or args.reachable_impact),reachable_impact_targets=args.reachable_impact,replanning=args.replanning,inverse_impact_audit=getattr(controller,'inverse_impact_audit',[]),
        fallback_audit=fallback_audit,plans=controller.logs,impact_calls=impacts,physical_calls=physical)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else float(v))+'\n')
    print(json.dumps(dict(impact_calls=len(impacts),physical_calls=len(physical),
        physical_accepted=sum(x['result']['accepted'] for x in physical),output=str(args.output))))

if __name__=='__main__':main()
