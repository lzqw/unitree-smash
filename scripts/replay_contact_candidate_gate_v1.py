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
    parser.add_argument('--inverse-impact',action='store_true')
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--interface',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Preserve prior replay output')
    root=Path(__file__).resolve().parents[1]
    observations=np.load(args.input/'public_observations.npy',mmap_mode='r')
    # Stop at the first recorded planning event, selected without outcome labels.
    old=json.loads((args.input/'causal_plans.json').read_text())
    clock=old[0]['observation_clock_s'][0]
    end=int(np.argmin(np.abs(observations[:,0,775]-clock)))+1
    cfg=json.loads(args.interface.read_text())['interface']
    interface=PositionPDInterface(**cfg)
    authored=json.loads((root/'reports/G07_full_racket_v4/CPU_REPLAY_AUTHORED_JOINTS_0724.json').read_text())['defaults_before_neutral_arm_override']
    order=json.loads((root/'configs/go2_piper_racket_landing_model_v1.json').read_text())['joint_order']
    default=np.array([authored[n] for n in order]);default[12:]=json.loads((root/'configs/full_racket_reset_v4.json').read_text())['arm_joint_positions_rad']
    default=torch.tensor(default,dtype=torch.float32);limits=torch.tensor(interface.limits,dtype=torch.float32)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    controller=FullRacketRallyControllerV4(checkpoint,observations.shape[1],'cpu')
    if args.inverse_impact:
        from dataclasses import replace
        controller.contact_first_planner.config=replace(controller.contact_first_planner.config,inverse_impact_targets=True)
    physical=[];impacts=[]
    def factory(lane,public,q,qd):
        model=LoadedPinocchioModelV1();state=model.state_from_public(public,default.numpy())
        rotation,origin=planning_world_pose(public,[1.1,0.,1.8])
        legs=CausalFrozenLegReferenceFactory(controller.teacher,controller.parameters,
            controller.contact_first_current_public,default,limits,controller.event_time,
            controller.target_offset_world,lane,controller.physics,controller.contact_first_planner.config.dt_s)
        def measure(s):
            arm=torch.tensor(s.joint_position[12:],dtype=default.dtype)[None]
            point,normal,jp,_=controller.teacher.kinematics(arm,controller.teacher.contact_offset)
            point,normal,jp=[x[0].numpy().astype(float) for x in (point,normal,jp)]
            R=s.root_rotation_world
            velocity=R@(s.v[:3]+np.cross(s.v[3:6],point)+jp@s.joint_velocity[12:])
            return rotation.T@(s.root_position_world+R@point-origin),rotation.T@R@normal,rotation.T@velocity
        evaluator=ContactFirstFullBodyEvaluatorV1(interface_config=cfg,model=model,
            initial_state=state,leg_reference_factory=legs,contact_prediction='unilateral_release',
            record_state_trace=True,racket_measure_fn=measure)
        def evaluate(*a):
            result=evaluator(*a);physical.append(dict(lane=lane,result=result));return result
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
                controller.action({'policy':torch.tensor(np.array(observations[tick]))},default,limits)
    finally:ImpactOutcomeEvaluator.evaluate=original
    result=dict(scope='CPU gate replay; public reconstruction, no native truth or online execution',
        finite_mass_refinement='NOT_EVALUATED in this offline adapter',
        future_recorded_actions_read=False,source_ticks=end,training=False,native='NOT_RUN',
        input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (args.input/'public_observations.npy',args.checkpoint,args.interface)},
        inverse_impact_targets=args.inverse_impact,inverse_impact_audit=getattr(controller,'inverse_impact_audit',[]),
        plans=controller.logs,impact_calls=impacts,physical_calls=physical)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(impact_calls=len(impacts),physical_calls=len(physical),
        physical_accepted=sum(x['result']['accepted'] for x in physical),output=str(args.output))))

if __name__=='__main__':main()
