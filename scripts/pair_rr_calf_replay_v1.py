"""Same-state recorded/frozen/joint command replay. Future truth is offline only."""
import argparse,gzip,json
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from quadsmash.control.full_racket_rally_controller_v4 import FullRacketRallyControllerV4
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.causal_frozen_leg_prefix_v1 import CausalFrozenLegReferenceFactory
from scripts.unilateral_implicit_prediction_v1 import predict_unilateral_support
from scripts.contact_first_fullbody_evaluator_v1 import ContactFirstFullBodyEvaluatorV1
from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--prior-replay',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--joint-stops',action='store_true');p.add_argument('--substeps',type=int,default=4);a=p.parse_args()
    if a.substeps not in (4,160):raise ValueError('Use fixed coarse4 or native-timestep160 diagnostic, not a parameter scan')
    if a.output.exists():raise ValueError('Preserve evidence')
    root=Path(__file__).resolve().parents[1];obs=np.load(a.input/'public_observations.npy');actions=np.load(a.input/'public_actions.npy')
    prior=json.loads(gzip.decompress(a.prior_replay.read_bytes()));source=prior['physical_calls'][0]
    start=249;lane=source['lane'];arm=np.asarray(source['arm_sequence_rad']);steps=source['metadata']['contact_index']
    model=LoadedPinocchioModelV1();cfg=json.loads((root/'reports/codex_teacher_joint_eval_20261116/controller.json').read_text())['interface'];interface=PositionPDInterface(**cfg)
    authored=json.loads((root/'reports/G07_full_racket_v4/CPU_REPLAY_AUTHORED_JOINTS_0724.json').read_text())['defaults_before_neutral_arm_override']
    default=np.array([authored[n] for n in model.joint_order]);default[12:]=json.loads((root/'configs/full_racket_reset_v4.json').read_text())['arm_joint_positions_rad']
    default=torch.tensor(default,dtype=torch.float32);limits=torch.tensor(interface.limits,dtype=torch.float32)
    controller=FullRacketRallyControllerV4(torch.load(a.checkpoint,map_location='cpu',weights_only=False),obs.shape[1],'cpu');controller.contact_first_planner=None
    runs=[]
    with torch.inference_mode():
        for tick in range(start):controller.action({'policy':torch.tensor(obs[tick])},default,limits)
        for mode in ('historical_actual18','frozen_base18','frozen_legs_planned_arm18'):
            state=model.state_from_public(obs[start,lane],default.numpy());geometry=AuthoredGroundClearance(root/'configs/go2_piper_authored_ground_geometry_v1.json',model,margin_m=0.)
            reference=CausalFrozenLegReferenceFactory(controller.teacher,controller.parameters,torch.tensor(obs[start]),default,limits,
                controller.event_time,controller.target_offset_world,lane,controller.physics)()
            rows=[];stopped=None
            for node in range(steps):
                legs=reference(deepcopy(state),node)
                if mode=='historical_actual18':target=interface.limits[:,0]+.5*(actions[start+node,lane].clip(-1,1)+1)*np.diff(interface.limits,axis=1)[:,0]
                elif mode=='frozen_base18':target=reference.last_proposal_target.copy()
                else:target=np.r_[legs,arm[node]]
                row=dict(node=node,target_rad=target.tolist(),previous_observation_source='actual previous final18 at start, then committed final18')
                trace=[]
                try:result=predict_unilateral_support(model,interface,state,target,duration_s=.01,substeps=a.substeps,enforce_joint_stops=a.joint_stops,iteration_trace=trace)
                except (ValueError,np.linalg.LinAlgError) as error:
                    row['error']=str(error);row['solver_trace']=trace;row['initial_q']=state.q.tolist();row['initial_v']=state.v.tolist();rows.append(row);stopped='solver';break
                reference.commit_final_command(target,node);state=result['state'];violations=ContactFirstFullBodyEvaluatorV1._violations(result['diagnostics']);witness=geometry.minimum_witness(state.q)
                if witness['gap_m'] < -1e-6:violations['nonfoot_ground_violation_m']=-witness['gap_m']
                exceed=np.maximum(interface.limits[:,0]-state.joint_position,state.joint_position-interface.limits[:,1]);j=int(exceed.argmax())
                truth=model.state_from_public(obs[start+node+1,lane],default.numpy())
                speed_excess=abs(state.joint_velocity)-interface.velocity
                speed_joint=int(speed_excess.argmax())
                row.update(velocity_witness=dict(joint=model.joint_order[speed_joint],
                    predicted_rad_s=float(state.joint_velocity[speed_joint]),
                    native_rad_s=float(truth.joint_velocity[speed_joint]),
                    limit_rad_s=float(interface.velocity[speed_joint]),
                    maximum_violation_rad_s=float(max(0,speed_excess[speed_joint]))),
                    diagnostics=result['diagnostics'],
                    root_position_error_m=float(np.linalg.norm(state.root_position_world-truth.root_position_world)),
                    foot_forces_world_n=result['foot_forces_world_n'].tolist(),
                    violations=violations,joint=model.joint_order[j],predicted_joint_rad=float(state.joint_position[j]),
                    native_joint_rad=float(truth.joint_position[j]),limit_rad=interface.limits[j].tolist(),
                    joint_prediction_error_rad=float(state.joint_position[j]-truth.joint_position[j]),
                    maximum_violation_rad=float(max(0,exceed[j])),ground_witness=witness)
                rows.append(row)
                if violations:
                    row['solver_trace']=trace
                    stopped='original_hard_gate';break
            runs.append(dict(mode=mode,start_tick=start,lane=lane,requested_steps=steps,checked_steps=len(rows),stop=stopped,rows=rows))
    a.output.write_text(json.dumps(dict(scope='Paired offline model check, not native counterfactual',joint_stops=a.joint_stops,substeps=a.substeps,integration_dt_s=.01/a.substeps,physx_equivalent=False,runs=runs),indent=2)+'\n')
    print(json.dumps([dict(mode=r['mode'],steps=r['checked_steps'],stop=r['stop'],last=r['rows'][-1].get('violations')) for r in runs]))
if __name__=='__main__':main()
