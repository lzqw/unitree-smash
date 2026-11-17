"""Offline recorded-command replay and independent frozen-base action comparison."""
import argparse,json
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from quadsmash.control.full_racket_rally_controller_v4 import FullRacketRallyControllerV4
from scripts.causal_frozen_leg_prefix_v1 import CausalFrozenLegReferenceFactory
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.unilateral_implicit_prediction_v1 import predict_unilateral_support
from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance
from scripts.contact_first_fullbody_evaluator_v1 import ContactFirstFullBodyEvaluatorV1
from scipy.spatial.transform import Rotation


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve evidence')
    root=Path(__file__).resolve().parents[1];obs=np.load(a.input/'public_observations.npy');actions=np.load(a.input/'public_actions.npy')
    cfg=json.loads((root/'reports/codex_teacher_joint_eval_20261116/controller.json').read_text())['interface'];interface=PositionPDInterface(**cfg)
    model=LoadedPinocchioModelV1();authored=json.loads((root/'reports/G07_full_racket_v4/CPU_REPLAY_AUTHORED_JOINTS_0724.json').read_text())['defaults_before_neutral_arm_override']
    default=np.array([authored[n] for n in model.joint_order]);default[12:]=json.loads((root/'configs/full_racket_reset_v4.json').read_text())['arm_joint_positions_rad']
    default=torch.tensor(default,dtype=torch.float32);limits=torch.tensor(interface.limits,dtype=torch.float32)
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    runs=[]
    with torch.inference_mode():
        for start in (200,216):
            controller=FullRacketRallyControllerV4(checkpoint,obs.shape[1],'cpu')
            controller.contact_first_planner=None  # historical prefix had no adopted planner actions
            for tick in range(start):controller.action({'policy':torch.tensor(obs[tick])},default,limits)
            for lane in range(obs.shape[1]):
                state=model.state_from_public(obs[start,lane],default.numpy())
                reference=CausalFrozenLegReferenceFactory(controller.teacher,controller.parameters,torch.tensor(obs[start]),
                    default,limits,controller.event_time,controller.target_offset_world,lane,controller.physics)()
                native_teacher=deepcopy(controller.teacher)
                geometry=AuthoredGroundClearance(root/'configs/go2_piper_authored_ground_geometry_v1.json',model,margin_m=0.)
                initial_witness=geometry.minimum_witness(state.q)
                rows=[];failure=None
                for node in range(80):
                    tick=start+node
                    proposed=reference(deepcopy(state),node)
                    action=native_teacher.action({'policy':torch.tensor(obs[tick])},controller.parameters,default,limits,
                        event_time=controller.event_time,target_offset_world=controller.target_offset_world)
                    native_target=limits[:,0]+.5*(action[lane].clamp(-1,1)+1)*(limits[:,1]-limits[:,0])
                    target=interface.limits[:,0]+.5*(np.clip(actions[tick,lane],-1,1)+1)*np.diff(interface.limits,axis=1)[:,0]
                    row=dict(node=node,leg_proposal_difference_rad=float(np.max(np.abs(proposed-native_target[:12].numpy()))),
                        command_history_roundtrip_rad=float(np.max(np.abs((limits[:,0]+.5*(torch.tensor(actions[tick,lane])+1)*(limits[:,1]-limits[:,0])).numpy()-target))))
                    try:result=predict_unilateral_support(model,interface,state,target,duration_s=.01,substeps=4)
                    except (ValueError,np.linalg.LinAlgError) as error:
                        row['failure']=str(error);rows.append(row);failure='prediction_solver';break
                    reference.commit_final_command(target,node)
                    state=result['state'];truth=model.state_from_public(obs[tick+1,lane],default.numpy())
                    gap=geometry.minimum_witness(state.q)
                    row.update(root_position_error_m=float(np.linalg.norm(state.root_position_world-truth.root_position_world)),
                        root_rotation_error_rad=float(np.linalg.norm(Rotation.from_matrix(truth.root_rotation_world.T@state.root_rotation_world).as_rotvec())),
                        joint_position_error_rad=float(np.max(np.abs(state.joint_position-truth.joint_position))),
                        joint_velocity_error_rad_s=float(np.max(np.abs(state.joint_velocity-truth.joint_velocity))),
                        contact_equal=bool(np.array_equal(state.contacts,truth.contacts)),ground_witness=gap)
                    violations=ContactFirstFullBodyEvaluatorV1._violations(result['diagnostics'])
                    if gap['gap_m'] < -1e-6:violations['nonfoot_ground_violation_m']=-gap['gap_m']
                    row['violations']=violations;rows.append(row)
                    if violations:failure='original_physical_gate';break
                runs.append(dict(start_tick=start,lane=lane,requested_nodes=80,initial_witness=initial_witness,recorded_nodes=len(rows),failure=failure,rows=rows))
    a.output.write_text(json.dumps(dict(scope='Offline only: recorded future commands and observations are diagnostic truth, never online planner inputs',
        replay_physical_gate='stop at first original violation; no tolerance changes',runs=runs),indent=2)+'\n')
    print(json.dumps([dict(start=r['start_tick'],lane=r['lane'],nodes=r['recorded_nodes'],failure=r['failure']) for r in runs]))
if __name__=='__main__':main()
