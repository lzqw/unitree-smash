"""One causal late-contact diagnostic; no optimization or teacher installation."""
import argparse,copy,gzip,hashlib,json
from pathlib import Path
import numpy as np,pinocchio as pin
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1,PublicState
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance
from scripts.experimental_sphere_admission_v1 import ExperimentalSphereAdmissionV1
from scripts.terminal_push_refinement_v1 import TerminalPushRollout
from scripts.contact_terminal_attribution_v1 import contact_kinematics
from quadsmash.control.initial_intercept import observed_initial_ball,propagate_ball_state

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prefix',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    record=json.loads(args.prefix.read_text());selected=record['trials'][record['selected_index']]['result'];initial=json.loads(Path('reports/push_off_task_20261118/DIAGNOSED.json').read_text())['rows'][0]
    model=LoadedPinocchioModelV1();q=np.array(initial['initial_q']);v=np.array(initial['initial_v']);state=PublicState(q,v,pin.Quaternion(q[3:7]).matrix(),q[:3].copy(),q[model.q_indices].copy(),v[model.v_indices].copy(),np.array(initial['baseline_initial_contacts'],bool),0.);state.previous_command_target=np.array(initial['previous_final_command'])
    commands=np.array([t['final_target_rad'] for t in selected['trajectory']]);ref=np.array(record['selected_arm_reference_rad']);commands=np.vstack([commands,commands[-1],commands[-1]]);ref=np.vstack([ref,ref[-1],ref[-1]])
    source=json.load(gzip.open('reports/experimental_fullbody_gate_20261118/REPLAY.json.gz','rt'));metadata=copy.deepcopy([x['metadata'] for x in source['physical_calls'] if x['lane']==initial['lane']][0]);metadata.update(contact_index=35,contact_time_s=.35)
    obs=np.load('reports/joint_loop_round_20261117/attempts/probe01/native/public_observations.npy',mmap_mode='r')[round(initial['clock_s']/.01),initial['lane']];p0,v0=observed_initial_ball(obs[None],[1.1,0,1.8],[1,0,0]);ball,incoming=propagate_ball_state(p0[0],v0[0],.35,metadata['task_physics']);R0=pin.Quaternion(q[3:7]).matrix();metadata.update(target_position_world_m=R0.T@(ball-q[:3]),incoming_velocity_planning=R0.T@incoming)
    interface=PositionPDInterface(**json.loads(Path('reports/codex_teacher_joint_eval_20261116/controller.json').read_text())['interface']);geometry=AuthoredGroundClearance('configs/go2_piper_authored_ground_geometry_v1.json',model,margin_m=0.);predictor=ExperimentalSphereAdmissionV1(model,geometry,Path('reports/prediction_model_validation_20261118/AUTHORED_SPHERES.json'));result=TerminalPushRollout(model,interface,predictor,state,commands,ref,metadata)(np.zeros(12));terminal=result['trajectory'][-1];offset=np.array(metadata['contact_point_local']);offset[0]+=metadata['task_physics']['collision_radius_m'];actual,normal,_,_=contact_kinematics(model,np.array(terminal['q']),offset)
    output=dict(path_valid=result['path_valid'],failure=result['failure'],position_error_m=float(np.linalg.norm(actual-ball)),ball_world_m=ball,actual_contact_world_m=actual,contact_time_s=.35,prefix_sha256=hashlib.sha256(args.prefix.read_bytes()).hexdigest(),future_native_states_read=False,native_execution=False)
    args.output.write_text(json.dumps(output,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else float(x))+'\n');print(json.dumps(output,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else float(x)) )
if __name__=='__main__':main()
