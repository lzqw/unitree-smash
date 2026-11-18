"""Conditional analytic goal projection against one complete CPU prefix.

Uses future CPU state for OFFLINE diagnosis only, never future native data.
Does not install a new target or authorize the prefix.
"""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import pinocchio as pin
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1
from scripts.contact_terminal_attribution_v1 import contact_kinematics
from scripts.full_prefix_tracking_v1 import arm_tracking_response
from quadsmash.control.task_goal_projection_v1 import project_task_goal
from quadsmash.control.impact_outcome_evaluator_v1 import ImpactOutcomeEvaluator
from quadsmash.physics.full_racket_v4 import load_racket


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise ValueError('Preserve prior evidence')
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if digest(args.prefix)!=args.prefix.with_suffix('.sha256').read_text().strip():
        raise ValueError('Prefix hash mismatch')
    record=json.loads(args.prefix.read_text())
    for path,expected in record['input_sha256'].items():
        if digest(path)!=expected: raise ValueError('Input drift: '+path)
    initial_path=Path('reports/push_off_task_20261118/DIAGNOSED.json')
    if str(initial_path) not in record['input_sha256']: raise ValueError('Missing initial state provenance')
    initial=json.loads(initial_path.read_text())['rows'][0]
    meta=record['metadata']; selected=record['trials'][record['selected_index']]['result']
    model=LoadedPinocchioModelV1()
    arm_tracking_response(selected,model.q_indices[12:],meta['contact_index'])
    final=selected['trajectory'][-1]
    q0=np.asarray(initial['initial_q']); rotation=pin.Quaternion(q0[3:7]).matrix()
    ball=q0[:3]+rotation@np.asarray(meta['target_position_world_m'])
    incoming=rotation@np.asarray(meta['incoming_velocity_planning'])
    normal=rotation@np.asarray(meta['target_normal_world'])
    velocity=rotation@np.asarray(meta['retargeted_velocity_current_frame_m_s'])
    offset=np.asarray(meta['contact_point_local']).copy(); offset[0]+=meta['task_physics']['collision_radius_m']
    actual_p,actual_n,jacobian,_=contact_kinematics(model,final['q'],offset)
    actual_v=jacobian@np.asarray(final['v'])
    ep=float(np.linalg.norm(actual_p-ball))
    if abs(ep-selected['terminal_errors']['position_m'])>1e-10: raise ValueError('Terminal FK mismatch')
    impact=ImpactOutcomeEvaluator(meta['task_physics'],meta['task_court'],load_racket())
    start=time.perf_counter()
    result=project_task_goal(impact,ball,incoming,meta['contact_point_local'],normal,velocity,actual_n,actual_v)
    result.update(seconds=time.perf_counter()-start,prefix=str(args.prefix),prefix_sha256=digest(args.prefix),
        input_sha256=record['input_sha256'], actual_position_error_m=ep,
        actual_contact_position_world_m=actual_p, actual_normal_world=actual_n,
        actual_velocity_world_m_s=actual_v, ball_world_m=ball,incoming_world_m_s=incoming,
        contact_time_s=meta['contact_time_s'], future_native_states_read=False,
        control_improved=False, target_installed=False, FULLBODY_PREDICTION_VALID=False,
        NATIVE_HIT='NOT_RUN', PHYSX_REFERENCE_VALID='NOT_RUN',
        evaluation_stage='teacher_debug',formal_eligible=False,formal_acceptance=False,formal_cases_read=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False,
        default=lambda x:x.tolist() if isinstance(x,np.ndarray) else float(x))+'\n')
    args.output.with_suffix('.sha256').write_text(digest(args.output)+'\n')
    print(json.dumps(dict(improved=result['improved'],original_errors=result['original']['reference_errors'].tolist(),
        projected_errors=result['selected']['reference_errors'].tolist(),position_error_m=ep,
        gamma=result['selected']['task_margins']['gamma'],queries=result['queries'],seconds=result['seconds'])))

if __name__=='__main__':main()
