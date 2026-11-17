"""Replay fixed CPU commands, attribute wrench error without changing control."""
import argparse,json
from pathlib import Path
import numpy as np
import pinocchio as pin
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1,PublicState
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance
from scripts.experimental_sphere_admission_v1 import ExperimentalSphereAdmissionV1,admission_diagnostics
from scripts.predict_speed_limited_spheres_v1 import predict_speed_limited_spheres


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve attempts')
    root=Path(__file__).resolve().parents[1]
    initial=json.loads((root/'reports/push_off_task_20261118/DIAGNOSED.json').read_text())['rows'][0]
    trace=json.loads((root/'reports/momentum_push_20261118/RESULTS.json').read_text())['results'][0]['trajectory']
    model=LoadedPinocchioModelV1();q=np.array(initial['initial_q']);v=np.array(initial['initial_v'])
    state=PublicState(q.copy(),v.copy(),pin.Quaternion(q[3:7]).matrix(),q[:3].copy(),q[model.q_indices].copy(),v[model.v_indices].copy(),np.array(initial['baseline_initial_contacts'],bool),0.)
    state.previous_command_target=np.array(initial['previous_final_command'])
    interface=PositionPDInterface(**json.loads((root/'reports/codex_teacher_joint_eval_20261116/controller.json').read_text())['interface'])
    geometry=AuthoredGroundClearance(root/'configs/go2_piper_authored_ground_geometry_v1.json',model,margin_m=0.)
    adapter=ExperimentalSphereAdmissionV1(model,geometry,root/'reports/prediction_model_validation_20261118/AUTHORED_SPHERES.json')
    rows=[]
    for old in trace:
        before=model.snapshot(state.q,state.v)
        result=predict_speed_limited_spheres(model,interface,state,np.array(old['final_target_rad']),adapter.spheres,
            substeps=160,geometry=geometry,enforce_joint_stops=True,diagnostic_nonfoot_spheres=adapter.extras,
            diagnostic_hip_mount_response=adapter.hip,contact_max_iterations=adapter.iterations,audit_wrench=True)
        admission_diagnostics(result,.01/160)
        following=result['state'];w=result['wrench_audit'];after=model.snapshot(following.q,following.v)
        eq=float(max(np.max(abs(following.q-old['q'])),np.max(abs(following.v-old['v']))))
        if eq!=0:raise ValueError('Read-only audit changed replayed state')
        actual=(after['momentum_world'][3:]-before['momentum_world'][3:])/.01
        moment=w['contact_angular_impulse_com_world_nms'].sum(0)/.01
        row=dict(node=old['node'],state_difference=eq,nonlinear_Hdot=actual,integrated_contact_moment=moment,
            integration_balance_error_Nm=actual-moment,nonlinear_mean_force=w['contact_impulse_world_ns'][:4]/.01,
            mean_drive_torque=w['drive_torque_integral_nms']/.01,saturated_substeps=w['saturated_substeps'],
            contact_events=result['events'],qp_accepted=old['qp']['accepted'])
        if old['qp']['accepted']:
            forces=np.array(old['qp']['force_world'])
            positions=np.array([before['body_positions_world'][model.names.index(n)] for n in model.feet])
            # Same forces with body-origin versus authored bottom-point lever arms.
            points=np.array([positions[i]+before['body_rotations_world'][model.names.index(s['body'])]@np.array(s['center_body_m'])-np.array([0,0,s['radius_m']]) for i,s in enumerate(adapter.spheres)])
            body_moment=np.cross(positions-before['com_world'],forces).sum(0)
            surface_moment=np.cross(points-before['com_world'],forces).sum(0)
            row.update(qp_force=forces,qp_body_moment=body_moment,qp_same_force_surface_moment=surface_moment,
                pure_lever_arm_difference_Nm=surface_moment-body_moment,
                force_time_response_remainder_Nm=moment-surface_moment,
                qp_drive_torque=old['qp']['torque'])
        rows.append(row);state=following
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(scope='fixed CPU command replay, no native evidence',rows=rows),indent=2,default=lambda x:x.tolist())+'\n')
    print(json.dumps(dict(steps=len(rows),max_balance_error_Nm=max(np.linalg.norm(r['integration_balance_error_Nm']) for r in rows),
        max_lever_arm_delta_Nm=max(np.linalg.norm(r['pure_lever_arm_difference_Nm']) for r in rows if r['qp_accepted']),
        max_force_response_remainder_Nm=max(np.linalg.norm(r['force_time_response_remainder_Nm']) for r in rows if r['qp_accepted']))))
if __name__=='__main__':main()
