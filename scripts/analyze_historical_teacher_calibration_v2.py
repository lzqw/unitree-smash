"""Compare native history replay and CPU one-step/open-loop predictions."""
import argparse,json
from pathlib import Path
import numpy as np
import pinocchio as pin
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1,PublicState,rotation_wxyz
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.unilateral_implicit_prediction_v1 import predict_unilateral_support


def main():
    p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True);p.add_argument('--interface',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve previous calibration analysis')
    completion=json.loads((a.native/'CALIBRATION_COMPLETE.json').read_text())
    if completion['ticks']!=800:raise ValueError('Complete from-reset native replay required')
    z=np.load(a.native/'control_state.npz',allow_pickle=False);model=LoadedPinocchioModelV1();interface=PositionPDInterface(**json.loads(a.interface.read_text())['interface'])
    def state(t,lane):
        rotation=rotation_wxyz(z['root_quaternion'][t,lane]);q=pin.neutral(model.model);v=np.zeros(model.model.nv)
        q[:3]=z['root_position'][t,lane];q[3:7]=pin.Quaternion(rotation).coeffs()
        q[model.q_indices]=z['joint_position'][t,lane];v[model.v_indices]=z['joint_velocity'][t,lane]
        # The first6 public channels are actual root LINK twist in body frame,
        # avoiding an unrecorded COM-offset approximation.
        v[:6]=z['public'][t,lane,:6]
        s=PublicState(q,v,rotation,q[:3].copy(),q[model.q_indices].copy(),v[model.v_indices].copy(),z['contacts'][t,lane].copy(),0.)
        s.previous_command_target=z['previous_target'][t,lane].copy();s.measured_force_world=z['foot_force'][t,lane].copy()
        return s
    runs=[]
    for lane in range(2):
        for mode in ('one_step_from_native','open_loop_from_native249'):
            s=state(249,lane);rows=[]
            for tick in range(249,263):
                if mode=='one_step_from_native':s=state(tick,lane)
                target=z['previous_target'][tick+1,lane];trace=[]
                row=dict(tick=tick,native_velocity_before=z['joint_velocity'][tick,lane].tolist(),native_velocity_after=z['joint_velocity'][tick+1,lane].tolist(),actuator_estimated_torque=z['applied_drive_torque'][tick,lane].tolist(),native_foot_force=z['foot_force'][tick,lane].tolist(),native_contacts=z['contacts'][tick,lane].tolist(),command=target.tolist())
                try:
                    result=predict_unilateral_support(model,interface,s,target,duration_s=.01,substeps=160,enforce_joint_stops=True,iteration_trace=trace)
                    s=result['state'];s.previous_command_target=target.copy();truth=state(tick+1,lane)
                    row.update(prediction_completed=True,diagnostics=result['diagnostics'],predicted_velocity=s.joint_velocity.tolist(),
                        max_velocity_error_rad_s=float(max(abs(s.joint_velocity-truth.joint_velocity))),root_position_error_m=float(np.linalg.norm(s.q[:3]-truth.q[:3])),
                        predicted_force=result['foot_forces_world_n'].tolist())
                except (ValueError,np.linalg.LinAlgError) as e:
                    row.update(prediction_completed=False,error=str(e),status='MODEL_UNCERTAIN',solver_trace=trace)
                rows.append(row)
                if not row['prediction_completed'] and mode.startswith('open_loop'):break
            runs.append(dict(lane=lane,mode=mode,rows=rows))
    a.output.write_text(json.dumps(dict(native_completed=True,native_policy_evaluation=False,control_dt_s=.01,prediction_substeps=160,
        joint_order=model.joint_order,physx_equivalent=False,
        torque_semantics='IsaacLab ImplicitActuator approximation; not actual PhysX drive torque',
        raw_applied_drive_torque_key_is_estimate=True,runs=runs),indent=2))
    print(json.dumps([dict(lane=r['lane'],mode=r['mode'],completed=sum(x['prediction_completed'] for x in r['rows']),attempted=len(r['rows'])) for r in runs]))
if __name__=='__main__':main()
