"""Measured-state one-step predictor errors with aligned contact-label diagnostics.

Replay actual dispatched commands only. This does not alter the predictor,
native contact observations, acceptance tolerances, or any queued controller.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pinocchio as pin

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.parallel_landing_solver_v1 import build_controller
from scripts.audit_native_frictional_recovery_v1 import state_at
from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance
from scripts.audit_sphere_touchdown_predictions_v1 import sphere_gaps
from scripts import predict_frictional_spheres_v1 as prediction


def summarize(rows,model):
    joint=np.asarray([r['joint_velocity_error_rad_s'] for r in rows])
    qerr=np.asarray([r['joint_position_error_rad'] for r in rows])
    rmse=np.sqrt(np.mean(joint**2,axis=0))
    return dict(transitions=len(rows),
        joint_velocity_rmse_rad_s=dict(zip(model.joint_order,rmse.tolist())),
        joint_position_rmse_rad=dict(zip(model.joint_order,np.sqrt(np.mean(qerr**2,axis=0)).tolist())),
        largest_joint_velocity_error_rad_s=float(np.max(abs(joint))),
        worst_joint_by_rmse=model.joint_order[int(np.argmax(rmse))],
        root_linear_velocity_rmse_m_s=float(np.sqrt(np.mean(np.asarray([r['root_linear_velocity_world_error_m_s'] for r in rows])**2))),
        root_angular_velocity_rmse_rad_s=float(np.sqrt(np.mean(np.asarray([r['root_angular_velocity_world_error_rad_s'] for r in rows])**2))),
        raw_contact_mismatching_feet=sum(r['raw_contact_mismatching_feet'] for r in rows),
        force_hysteresis_mismatching_feet=sum(r['force_hysteresis_mismatching_feet'] for r in rows),
        contact_comparisons=4*len(rows))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['run','bundle','sphere-audit','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve earlier prediction audit')
    result=json.loads((a.run/'sequence_native_result.json').read_text())
    job=json.loads((a.run/'job.json').read_text())
    if job['status']!='SUCCEEDED' or job['exit_code']!=0:raise ValueError('Completed raw native job required')
    manifest=json.loads((a.bundle/'manifest.json').read_text())
    c=build_controller(manifest['configuration']);m=c.qp.model
    geometry=AuthoredGroundClearance(ROOT/'configs/go2_piper_authored_ground_geometry_v1.json',m,margin_m=0.)
    spheres=json.loads(a.sphere_audit.read_text())['spheres']
    with np.load(a.run/'sequence_native_trace.npz',allow_pickle=False) as z:
        trace={k:z[k].copy() for k in z.files}
    metadata=json.loads((a.run/'contact_telemetry_metadata.json').read_text())
    if metadata['physics_dt_s']!=.01/160:raise ValueError('Unexpected native substep duration')
    sensor_root=Path('/home/autodrive/Isaac/IsaacLab/source/isaaclab/isaaclab/sensors/contact_sensor')
    if 'The net normal contact forces in world frame.' not in (sensor_root/'contact_sensor_data.py').read_text():
        raise ValueError('Review installed sensor force semantics before comparison')
    files=[Path(__file__),a.bundle/'manifest.json',a.sphere_audit,
           sensor_root/'contact_sensor_data.py',sensor_root/'contact_sensor.py',
           ROOT/'quadsmash/tasks/stand/coupled_env.py',ROOT/'configs/go2_piper_authored_ground_geometry_v1.json']
    files += [a.run/name for name in ['sequence_native_result.json','sequence_native_trace.npz','job.json','contact_telemetry_metadata.json']]
    files += [Path(v.__file__) for k,v in list(sys.modules.items()) if k.startswith('scripts.') and getattr(v,'__file__',None)]
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    hashes={str(p.resolve()):sha(p) for p in files}
    original=prediction.frictional_velocity_step;groups=[]
    for substeps in [16,160]:
        rows=[]
        for index,record in enumerate(result['records'][:-1]):
            if record['mode']!='frictional_receding_support_sequence':continue
            tick=result['start_tick']+index
            state=state_at(m,trace,index);actual=state_at(m,trace,index+1)
            target=trace['final_target_rad'][index].copy()
            impulse_sum=np.zeros((4,3));last_force=None;observed=state.contacts.copy();peak_torque=np.zeros(18)
            def observe(*args,**kwargs):
                nonlocal impulse_sum,last_force,observed,peak_torque
                answer=original(*args,**kwargs)
                impulse=answer['impulse'].reshape(4,3)
                impulse_sum+=impulse;last_force=impulse/args[8]
                # The installed native sensor reports normal force only. This
                # model has a horizontal ground plane, so compare its normal
                # component, not its full Coulomb force vector.
                normal=np.abs(last_force[:,2])
                observed=np.where(observed,normal>.5,normal>1.)
                peak_torque=np.maximum(peak_torque,abs(answer['torque']))
                return answer
            prediction.frictional_velocity_step=observe
            try:
                item=prediction.predict_frictional_spheres(m,c.qp.interface,state,target,spheres,substeps=substeps,geometry=geometry)
            finally:prediction.frictional_velocity_step=original
            predicted=item['state'];dq=pin.difference(m.model,actual.q,predicted.q)
            rp=pin.Quaternion(predicted.q[3:7]).matrix();ra=pin.Quaternion(actual.q[3:7]).matrix()
            rows.append(dict(tick=tick,next_tick=tick+1,
                joint_position_error_rad=dq[m.v_indices].tolist(),
                joint_velocity_error_rad_s=(predicted.v[m.v_indices]-actual.v[m.v_indices]).tolist(),
                root_position_error_m=(predicted.q[:3]-actual.q[:3]).tolist(),
                root_rotation_tangent_error_rad=dq[3:6].tolist(),
                root_linear_velocity_world_error_m_s=(rp@predicted.v[:3]-ra@actual.v[:3]).tolist(),
                root_angular_velocity_world_error_rad_s=(rp@predicted.v[3:6]-ra@actual.v[3:6]).tolist(),
                native_contacts=actual.contacts.tolist(),predicted_impulse_contacts=predicted.contacts.tolist(),
                predicted_force_hysteresis_contacts=observed.tolist(),
                raw_contact_mismatching_feet=int(np.sum(predicted.contacts!=actual.contacts)),
                force_hysteresis_mismatching_feet=int(np.sum(observed!=actual.contacts)),
                native_next_force_world_n=trace['foot_force_world_n'][index+1].tolist(),
                predicted_last_normal_force_n=last_force[:,2].tolist(),
                predicted_last_force_world_n=last_force.tolist(),predicted_mean_force_world_n=(impulse_sum/.01).tolist(),
                native_next_sphere_gaps_m=sphere_gaps(m,actual.q,spheres).tolist(),
                predicted_next_sphere_gaps_m=sphere_gaps(m,predicted.q,spheres).tolist(),
                model_peak_motor_torque_nm=peak_torque.tolist(),diagnostics=item['diagnostics']))
        groups.append(dict(substeps=substeps,rows=rows,summary=summarize(rows,m)))
    for path,digest in hashes.items():
        if sha(path)!=digest:raise ValueError('Input changed '+path)
    out=dict(scope=__doc__,groups=groups,source_sha256=hashes,case=result['case'],
        native_rerun=False,existing_strict_native_audit_promoted=False,valid_expert_labels=False,
        contact_semantics='Raw model impulse>1e-8 Ns versus native NORMAL force-norm 1/0.5N hysteresis; aligned labels are observational only',
        force_comparison_caveat='Installed native sensor reports sampled normal forces, not total force or interval-integrated impulse. Do not infer zero native friction from zero XY or compare native normal force against full model Coulomb force.')
    a.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps([dict(substeps=g['substeps'],summary=g['summary']) for g in groups]))


if __name__=='__main__':main()
