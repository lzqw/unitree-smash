"""Matched-context derivative audit only; no candidate optimizer or native run."""
import argparse,copy,hashlib,json,time
from pathlib import Path
import numpy as np
import pinocchio as pin
from scripts.coupled_contact_prefix_v1 import CoupledContactRollout,LateRateContactRollout
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1,PublicState
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance
from scripts.experimental_sphere_admission_v1 import ExperimentalSphereAdmissionV1


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve evidence')
    r=json.loads(a.source.read_text())
    if hashlib.sha256(a.source.read_bytes()).hexdigest()!=a.source.with_suffix('.sha256').read_text().strip():raise ValueError('Source hash mismatch')
    for path,digest in r['input_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('Input drift')
    if r['rate_window_ticks']!=8:raise ValueError('Expected audited 80ms response')
    base=r['trials'][0]['result'];trace=base['trajectory'];m=r['metadata'];n=m['contact_index']
    if not base['path_valid'] or len(trace)!=n:raise ValueError('Complete baseline required')
    initial=json.loads(Path('reports/push_off_task_20261118/DIAGNOSED.json').read_text())['rows'][0]
    model=LoadedPinocchioModelV1();q=np.array(initial['initial_q']);v=np.array(initial['initial_v'])
    state=PublicState(q,v,pin.Quaternion(q[3:7]).matrix(),q[:3].copy(),q[model.q_indices].copy(),v[model.v_indices].copy(),np.array(initial['baseline_initial_contacts'],bool),0.)
    state.previous_command_target=np.array(initial['previous_final_command'])
    interface=PositionPDInterface(**json.loads(Path('reports/codex_teacher_joint_eval_20261116/controller.json').read_text())['interface'])
    geometry=AuthoredGroundClearance('configs/go2_piper_authored_ground_geometry_v1.json',model,margin_m=0.)
    predictor=ExperimentalSphereAdmissionV1(model,geometry,Path('reports/prediction_model_validation_20261118/AUTHORED_SPHERES.json'))
    args=(model,interface,predictor,state,[s['final_target_rad'] for s in trace],r['base_arm_reference_rad'],m)
    old=CoupledContactRollout(*args);late=LateRateContactRollout(*args)
    started=time.perf_counter();check=old(np.zeros(24))
    if not check['path_valid']:raise ValueError('Baseline replay failed')
    for expected,actual in zip(trace,check['trajectory']):
        if not np.array_equal(expected['q'],actual['q']) or not np.array_equal(expected['v'],actual['v']):raise ValueError('Baseline mismatch')
    y=np.array(base['normalized_residual']);direction=-y[6:]/np.linalg.norm(y[6:]);rows=[];coeff_old=[];coeff_new=[]
    for j in range(18,24):
        matches=[t for t in r['trials'] if t['label']=='derivative_'+str(j)]
        if len(matches)!=1 or not matches[0]['valid']:raise ValueError('Missing complete new derivative')
        new=matches[0];amplitude=np.array(new['amplitude_rad']);step=amplitude[j]
        result=old(amplitude);row=dict(parameter=j,amplitude_rad=amplitude,old_result=result,new_result_label=new['label'])
        if result['path_valid']:
            old_j=(np.array(result['normalized_residual'])-y)/step;new_j=(np.array(new['result']['normalized_residual'])-y)/step
            row.update(old_residual_derivative=old_j,new_residual_derivative=new_j)
            coeff_old.append(float(direction@old_j[6:]*.75));coeff_new.append(float(direction@new_j[6:]*.75))
        rows.append(row);print(json.dumps(dict(parameter=j,path_valid=result['path_valid'],failure=result['failure'])),flush=True)
    summary=None
    if all(row['old_result']['path_valid'] for row in rows):
        bo=old.amplitude_bounds()[18:];bn=late.amplitude_bounds()[18:];rad=old.radii[18:]
        lo=np.maximum(np.maximum(bo[:,0],bn[:,0]),-rad);hi=np.minimum(np.minimum(bo[:,1],bn[:,1]),rad)
        summary=dict(common_parameter_lower_rad=lo,common_parameter_upper_rad=hi,
            old_optimistic_error_direction_projection_m_s=float(np.maximum(np.array(coeff_old)*lo,np.array(coeff_old)*hi).sum()),
            late_optimistic_error_direction_projection_m_s=float(np.maximum(np.array(coeff_new)*lo,np.array(coeff_new)*hi).sum()),
            scope='Same-context local linear projection, ignores coupled path/task constraints. Not a feasible command or global capacity proof.')
    out=dict(source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),input_sha256=r['input_sha256'],seconds=time.perf_counter()-started,
        baseline_exact=True,baseline_errors=base['terminal_errors'],rows=rows,summary=summary,
        model_calls=7,optimization_proposals=0,native_execution=False,valid_teacher_reference=False)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else float(x))+'\n')
    print(json.dumps(summary,default=lambda x:x.tolist()))
if __name__=='__main__':main()
