"""Evaluate full frictional sphere prediction on the prior 661 native samples."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from quadsmash.rl.episode_trace_v4 import unpack_public
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.predict_frictional_spheres_v1 import predict_frictional_spheres
from scripts.summarize_precontact_prediction_v1 import stats


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['run','sphere-audit','previous-audit','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve evidence')
    sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    sphere=json.loads(a.sphere_audit.read_text());previous=json.loads(a.previous_audit.read_text())
    for data in [sphere,previous]:
        if Path(data['run']).resolve()!=a.run.resolve():raise ValueError('Different native run')
        for f,digest in data['source_sha256'].items():
            if sha(f)!=digest:raise ValueError('Changed prior evidence '+f)
    files=[Path(__file__),a.sphere_audit,a.previous_audit,ROOT/'scripts/frictional_impulse_step_v1.py',
        ROOT/'scripts/predict_frictional_spheres_v1.py',a.run/'native_command_inputs.json',a.run/'landing_mpc_records.json']
    hashes={str(f.resolve()):sha(f) for f in files}
    native=json.loads((a.run/'native_command_inputs.json').read_text());model=LoadedPinocchioModelV1()
    interface=PositionPDInterface(native['joint_limits_rad'],np.r_[np.full(12,25.),np.full(6,80.)],
        np.r_[np.full(12,.5),np.full(6,4.)],np.r_[np.full(12,23.5),np.full(6,100.)],
        np.r_[np.full(12,30.),np.full(6,5.)],.65)
    logs=json.loads((a.run/'landing_mpc_records.json').read_text())
    logs={(r['case'],r['control_tick']):r for r in logs if r.get('episode_index')==0}
    cache={};rows=[];started=time.perf_counter()
    for prior in previous['records']:
        case,tick=prior['case'],prior['tick'];log=logs[(case,tick)]
        if case not in cache:
            path=a.run/'fixed_policy_traces'/f'case_{case:03d}.npz';hashes[str(path.resolve())]=sha(path)
            with np.load(path) as z:cache[case]=unpack_public(z['packed_public'])
        raw=cache[case];state=model.state_from_public(raw[tick],native['default_joint_position_rad'])
        actual=model.state_from_public(raw[tick+1],native['default_joint_position_rad'])
        row={k:prior[k] for k in ['case','tick','full_horizon_accepted','actual_sphere_crossing','observed_contact_change']}
        row['reference_errors']={name:r['leg_error_rad'] for name,r in prior['models'].items() if r['ok']}
        before=time.perf_counter()
        try:
            result=predict_frictional_spheres(model,interface,state,log['final_dispatched_joint_target_rad'],sphere['spheres'])
            predicted=result['state'];row.update(ok=True,leg_error_rad=float(np.max(abs(predicted.joint_position[:12]-actual.joint_position[:12]))),
                arm_error_rad=float(np.max(abs(predicted.joint_position[12:]-actual.joint_position[12:]))),
                root_position_error_m=float(np.linalg.norm(predicted.q[:3]-actual.q[:3])),
                diagnostics=result['diagnostics'],events=result['events'])
        except (ValueError,np.linalg.LinAlgError) as e:row.update(ok=False,error=repr(e))
        row['wall_seconds']=time.perf_counter()-before;rows.append(row)
    groups={}
    for name,subset in [('all',rows),('sphere_crossing',[r for r in rows if r['actual_sphere_crossing']]),
                        ('accepted_horizon',[r for r in rows if r['full_horizon_accepted']])]:
        good=[r for r in subset if r['ok']]
        group=dict(samples=len(subset),failed=len(subset)-len(good),leg_error_rad=stats([r['leg_error_rad'] for r in good]),
            wall_seconds=stats([r['wall_seconds'] for r in subset]),
            maximum_diagnostics={k:max(r['diagnostics'][k] for r in good) for k in good[0]['diagnostics']} if good else {})
        group['paired']={}
        for model_name in ['refreshed_fixed','normal_impact16']:
            paired=[r for r in good if model_name in r['reference_errors']]
            group['paired'][model_name]=dict(samples=len(paired),reference_error_rad=stats([r['reference_errors'][model_name] for r in paired]),
                improved=sum(r['leg_error_rad']<r['reference_errors'][model_name]-1e-6 for r in paired),
                worsened=sum(r['leg_error_rad']>r['reference_errors'][model_name]+1e-6 for r in paired))
        groups[name]=group
    for f,digest in hashes.items():
        if sha(f)!=digest:raise ValueError('Changed input during audit '+f)
    result=dict(scope=__doc__,run=str(a.run.resolve()),native_execution=False,valid_teacher=False,student_updated=False,
        groups=groups,records=rows,source_sha256=hashes,wall_seconds=time.perf_counter()-started,
        limits=['Original hardware and task unchanged; prediction alone does not verify recovery',
            'All foot contacts use sphere-surface friction; no nonfoot/self-collision impulse',
            'Bounded nonconvex contact iteration can fail; failures excluded only with explicit counts',
            'No claim of equivalence to PhysX compliance, contact margins or restitution'])
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(groups))


if __name__=='__main__':main()
