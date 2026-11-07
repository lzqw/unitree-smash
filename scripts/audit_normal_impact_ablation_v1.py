"""Matched command/state comparison of fixed mode and normal-impact prediction.

No native controller change. Includes every applied pre-recorded-nonfoot sample
from the sphere audit, not only selected successful error reductions.
"""
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
from scripts.predict_sphere_normal_impact_v1 import predict_sphere_normal_impact
from scripts.summarize_precontact_prediction_v1 import stats


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['run','sphere-audit','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve prior evidence')
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    spheres=json.loads(a.sphere_audit.read_text());paths=[Path(__file__),a.sphere_audit,
        ROOT/'scripts/predict_sphere_normal_impact_v1.py',ROOT/'scripts/implicit_support_prediction_v1.py',
        a.run/'landing_mpc_records.json',a.run/'native_command_inputs.json']
    if Path(spheres['run']).resolve()!=a.run.resolve():raise ValueError('Different native run')
    hashes={str(f.resolve()):sha(f) for f in paths}
    for f,digest in spheres['source_sha256'].items():
        if sha(f)!=digest:raise ValueError('Changed sphere evidence '+f)
    native=json.loads((a.run/'native_command_inputs.json').read_text());model=LoadedPinocchioModelV1()
    interface=PositionPDInterface(native['joint_limits_rad'],np.r_[np.full(12,25.),np.full(6,80.)],
        np.r_[np.full(12,.5),np.full(6,4.)],np.r_[np.full(12,23.5),np.full(6,100.)],
        np.r_[np.full(12,30.),np.full(6,5.)],.65)
    records=json.loads((a.run/'landing_mpc_records.json').read_text())
    records={(r['case'],r['control_tick']):r for r in records if r.get('episode_index')==0}
    cache={};rows=[];started=time.perf_counter()
    for reference in spheres['records']:
        case,tick=reference['case'],reference['tick'];record=records[(case,tick)]
        if case not in cache:
            path=a.run/'fixed_policy_traces'/f'case_{case:03d}.npz';hashes[str(path.resolve())]=sha(path)
            with np.load(path) as z:cache[case]=unpack_public(z['packed_public'])
        raw=cache[case];state=model.state_from_public(raw[tick],native['default_joint_position_rad'])
        actual=model.state_from_public(raw[tick+1],native['default_joint_position_rad'])
        state.contacts=np.asarray(record['support_contacts'],bool)
        row=dict(case=case,tick=tick,full_horizon_accepted=reference['full_horizon_accepted'],
            observed_contact_change=reference['observed_contact_change'],
            actual_sphere_crossing=any(reference['geometric_crossing_by_tolerance']['0.0001']),
            original_fixed_error_rad=reference['leg_error_rad'],models={})
        variants=[('refreshed_fixed',16,False),('normal_impact16',16,True)]
        if row['actual_sphere_crossing']:variants.append(('normal_impact160',160,True))
        for name,steps,enabled in variants:
            try:
                result=predict_sphere_normal_impact(model,interface,state,
                    record['final_dispatched_joint_target_rad'],spheres['spheres'],substeps=steps,detect_impacts=enabled)
                pred=result['state']
                row['models'][name]=dict(ok=True,leg_error_rad=float(np.max(abs(pred.joint_position[:12]-actual.joint_position[:12]))),
                    arm_error_rad=float(np.max(abs(pred.joint_position[12:]-actual.joint_position[12:]))),
                    root_position_error_m=float(np.linalg.norm(pred.q[:3]-actual.q[:3])),
                    diagnostics=result['diagnostics'],events=result['events'])
            except (ValueError,np.linalg.LinAlgError) as e:
                row['models'][name]=dict(ok=False,error=repr(e))
        rows.append(row)
    groups={}
    for label,selected in [('all',rows),('sphere_crossing',[r for r in rows if r['actual_sphere_crossing']]),
                          ('no_sphere_crossing',[r for r in rows if not r['actual_sphere_crossing']]),
                          ('accepted_horizon',[r for r in rows if r['full_horizon_accepted']])]:
        group=dict(samples=len(selected),models={})
        for name in ['refreshed_fixed','normal_impact16','normal_impact160']:
            available=[r['models'][name] for r in selected if name in r['models']]
            good=[r for r in available if r['ok']]
            group['models'][name]=dict(attempted=len(available),failed=len(available)-len(good),
                leg_error_rad=stats([r['leg_error_rad'] for r in good]),
                detected_impact_samples=sum(any(e['kind']=='normal_impact' for e in r['events']) for r in good),
                old_contact_force_invalid_samples=sum(r['diagnostics']['old_normal_violation_n']>3e-4 or
                    r['diagnostics']['old_friction_violation_n']>3e-4 for r in good))
        paired=[r for r in selected if all(r['models'][n]['ok'] for n in ['refreshed_fixed','normal_impact16'])]
        group['paired_error_improved']=sum(r['models']['normal_impact16']['leg_error_rad']<
            r['models']['refreshed_fixed']['leg_error_rad']-1e-6 for r in paired)
        group['paired_error_worsened']=sum(r['models']['normal_impact16']['leg_error_rad']>
            r['models']['refreshed_fixed']['leg_error_rad']+1e-6 for r in paired)
        groups[label]=group
    for f,digest in hashes.items():
        if sha(f)!=digest:raise ValueError('Changed audit input during execution')
    result=dict(scope=__doc__,native_execution=False,student_updated=False,valid_teacher=False,
        run=str(a.run.resolve()),wall_seconds=time.perf_counter()-started,groups=groups,records=rows,
        source_sha256=hashes,limits=['New contacts have normal force only; no full tangential friction dynamics',
            'Existing contacts retain old support law and may violate unilateral/friction limits, reported explicitly',
            'No nonfoot or self-collision impulses', 'Prediction improvement is not native recovery improvement'])
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(groups))


if __name__=='__main__':main()
