"""Infer net external impulse from recorded loaded-body endpoint momentum.

This is an endpoint consistency check, not a measured per-foot force history.
Attribution to ground assumes no material additional external impulse during
the selected post-hit, pre-nonfoot-contact interval. No teacher promotion.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pinocchio as pin

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.parallel_landing_solver_v1 import build_controller


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['run','bundle','one-step-audit','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args()
    if a.output.exists():raise ValueError('Preserve earlier evidence')
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    audit=json.loads(a.one_step_audit.read_text())
    for path,digest in audit['source_sha256'].items():
        if sha(path)!=digest:raise ValueError('Changed source '+path)
    manifest=json.loads((a.bundle/'manifest.json').read_text())
    model=build_controller(manifest['configuration']).qp.model
    raw=json.loads((a.run/'sequence_native_result.json').read_text())
    contacts=json.loads((a.run/'contacts.json').read_text())
    events=json.loads((a.run/'first_nonfoot_contact_details.json').read_text())
    meta=json.loads((a.run/'contact_telemetry_metadata.json').read_text())
    reference=next(g for g in audit['groups'] if g['substeps']==160)
    start=min(r['tick'] for r in reference['rows'])*.01
    end=(max(r['tick'] for r in reference['rows'])+1)*.01
    case=raw['case']
    if any(start<=r['time_s']<=end and r['env_id']==case for r in contacts):
        raise ValueError('Interval contains recorded ball contact')
    if any(case in r['first_event_env_ids'] and r['physics_frame']*meta['physics_dt_s']<=end for r in events):
        raise ValueError('Interval includes recorded nonfoot ground contact')
    with np.load(a.run/'sequence_native_trace.npz',allow_pickle=False) as z:
        q=z['q'].copy();v=z['v'].copy()
    gravity=model.model.gravity.linear.copy()
    def momentum(index):
        pin.centerOfMass(model.model,model.data,q[index],v[index])
        return model.mass*model.data.vcom[0].copy()
    rows=[]
    for row in reference['rows']:
        i=row['tick']-raw['start_tick']
        before=momentum(i);after=momentum(i+1)
        # p_dot = m*g + sum(f_external), all in the world frame.
        inferred=(after-before)/.01-model.mass*gravity
        predicted=np.asarray(row['predicted_mean_force_world_n']).sum(axis=0)
        endpoint=np.asarray(row['native_next_force_world_n']).sum(axis=0)
        rows.append(dict(tick=row['tick'],momentum_before_world_ns=before.tolist(),
            momentum_after_world_ns=after.tolist(),inferred_mean_external_force_world_n=inferred.tolist(),
            model_mean_ground_force_world_n=predicted.tolist(),
            native_endpoint_normal_force_world_n=endpoint.tolist(),
            model_mean_minus_inferred_world_n=(predicted-inferred).tolist(),
            endpoint_normal_minus_inferred_z_n=float(endpoint[2]-inferred[2])))
    error=np.asarray([r['model_mean_minus_inferred_world_n'] for r in rows])
    endpoint=np.asarray([r['endpoint_normal_minus_inferred_z_n'] for r in rows])
    files=[Path(__file__),a.one_step_audit,a.bundle/'manifest.json']+[a.run/n for n in ['contacts.json','first_nonfoot_contact_details.json','contact_telemetry_metadata.json']]
    hashes=dict(audit['source_sha256']);hashes.update({str(p.resolve()):sha(p) for p in files})
    result=dict(scope=__doc__,case=case,interval_s=[start,end],mass_kg=model.mass,
        gravity_world_m_s2=gravity.tolist(),rows=rows,
        model_mean_force_rmse_xyz_n=np.sqrt(np.mean(error**2,axis=0)).tolist(),
        model_mean_force_max_abs_xyz_n=np.max(abs(error),axis=0).tolist(),
        sampled_endpoint_normal_vs_interval_z_rmse_n=float(np.sqrt(np.mean(endpoint**2))),
        sampled_endpoint_normal_vs_interval_z_max_abs_n=float(np.max(abs(endpoint))),
        source_sha256=hashes,actual_foot_force_distribution_known=False,
        recorded_ball_contact_in_interval=False,recorded_nonfoot_contact_before_end=False,
        native_rerun=False,valid_teacher=False,valid_expert_labels=False)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['model_mean_force_rmse_xyz_n','model_mean_force_max_abs_xyz_n','sampled_endpoint_normal_vs_interval_z_rmse_n']}))


if __name__=='__main__':main()
