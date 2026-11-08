"""Distinguish endpoint force samples from recent native support impulse.

This is retrospective evidence, not a replacement for current contact facts.
Past impulse alone does not authorize a ground constraint on an airborne foot.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pinocchio as pin

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.pure_control_reference_clock_v12 import PureControlReferenceClockV12
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.audit_sphere_touchdown_predictions_v1 import sphere_gaps


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['run','configuration','prediction-audit','output']:
        p.add_argument('--'+n,type=Path,required=True)
    args=p.parse_args();read=lambda f:json.loads(Path(f).read_text())
    if args.output.exists():raise ValueError('Fresh audit output required')
    prediction=read(args.prediction_audit)
    for f,h in prediction['source_sha256'].items():
        if hashlib.sha256(Path(f).read_bytes()).hexdigest()!=h:raise ValueError('Changed prediction source '+f)
    cfg=read(args.configuration);native=read(args.run/'native_command_inputs.json')
    c=PureControlReferenceClockV12(PositionPDInterface(**cfg['interface']),native['default_joint_position_rad'],cfg['model_reference_library'])
    names=read(args.run/'pure_control_ground_forces.json')['body_names']
    feet=[names.index(n) for n in c.model.feet]
    with np.load(args.run/'pure_control_ground_forces.npz') as z:
        impulse=z['interval_normal_impulse_ns'].copy()
        frames=z['interval_frames_over1n'].copy()
        peaks=z['interval_peak_normal_n'].copy()
    with np.load(args.run/'pure_control_commands.npz') as z:
        q,v=z['q'].copy(),z['v'].copy()
        index={(int(t),int(l)):i for i,(t,l) in enumerate(zip(z['tick'],z['lane']))}
    rows=[]
    for r in prediction['records']:
        if r['substeps']!=160 or not r['completed']:continue
        case,tick=r['case'],r['tick'];i=index[(tick+1,case)]
        gap=sphere_gaps(c.model,q[i],c.geometry.spheres)
        epsilon=1e-6
        rate=(sphere_gaps(c.model,pin.integrate(c.model.model,q[i],epsilon*v[i]),c.geometry.spheres)-gap)/epsilon
        for j in np.flatnonzero(np.asarray(r['predicted_observed_contacts'])!=np.asarray(r['next_contacts'])):
            foot=feet[j]
            rows.append(dict(case=case,tick=tick,foot=c.model.feet[j],
                predicted_endpoint_contact=r['predicted_observed_contacts'][j],native_endpoint_contact=r['next_contacts'][j],
                model_endpoint_normal_n=r['predicted_last_normal_force_n'][j],native_endpoint_normal_n=r['native_next_normal_force_n'][j],
                native_interval_mean_normal_n=float(impulse[tick,case,foot,2]/.01),
                native_interval_loaded_fraction=float(frames[tick,case,foot]/160),
                native_interval_peak_normal_n=float(peaks[tick,case,foot]),
                native_endpoint_foot_surface_gap_m=float(gap[j]),
                native_endpoint_foot_vertical_velocity_m_s=float(rate[j])))
    summary=dict(mismatching_foot_endpoints=len(rows),
        native_endpoint_over5n=sum(r['native_endpoint_normal_n']>5 for r in rows),
        previous_interval_mean_over5n=sum(r['native_interval_mean_normal_n']>5 for r in rows),
        previous_interval_majority_loaded=sum(r['native_interval_loaded_fraction']>.5 for r in rows),
        gap_quantiles_m=np.percentile([r['native_endpoint_foot_surface_gap_m'] for r in rows],[0,50,95,100]).tolist(),
        absolute_vertical_speed_quantiles_m_s=np.percentile([abs(r['native_endpoint_foot_vertical_velocity_m_s']) for r in rows],[0,50,95,100]).tolist(),
        native_contact_or_control_changed=False)
    files=[Path(__file__),args.run/'native_command_inputs.json',args.run/'pure_control_commands.npz',
        args.run/'pure_control_ground_forces.npz',args.run/'pure_control_ground_forces.json',args.configuration,args.prediction_audit]
    pins={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    pins.update(prediction['source_sha256'])
    output=dict(summary=summary,records=rows,source_sha256=pins,
        force_thresholds_are_diagnostic_not_new_control_gates=True,
        vertical_velocity_is_local_kinematic_derivative=True,
        past_impulse_not_proof_of_current_support=True,valid_expert_labels=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
