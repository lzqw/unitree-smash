"""Separate actual ground/net collisions from clean model-response intervals.

This retrospective filtering is diagnostic only and never a control gate.
It retains all rejections and does not infer landing success from a solver.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['run','audit','retry','output']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();read=lambda path:json.loads(Path(path).read_text())
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if a.output.exists():raise ValueError('Fresh diagnostic output required')
    audit,retry=read(a.audit),read(a.retry)
    for data in [audit,retry]:
        for filename,digest in data['source_sha256'].items():
            if sha(filename)!=digest:raise ValueError('Changed audit input '+filename)
        for name in ['job.json','pure_control_commands.npz','pure_control_ground_forces.npz']:
            path=(a.run/name).resolve()
            if data['source_sha256'].get(str(path))!=sha(path):
                raise ValueError('Response audit used another native history')
    with np.load(a.run/'landing_contact_detail.npz') as z:net=z['net_peak'].copy()
    if not read(a.run/'landing_contact_detail.json')['complete']:
        raise ValueError('Completed native net telemetry required')
    rows=[]
    for row in audit['rows']:
        peak=float(net[row['tick'],row['case']].max())
        rows.append(dict(row,actual_net_peak_n=peak,
            actual_ground_and_net_clean=not row['actual_nonfoot_in_interval'] and peak==0.))
    keys=['position_error_m','attitude_error_rad','max_joint_position_error_rad',
        'linear_velocity_error_m_s','angular_velocity_error_rad_s','max_joint_velocity_error_rad_s']
    groups={}
    for label in sorted(audit['groups']):
        group=[r for r in rows if r['label']==label]
        clean=[r for r in group if r['actual_ground_and_net_clean']]
        returned=[r for r in clean if r['predicted']]
        groups[label]=dict(attempts=len(group),clean_intervals=len(clean),responses=len(returned),
            clean_rejections=[dict(case=r['case'],tick=r['tick'],reason=r['rejection']) for r in clean if not r['predicted']],
            errors={key:dict(zip(['median','p90','maximum'],np.quantile([r[key] for r in returned],[.5,.9,1.]).tolist())) for key in keys} if returned else {},
            mismatching_feet=sum(r['mismatching_feet'] for r in returned),
            contact_comparisons=4*len(returned))
    rejected_net=[r for r in rows if not r['actual_nonfoot_in_interval'] and not r['predicted'] and 'authored_net_collision:' in r['rejection']]
    numerical=[]
    for row in retry['rows']:
        if not row['contact_solver_retries']:continue
        if not row['predicted']:
            numerical.append(dict(case=row['case'],tick=row['tick'],predicted=False,rejection=row['rejection']));continue
        numerical.append(dict(case=row['case'],tick=row['tick'],predicted=True,
            retried_microsteps=len(row['contact_solver_retries']),
            maximum_contact_iterations=max(x['contact_iterations'] for x in row['contact_solver_retries']),
            model_nonfoot_violation_m=row['model_nonfoot_violation_m'],
            linear_velocity_error_m_s=row['linear_velocity_error_m_s'],angular_velocity_error_rad_s=row['angular_velocity_error_rad_s'],
            max_joint_velocity_error_rad_s=row['max_joint_velocity_error_rad_s'],
            joint_stop_active_microsteps=row['diagnostics']['position_limit_active_substeps'],
            maximum_joint_stop_impulse_nms=row['diagnostics']['maximum_joint_stop_impulse_nms']))
    files=[Path(__file__),a.audit,a.retry,a.run/'landing_contact_detail.npz',a.run/'landing_contact_detail.json']
    result=dict(groups=groups,numerical_retries=numerical,
        net_rejections_with_no_nonfoot_ground=[dict(case=r['case'],tick=r['tick'],actual_net_peak_n=r['actual_net_peak_n']) for r in rejected_net],
        ordinary_retry_controls=retry['ordinary_controls'],
        source_sha256={str(path.resolve()):sha(path) for path in files},
        native_rerun=False,acceptance_changed=False,solver_retry_deployed=False,
        clean_filter_is_retrospective=True,long_horizon_accuracy_not_inferred=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(clean_intervals=sum(g['clean_intervals'] for g in groups.values()),
        numerical_retry_cases=len(numerical),net_rejections=len(rejected_net),
        all_rejected_net_intervals_have_actual_net_force=all(r['actual_net_peak_n']>0 for r in rejected_net))))


if __name__=='__main__':main()
