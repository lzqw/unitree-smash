"""Locate prediction errors in an already audited physical-step record.

Force-norm and error thresholds below are retrospective reporting bins only;
they do not change controller contacts, task outcomes, or acceptance limits.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Fresh evidence output required')
    source=json.loads(a.source.read_text());rows=source['rows']
    summaries={}
    for case in sorted({r['case'] for r in rows}):
        subset=[r for r in rows if r['case']==case and r['completed']]
        clear=[r for r in subset if min(r['actual_initial_nonfoot_gap_m'],r['actual_final_nonfoot_gap_m'])>.001]
        clear_frames={r['physics_frame'] for r in clear}
        groups={}
        def new_load(row):
            before=np.linalg.norm(row['actual_initial_foot_force_n'],axis=1)>1.
            after=np.linalg.norm(row['actual_final_foot_force_n'],axis=1)>1.
            return bool(np.any(after&~before))
        def geometry_cross(row):
            return bool(np.any((np.asarray(row['actual_initial_foot_gap_m'])>0)
                               &(np.asarray(row['actual_final_foot_gap_m'])<=0)))
        selectors={'all_completed':lambda r:True,
                   'endpoint_clear_1mm':lambda r:r['physics_frame'] in clear_frames,
                   'new_normal_load_1n':new_load,
                   'sphere_plane_crossing':geometry_cross}
        for name,select in selectors.items():
            chosen=[r for r in subset if select(r)]
            error=[r['maximum_joint_velocity_error_rad_s'] for r in chosen]
            groups[name]=dict(intervals=len(chosen),maximum_joint_velocity_error_rad_s=max(error,default=None),
                median_joint_velocity_error_rad_s=float(np.median(error)) if error else None)
        worst=sorted(clear,key=lambda r:r['maximum_joint_velocity_error_rad_s'],reverse=True)[:8]
        summaries[str(case)]=dict(groups=groups,
            all_intervals=len([r for r in rows if r['case']==case]),
            failed_intervals=len([r for r in rows if r['case']==case and not r['completed']]),
            worst_endpoint_clear=[{k:r[k] for k in ['physics_frame','worst_joint',
                'maximum_joint_velocity_error_rad_s','actual_initial_foot_gap_m',
                'actual_final_foot_gap_m','actual_initial_foot_force_n','actual_final_foot_force_n',
                'model_foot_impulse_ns','actual_joint_velocity','model_joint_velocity']} for r in worst],
            above_1_rad_s_frames=[r['physics_frame'] for r in clear if r['maximum_joint_velocity_error_rad_s']>1.])
    result=dict(scope=__doc__,source=str(a.source.resolve()),summaries=summaries,
        source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),
        program_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        causal_root_cause_proven=False,controller_changed=False,valid_expert_labels=False)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({case:{k:v for k,v in value.items() if k not in ['worst_endpoint_clear','above_1_rad_s_frames']}
                      for case,value in summaries.items()}))


if __name__=='__main__':main()
