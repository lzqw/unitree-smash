"""Compare a finite-iteration impact hypothesis on preselected native events."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pinocchio as pin

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.check_constrained_recovery_suffix_v1 import source_context
from scripts.search_preserved_early_support_v1 import initialize
from scripts.search_landing_support_tail_v3 import _WORKER
from scripts.native_recovery_query_bank_v1 import state_from_sample
from scripts.temporal_impact_probe_v1 import temporal_impact_probe
from scripts.audit_joint_sequence_proposal_v1 import plain


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['plan','run','comparison','output']:
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Fresh output required')
    read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    evidence=read(args.comparison);pins=dict(evidence['source_sha256'])
    for p,h in pins.items():
        if sha(p)!=h:raise ValueError('Changed native evidence '+p)
    source=read(read(args.plan)['source']);_,_,_,_,context=source_context(source['configuration']['source']);initialize(*context)
    model=_WORKER['controller'].qp.model;interface=_WORKER['controller'].qp.interface
    meta=read(args.run/'foot_impact_observation.json')
    with np.load(args.run/'foot_impact_observation.npz',allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    selected=[r for r in evidence['rows'] if r['completed'] and not r['original_step_exact']]
    rows=[]
    for event in selected:
        case=event['case'];frame=event['physics_frame'];lane=meta['lanes'].index(case)
        i=int(np.flatnonzero(data['physics_frame']==frame)[0]);states=[]
        for j in [i,i+1]:
            sample={k:v[j,lane] for k,v in data.items() if k!='physics_frame'}
            states.append(state_from_sample(model,sample,meta['origins_world'][lane]))
        before,after=states
        try:
            result=temporal_impact_probe(model,interface,before,after.previous_command_target,
                _WORKER['spheres'],meta['physics_dt_s'])
            dq=pin.difference(model.model,after.q,result['state'].q);dv=result['state'].v-after.v
            rows.append(dict(case=case,physics_frame=frame,completed=True,
                original_joint_velocity_error_rad_s=event['baseline']['maximum_joint_velocity_error_rad_s'],
                event_resolved_joint_velocity_error_rad_s=event['metrics']['maximum_joint_velocity_error_rad_s'],
                temporal_joint_velocity_error_rad_s=float(np.max(abs(dv[model.v_indices]))),
                root_position_error_m=float(np.linalg.norm(dq[:3])),
                inferred_model_normal_force_n=result['impulse'][:,2]/meta['physics_dt_s'],
                observed_normal_force_n=np.asarray(event['baseline']['actual_final_foot_force_n'])[:,2],
                momentum_residual=result['momentum_residual'],joint_speed_excess=result.get('joint_speed_excess',0.),
                original_step_exact=result['original_step_exact']))
        except (ValueError,np.linalg.LinAlgError) as error:
            rows.append(dict(case=case,physics_frame=frame,completed=False,error=repr(error)))
    for p in [args.plan,args.comparison,Path(__file__).resolve(),ROOT/'scripts/temporal_impact_probe_v1.py']:
        pins[str(p.resolve())]=sha(p)
    ref=ROOT/'reports/physx_tgs_reference_20261108/version_5_4_0/manifest.json'
    upstream=read(ref);pins[str(ref.resolve())]=sha(ref)
    for entry in upstream['files']:pins[str(Path(entry['path']).resolve())]=entry['sha256']
    for p,h in pins.items():
        if sha(p)!=h:raise ValueError('Source changed during comparison '+p)
    result=dict(scope=__doc__,rows=rows,source_sha256=pins,upstream=upstream,
        model_promoted=False,new_native_execution=False,valid_expert_labels=False,
        limitations=['Fixed-inertia prototype, not a native PhysX/TGS implementation.',
            'No persistent friction anchors, exact articulation solver or contact patch cache.',
            'Native solver iteration configuration remains unchanged.',
            'Six already inspected diagnostic events are not independent validation.'])
    tmp=args.output.with_suffix('.tmp');tmp.write_text(json.dumps(plain(result),indent=2,allow_nan=False)+'\n');tmp.replace(args.output)
    print(json.dumps(plain(rows)))


if __name__=='__main__':main()
