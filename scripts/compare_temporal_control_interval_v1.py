"""Check whether the finite-iteration probe improves complete control intervals."""
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
    for key in ['plan','run','baseline','probe','output']:parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Fresh output required')
    read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    baseline=read(args.baseline);probe=read(args.probe);pins=dict(probe['source_sha256'])
    for p,h in baseline['source_sha256'].items():
        if p in pins and pins[p]!=h:raise ValueError('Inconsistent evidence '+p)
        pins[p]=h
    for p,h in pins.items():
        if sha(p)!=h:raise ValueError('Changed evidence '+p)
    source=read(read(args.plan)['source']);_,_,_,_,context=source_context(source['configuration']['source']);initialize(*context)
    model=_WORKER['controller'].qp.model;interface=_WORKER['controller'].qp.interface
    meta=read(args.run/'foot_impact_observation.json')
    with np.load(args.run/'foot_impact_observation.npz',allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    frame_lookup={int(f):i for i,f in enumerate(data['physics_frame'])};rows=[]
    for old in baseline['rows']:
        case=old['case'];tick=old['tick'];lane=meta['lanes'].index(case);first=tick*160
        def actual(frame):
            i=frame_lookup[frame];sample={k:v[i,lane] for k,v in data.items() if k!='physics_frame'}
            return state_from_sample(model,sample,meta['origins_world'][lane])
        state=actual(first);events=[];max_residual=0.;max_speed_excess=0.
        try:
            for frame in range(first,first+160):
                observed=actual(frame+1)
                result=temporal_impact_probe(model,interface,state,observed.previous_command_target,
                    _WORKER['spheres'],meta['physics_dt_s'])
                state=result['state'];max_residual=max(max_residual,result['momentum_residual'])
                max_speed_excess=max(max_speed_excess,result.get('joint_speed_excess',0.))
                if not result['original_step_exact']:events.append(frame)
            dq=pin.difference(model.model,observed.q,state.q);dv=state.v-observed.v
            metrics=dict(root_position_error_m=float(np.linalg.norm(dq[:3])),root_rotation_error_rad=float(np.linalg.norm(dq[3:6])),
                root_linear_velocity_error_m_s=float(np.linalg.norm(dv[:3])),root_angular_velocity_error_rad_s=float(np.linalg.norm(dv[3:6])),
                maximum_joint_position_error_rad=float(np.max(abs(dq[model.v_indices]))),
                maximum_joint_velocity_error_rad_s=float(np.max(abs(dv[model.v_indices]))))
            rows.append(dict(case=case,tick=tick,completed=True,metrics=metrics,
                baseline=old['modes'],changed_physics_frames=events,max_momentum_residual=max_residual,
                maximum_joint_speed_excess=max_speed_excess))
        except (ValueError,np.linalg.LinAlgError) as error:rows.append(dict(case=case,tick=tick,completed=False,error=repr(error)))
    for p in [args.plan,args.baseline,args.probe,Path(__file__).resolve()]:pins[str(p.resolve())]=sha(p)
    for p,h in pins.items():
        if sha(p)!=h:raise ValueError('Source changed during comparison '+p)
    output=dict(scope=__doc__,rows=rows,source_sha256=pins,model_promoted=False,new_native_execution=False,
        valid_expert_labels=False,limitations=probe['limitations'])
    tmp=args.output.with_suffix('.tmp');tmp.write_text(json.dumps(plain(output),indent=2,allow_nan=False)+'\n');tmp.replace(args.output)
    print(json.dumps(plain([{k:v for k,v in r.items() if k!='baseline'} for r in rows])))


if __name__=='__main__':main()
