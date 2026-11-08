"""Localize impact versus subsequent drift using measured restart states.

These are offline counterfactual diagnostics. Restarts are never applied to
Isaac or used as a controller, successful rollout or expert demonstration.
"""
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
from scripts.compare_event_control_interval_v1 import cold_step
from scripts.audit_joint_sequence_proposal_v1 import plain


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['plan','run','audit','output']:
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Fresh output required')
    read=lambda p:json.loads(Path(p).read_text())
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    evidence=read(args.audit)
    if not evidence['passed'] or not evidence['passive_original_replay']:
        raise ValueError('Passive native evidence required')
    pins=dict(evidence['source_sha256'])
    for p,h in pins.items():
        if sha(p)!=h:raise ValueError('Changed evidence '+p)
    source=read(read(args.plan)['source'])
    _,_,_,_,context=source_context(source['configuration']['source']);initialize(*context)
    w=_WORKER;model=w['controller'].qp.model;interface=w['controller'].qp.interface
    meta=read(args.run/'foot_impact_observation.json')
    with np.load(args.run/'foot_impact_observation.npz',allow_pickle=False) as z:
        data={k:z[k].copy() for k in z.files}
    frames=data['physics_frame'];dt=meta['physics_dt_s']
    if not np.all(np.diff(frames)==1):raise ValueError('Continuous records required')
    rows=[]
    # Chosen from the already published control-interval and physical-step
    # comparisons, before these diagnostic predictions are executed.
    selected=[(85,298,47735),(87,303,48601),(87,305,48832)]
    for case,tick,impact in selected:
        lane=meta['lanes'].index(case);end=(tick+1)*160
        def actual(frame):
            i=int(np.flatnonzero(frames==frame)[0])
            sample={k:v[i,lane] for k,v in data.items() if k!='physics_frame'}
            return state_from_sample(model,sample,meta['origins_world'][lane])
        endpoint=actual(end)
        anchors=sorted(set([tick*160,impact,impact+1,impact+2,impact+10,impact+30]))
        for start in anchors:
            if start>=end:continue
            state=actual(start);trace=[]
            try:
                for frame in range(start,end):
                    observed=actual(frame+1)
                    state=cold_step(model,interface,state,observed.previous_command_target,w['spheres'],dt)
                    dv=state.v-observed.v
                    trace.append(dict(frame=frame+1,maximum_joint_velocity_error_rad_s=float(np.max(abs(dv[model.v_indices])))))
                dq=pin.difference(model.model,endpoint.q,state.q);dv=state.v-endpoint.v
                worst=int(np.argmax(abs(dv[model.v_indices])))
                rows.append(dict(case=case,control_tick=tick,impact_frame=impact,start_frame=start,end_frame=end,
                    completed=True,predicted_physics_steps=end-start,
                    root_position_error_m=float(np.linalg.norm(dq[:3])),
                    maximum_joint_position_error_rad=float(np.max(abs(dq[model.v_indices]))),
                    maximum_joint_velocity_error_rad_s=float(np.max(abs(dv[model.v_indices]))),
                    worst_joint=model.joint_order[worst],trace=trace))
            except (ValueError,np.linalg.LinAlgError) as error:
                rows.append(dict(case=case,control_tick=tick,start_frame=start,completed=False,error=repr(error)))
    for p in [args.plan,args.audit,Path(__file__).resolve()]:pins[str(p.resolve())]=sha(p)
    for name,module in list(sys.modules.items()):
        if name.startswith('scripts.') and getattr(module,'__file__',None):
            p=str(Path(module.__file__).resolve());digest=sha(p)
            if p in pins and pins[p]!=digest:raise ValueError('Pinned source changed '+p)
            pins[p]=digest
    for p,h in pins.items():
        if sha(p)!=h:raise ValueError('Source changed during diagnostic '+p)
    result=dict(scope=__doc__,rows=rows,source_sha256=pins,
        native_state_reset=False,new_native_execution=False,valid_expert_labels=False,
        limitations=['Measured restart removes earlier state error; it is not a realizable action.',
            'Later starts also shorten the prediction horizon, so comparisons are diagnostic only.',
            'No hidden friction memory or contact-solver state is supplied.'])
    tmp=args.output.with_suffix('.tmp');tmp.write_text(json.dumps(plain(result),indent=2,allow_nan=False)+'\n');tmp.replace(args.output)
    print(json.dumps([{k:v for k,v in r.items() if k!='trace'} for r in rows]))


if __name__=='__main__':main()
