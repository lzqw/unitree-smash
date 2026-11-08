"""Compare coupled model response to actual next native release-step state."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import pinocchio as pin
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.check_constrained_recovery_suffix_v1 import source_context,state_from_row
from scripts.search_landing_support_tail_v3 import _WORKER
from scripts.receding_full_command_search_v1 import initialize_court
from scripts.predict_position_limited_spheres_v1 import predict_position_limited_spheres
from scripts.audit_joint_sequence_proposal_v1 import plain


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['run','model-inputs','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--sample-event',choices=['release','first_ground'],default='release')
    a=p.parse_args();read=lambda p:json.loads(Path(p).read_text())
    if a.output.exists():raise ValueError('Fresh output required')
    x=read(a.model_inputs);s=read(read(x['configuration']['model_plan'])['source'])
    *_,context=source_context(s['configuration']['source']);initialize_court(*context)
    w=_WORKER;c=w['controller'];m=c.qp.model
    with np.load(a.run/'pure_control_commands.npz') as z:data={k:z[k].copy() for k in z.files}
    phase=[json.loads(line)['phase'] for line in (a.run/'pure_control_diagnostics.jsonl').read_text().splitlines()]
    records=[];start=time.monotonic()
    for lane in sorted(set(data['lane'].tolist())):
        ids=np.flatnonzero(data['lane']==lane)
        available=ids[:-1]
        if a.sample_event=='first_ground':
            available=np.asarray([i for i in available if any(phase[i]['ground_contacts'])],dtype=int)
        if not len(available):
            records.append(dict(case=lane,predicted=False,missing_event=True));continue
        i=int(available[0]);j=int(ids[np.flatnonzero(ids==i)[0]+1])
        if data['tick'][j]!=data['tick'][i]+1:raise ValueError('Actual consecutive states required')
        state=state_from_row(dict(q=data['q'][i],v=data['v'][i],contacts=phase[i]['ground_contacts'],target=data['previous'][i]))
        r=dict(case=lane,control_tick=int(data['tick'][i]),native_contacts=phase[i]['ground_contacts'],predicted=False)
        try:
            pred=predict_position_limited_spheres(m,c.qp.interface,state,data['final'][i],w['spheres'],geometry=w['geometry'],substeps=160)
            y=pred['state'];q=data['q'][j];v=data['v'][j]
            # Compare free-flyer velocities in world coordinates, not two
            # different body frames. Configuration error uses the manifold.
            rp=pin.Quaternion(y.q[3:7]).matrix();ra=pin.Quaternion(q[3:7]).matrix()
            difference=pin.difference(m.model,q,y.q)
            r.update(predicted=True,position_error_m=float(np.linalg.norm(y.q[:3]-q[:3])),
                attitude_error_rad=float(np.linalg.norm(difference[3:6])),
                maximum_joint_position_error_rad=float(np.max(abs(y.q[m.q_indices]-q[m.q_indices]))),
                linear_velocity_error_m_s=float(np.linalg.norm(rp@y.v[:3]-ra@v[:3])),
                angular_velocity_error_rad_s=float(np.linalg.norm(rp@y.v[3:6]-ra@v[3:6])),
                maximum_joint_velocity_error_rad_s=float(np.max(abs(y.v[m.v_indices]-v[m.v_indices]))),
                predicted_contacts=y.contacts,actual_next_contacts=phase[j]['ground_contacts'],
                model_diagnostics=pred['diagnostics'])
        except (ValueError,np.linalg.LinAlgError) as e:r['model_rejection']=repr(e)
        records.append(r)
    keys=['position_error_m','attitude_error_rad','maximum_joint_position_error_rad','linear_velocity_error_m_s',
        'angular_velocity_error_rad_s','maximum_joint_velocity_error_rad_s']
    accepted=[r for r in records if r['predicted']]
    summary={k:dict(zip(['median','p90','maximum'],np.quantile([r[k] for r in accepted],[.5,.9,1.]))) for k in keys} if accepted else {}
    out=dict(records=records,model_responses=len(accepted),cases=len(records),errors=summary,
        scope='Actual command and next native state; no native teleport, no success inference',sample_event=a.sample_event,
        wall_seconds=time.monotonic()-start,source_sha256={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in
            [a.run/'pure_control_commands.npz',a.run/'pure_control_diagnostics.jsonl',a.model_inputs,Path(__file__)]})
    tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(plain(out),indent=2,allow_nan=False)+'\n');tmp.replace(a.output)
    print(json.dumps(plain({k:v for k,v in out.items() if k not in ['records','source_sha256']})))


if __name__=='__main__':main()
