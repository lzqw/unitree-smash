"""Compare refined prediction on complete recorded physical or control intervals.

Every candidate starts once from the same measured state and executes the same
held command. No measured future forces or intermediate state resets are used.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np
import pinocchio as pin

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.check_constrained_recovery_suffix_v1 import source_context,state_from_row
from scripts.search_preserved_early_support_v1 import initialize
from scripts.search_landing_support_tail_v3 import _WORKER
from scripts.native_recovery_query_bank_v1 import state_from_sample
from scripts.subcycled_foot_pair_step_v1 import subcycled_foot_pair_step
from scripts.audit_joint_sequence_proposal_v1 import plain


def evaluate(job):
    w=_WORKER;model=w['controller'].qp.model;interface=w['controller'].qp.interface
    state=state_from_row(job['before']);after=state_from_row(job['after']);began=time.monotonic()
    impulse=np.zeros((4,3));residual=0.;rows=dict(case=job['case'],physics_frame=job['frame'],
        variant=job['variant'],native_intervals=job['steps'],completed=False)
    try:
        for step in range(job['steps']):
            if time.monotonic()>job['deadline']:raise TimeoutError('Declared diagnostic deadline')
            r=subcycled_foot_pair_step(model,interface,state,job['target'],w['spheres'],job['dt'],
                foot_friction=job['friction'],subdivisions=8,endpoint_velocity=job['variant']=='refined_endpoint')
            state.q=r['q'];state.v=r['velocity'];impulse+=r['ground_impulse']
            residual=max(residual,r['momentum_residual'])
        dq=pin.difference(model.model,after.q,state.q);dv=state.v-after.v
        rows.update(completed=True,joint_velocity_error_rad_s=float(np.max(abs(dv[model.v_indices]))),
            joint_position_error_rad=float(np.max(abs(dq[model.v_indices]))),
            root_position_error_m=float(np.linalg.norm(dq[:3])),
            root_linear_velocity_error_m_s=float(np.linalg.norm(dv[:3])),
            root_angular_velocity_error_rad_s=float(np.linalg.norm(dv[3:6])),
            ground_impulse_ns=impulse,momentum_residual=residual)
    except (ValueError,np.linalg.LinAlgError,TimeoutError) as e:
        rows.update(error=repr(e),failed_native_interval=step)
    rows['seconds']=time.monotonic()-began
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['plan','run','audit','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--interval',choices=['physics','control'],required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Fresh output required')
    read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    audit=read(a.audit)
    if not audit['filtered_observation_passed'] or not audit['passive_parent_replay']:
        raise ValueError('Transparent native source required')
    pins=dict(audit['source_sha256'])
    for f,h in pins.items():
        if sha(f)!=h:raise ValueError('Changed measured source '+f)
    source=read(read(a.plan)['source']);*_,context=source_context(source['configuration']['source'])
    initialize(*context);model=_WORKER['controller'].qp.model
    meta=read(a.run/'foot_impact_observation.json')
    material=read(a.run/'native_admission.json')['material_admission']['feet'];mu=material[0]['material']['dynamic']
    if any(m['material']['dynamic']!=mu or m['material']['combine']!='average' for m in material):
        raise ValueError('Unexpected original friction coefficients')
    with np.load(a.run/'foot_impact_observation.npz',allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
    jobs=[];stride=1 if a.interval=='physics' else 160;deadline=time.monotonic()+600
    for lane,case in enumerate(meta['lanes']):
        states=[]
        for i in range(len(d['physics_frame'])):
            s=state_from_sample(model,{k:v[i,lane] for k,v in d.items() if k!='physics_frame'},meta['origins_world'][lane])
            states.append(dict(q=s.q,v=s.v,contacts=s.contacts,target=s.previous_command_target))
        for i in range(0,len(states)-stride,stride):
            target=states[i+1]['target']
            if any(not np.array_equal(states[j]['target'],target) for j in range(i+1,i+stride+1)):
                raise ValueError('Held native command changed inside interval')
            for variant in ['refined','refined_endpoint']:
                jobs.append(dict(case=case,frame=int(d['physics_frame'][i]),before=states[i],after=states[i+stride],
                    target=target,dt=meta['physics_dt_s'],friction=mu,steps=stride,variant=variant,deadline=deadline))
    for f in [a.plan,a.audit,Path(__file__),ROOT/'scripts/subcycled_foot_pair_step_v1.py',
            ROOT/'scripts/foot_pair_contact_step_v1.py',ROOT/'scripts/endpoint_contact_velocity_probe_v1.py']:
        pins[str(f.resolve())]=sha(f)
    began=time.monotonic()
    with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=context) as pool:
        rows=list(pool.map(evaluate,jobs,chunksize=8 if stride==1 else 1))
    summaries={}
    for case in meta['lanes']:
        for variant in ['refined','refined_endpoint']:
            selected=[r for r in rows if r['case']==case and r['variant']==variant];done=[r for r in selected if r['completed']]
            summaries[f'{case}_{variant}']=dict(intervals=len(selected),completed=len(done),
                maximum_error_rad_s=max([r['joint_velocity_error_rad_s'] for r in done],default=None),
                median_error_rad_s=float(np.median([r['joint_velocity_error_rad_s'] for r in done])) if done else None,
                failures=[r for r in selected if not r['completed']])
    for f,h in pins.items():
        if sha(f)!=h:raise ValueError('Pinned dependency changed during comparison '+f)
    result=dict(scope=__doc__,interval=a.interval,summaries=summaries,rows=rows,source_sha256=pins,
        seconds=time.monotonic()-began,maximum_seconds=600,workers=2,
        uses_measured_future_contact=False,native_physics_changed=False,model_promoted=False,valid_expert_labels=False)
    temp=a.output.with_suffix('.tmp');temp.write_text(json.dumps(plain(result),indent=2,allow_nan=False)+'\n');temp.replace(a.output)
    print(json.dumps(summaries))


if __name__=='__main__':main()
