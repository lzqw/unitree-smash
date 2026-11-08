"""Paired original-versus-event-resolved prediction on all recorded intervals."""
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
from scripts.event_resolved_foot_step_v1 import event_resolved_foot_step
from scripts.audit_joint_sequence_proposal_v1 import plain


def compare(job):
    w=_WORKER;model=w['controller'].qp.model;before=state_from_row(job['before']);actual=state_from_row(job['after'])
    old=job['baseline'];row=dict(case=old['case'],physics_frame=old['physics_frame'],baseline=old)
    try:
        result=event_resolved_foot_step(model,w['controller'].qp.interface,before,job['target'],w['spheres'],job['dt'])
        predicted=result['state'];delta=pin.difference(model.model,actual.q,predicted.q);dv=predicted.v-actual.v
        metrics=dict(root_position_error_m=float(np.linalg.norm(delta[:3])),
            root_rotation_error_rad=float(np.linalg.norm(delta[3:6])),
            root_linear_velocity_error_m_s=float(np.linalg.norm(dv[:3])),
            root_angular_velocity_error_rad_s=float(np.linalg.norm(dv[3:6])),
            maximum_joint_position_error_rad=float(np.max(abs(delta[model.v_indices]))),
            maximum_joint_velocity_error_rad_s=float(np.max(abs(dv[model.v_indices]))))
        if result['original_step_exact'] and any(abs(old[k]-v)>1e-12 for k,v in metrics.items()):
            raise ValueError('Inactive event path differs from the original prediction')
        row.update(completed=True,metrics=metrics,events=result['events'],
            original_step_exact=result['original_step_exact'],momentum_residual=result['momentum_residual'],
            normal_impulse_ns=result['impulse'][:,2])
    except (ValueError,np.linalg.LinAlgError) as error:row.update(completed=False,error=repr(error))
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['plan','run','baseline','output']:parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Fresh output required')
    read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    baseline=read(args.baseline);pins=dict(baseline['source_sha256'])
    for p,h in pins.items():
        if sha(p)!=h:raise ValueError('Changed native prediction evidence '+p)
    if any(not r['completed'] for r in baseline['rows']):raise ValueError('Complete original comparison required')
    setup=read(args.plan);source=read(setup['source']);_,_,_,_,context=source_context(source['configuration']['source']);initialize(*context)
    model=_WORKER['controller'].qp.model;meta=read(args.run/'foot_impact_observation.json')
    with np.load(args.run/'foot_impact_observation.npz',allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    original={(r['case'],r['physics_frame']):r for r in baseline['rows']};jobs=[]
    for li,case in enumerate(meta['lanes']):
        states=[]
        for i in range(len(data['physics_frame'])):
            sample={k:v[i,li] for k,v in data.items() if k!='physics_frame'}
            state=state_from_sample(model,sample,meta['origins_world'][li])
            states.append(dict(q=state.q,v=state.v,contacts=state.contacts,target=state.previous_command_target))
        for i,frame in enumerate(data['physics_frame'][:-1]):
            jobs.append(dict(before=states[i],after=states[i+1],target=states[i+1]['target'],
                dt=meta['physics_dt_s'],baseline=original[case,int(frame)]))
    for p in [args.plan,args.baseline,Path(__file__).resolve(),ROOT/'scripts/event_resolved_foot_step_v1.py']:
        pins[str(p.resolve())]=sha(p)
    began=time.monotonic()
    with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=context) as pool:
        rows=list(pool.map(compare,jobs,chunksize=16))
    summary={}
    for case in meta['lanes']:
        selected=[r for r in rows if r['case']==case];complete=[r for r in selected if r['completed']]
        changed=[r for r in complete if not r['original_step_exact']]
        summary[str(case)]=dict(intervals=len(selected),completed=len(complete),
            failed=[{k:v for k,v in r.items() if k!='baseline'} for r in selected if not r['completed']],
            original_path_exact_count=sum(r['original_step_exact'] for r in complete),
            changed_intervals=[dict(physics_frame=r['physics_frame'],
                old_joint_velocity_error=r['baseline']['maximum_joint_velocity_error_rad_s'],
                new_joint_velocity_error=r['metrics']['maximum_joint_velocity_error_rad_s'],
                old_normal_impulse_ns=np.asarray(r['baseline']['model_foot_impulse_ns'])[:,2],
                new_normal_impulse_ns=r['normal_impulse_ns'],
                measured_normal_force_n=np.asarray(r['baseline']['actual_final_foot_force_n'])[:,2],
                events=r['events']) for r in changed])
    for p,h in pins.items():
        if sha(p)!=h:raise ValueError('Source changed during comparison '+p)
    result=dict(scope=__doc__,summary=summary,rows=rows,source_sha256=pins,seconds=time.monotonic()-began,
        new_native_execution=False,model_promoted=False,valid_expert_labels=False,
        limitations=['Exploratory comparison on diagnostic states, not independent generalization.',
            'Native force is a sampled normal-force signal; friction impulse is not directly observed.',
            'No claim that event splitting reproduces the finite-iteration PhysX solver.'])
    tmp=args.output.with_suffix('.tmp');tmp.write_text(json.dumps(plain(result),indent=2,allow_nan=False)+'\n');tmp.replace(args.output)
    print(json.dumps(plain(summary)))


if __name__=='__main__':main()
