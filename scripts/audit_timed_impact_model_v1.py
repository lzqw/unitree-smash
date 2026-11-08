"""Compare coupled position-stop predictions with the timed recovery impact records.

Every prediction restarts from the actual measured state and the command
executed over that interval. No model trajectory replaces native states.
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
from scripts.predict_speed_limited_spheres_v1 import surface_jacobians
from scripts.audit_sphere_touchdown_predictions_v1 import sphere_gaps
from scripts.frictional_position_limited_step_v1 import position_limited_velocity_step
from scripts.audit_joint_sequence_proposal_v1 import plain


def _predict(job):
    w=_WORKER;model=w['controller'].qp.model;interface=w['controller'].qp.interface
    before=state_from_row(job['before']);after=state_from_row(job['after']);dt=job['dt']
    q,v=before.q,before.v;snap=model.snapshot(q,v)
    gap=sphere_gaps(model,q,w['spheres'])
    jac=surface_jacobians(model,q,w['spheres']).reshape(12,24)
    select=np.zeros((24,18));select[model.v_indices,np.arange(18)]=1.
    result=position_limited_velocity_step(snap['mass_matrix'],v,snap['nonlinear'],select,
        interface.kp,interface.kd,job['target']-q[model.q_indices],interface.effort,dt,
        jac,-gap/np.where(gap>0,dt,.01),interface.friction,interface.velocity,q[model.q_indices],interface.limits)
    next_v=result['velocity'];next_q=pin.integrate(model.model,q,dt*next_v)
    delta=pin.difference(model.model,after.q,next_q);dv=next_v-after.v
    impulse=result['impulse'].reshape(4,3)
    before_gap,_=w['geometry'].evaluate(q)
    after_gap,_=w['geometry'].evaluate(after.q)
    return dict(case=job['case'],physics_frame=job['frame'],dt=dt,
        actual_initial_nonfoot_gap_m=float(before_gap.min()),
        actual_final_nonfoot_gap_m=float(after_gap.min()),
        root_position_error_m=float(np.linalg.norm(delta[:3])),
        root_rotation_error_rad=float(np.linalg.norm(delta[3:6])),
        root_linear_velocity_error_m_s=float(np.linalg.norm(dv[:3])),
        root_angular_velocity_error_rad_s=float(np.linalg.norm(dv[3:6])),
        maximum_joint_position_error_rad=float(np.max(abs(delta[model.v_indices]))),
        maximum_joint_velocity_error_rad_s=float(np.max(abs(dv[model.v_indices]))),
        worst_joint=model.joint_order[int(np.argmax(abs(dv[model.v_indices])))],
        actual_initial_foot_force_n=job['before_force'],actual_final_foot_force_n=job['after_force'],
        model_foot_impulse_ns=impulse,model_equivalent_normal_force_n=impulse[:,2]/dt,
        actual_initial_foot_gap_m=gap,actual_final_foot_gap_m=sphere_gaps(model,after.q,w['spheres']),
        modeled_final_foot_gap_m=sphere_gaps(model,next_q,w['spheres']),
        motor_torque_nm=result['torque'],momentum_residual=result['momentum_residual'],
        position_stop_active=result['position_limit_active'],position_stop_impulse_nms=result['position_limit_impulse'],
        actual_joint_velocity=after.v[model.v_indices],model_joint_velocity=next_v[model.v_indices])


def predict(job):
    try:
        return dict(_predict(job),completed=True)
    except (ValueError,np.linalg.LinAlgError) as error:
        return dict(case=job['case'],physics_frame=job['frame'],completed=False,error=repr(error))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['plan','run','audit','output']:parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Fresh output required')
    read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    evidence=read(args.audit)
    if not evidence['passed'] or not evidence['passive_parent_replay'] or not evidence['controller_unchanged']:
        raise ValueError('Passive parent replay and aligned impact records required')
    pins=dict(evidence['source_sha256'])
    for p,h in pins.items():
        if sha(p)!=h:raise ValueError('Changed audited input '+p)
    setup=read(args.plan);source=read(setup['source'])
    _,_,_,_,context=source_context(source['configuration']['source']);initialize(*context)
    model=_WORKER['controller'].qp.model
    meta=read(args.run/'foot_impact_observation.json')
    if abs(meta['physics_dt_s']-.0000625)>1e-12:raise ValueError('Original physical step required')
    with np.load(args.run/'foot_impact_observation.npz',allow_pickle=False) as z:
        data={k:z[k].copy() for k in z.files}
    frames=data['physics_frame']
    if not np.all(np.diff(frames)==1):raise ValueError('No interpolation across missing physics frames')
    jobs=[]
    for index,case in enumerate(meta['lanes']):
        states=[]
        for i in range(len(frames)):
            sample={k:v[i,index] for k,v in data.items() if k!='physics_frame'}
            state=state_from_sample(model,sample,meta['origins_world'][index])
            states.append(dict(q=state.q,v=state.v,contacts=state.contacts,target=state.previous_command_target))
        for i in range(len(frames)-1):
            # At a control boundary the prior buffer belongs to the previous
            # action. The following physical sample contains the executed target.
            jobs.append(dict(case=case,frame=int(frames[i]),before=states[i],after=states[i+1],
                target=states[i+1]['target'],dt=meta['physics_dt_s'],
                before_force=data['foot_force_world'][i,index],after_force=data['foot_force_world'][i+1,index]))
    for p in [args.plan,args.audit,Path(__file__).resolve()]:pins[str(p.resolve())]=sha(p)
    for name,module in list(sys.modules.items()):
        if name.startswith('scripts.') and getattr(module,'__file__',None):
            p=Path(module.__file__).resolve();pins[str(p)]=sha(p)
    began=time.monotonic()
    with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=context) as pool:
        rows=list(pool.map(predict,jobs,chunksize=16))
    summaries={}
    for case in meta['lanes']:
        all_case=[r for r in rows if r['case']==case]
        subset=[r for r in all_case if r['completed']]
        fields=[k for k in subset[0] if '_error_' in k] if subset else []
        worst=sorted(subset,key=lambda r:r['maximum_joint_velocity_error_rad_s'],reverse=True)[:5]
        summaries[str(case)]=dict(intervals=len(all_case),completed=len(subset),
            failures=[r for r in all_case if not r['completed']],
            errors={k:dict(maximum=max(r[k] for r in subset),median=float(np.median([r[k] for r in subset]))) for k in fields},
            worst_frames=[r['physics_frame'] for r in worst])
        clear=[r for r in subset if min(r['actual_initial_nonfoot_gap_m'],r['actual_final_nonfoot_gap_m'])>.001]
        summaries[str(case)]['both_endpoints_clear_1mm']=dict(intervals=len(clear),errors={
            k:dict(maximum=max(r[k] for r in clear),median=float(np.median([r[k] for r in clear])))
            for k in fields} if clear else {})
    for p,h in pins.items():
        if sha(p)!=h:raise ValueError('Input changed during audit '+p)
    result=dict(scope=__doc__,summaries=summaries,rows=rows,source_sha256=pins,
        seconds=time.monotonic()-began,new_native_execution=False,valid_expert_labels=False,
        limitations=['Actual contact sensor net force is not asserted to include tangential impulse.',
            'Endpoint clearance does not exclude an unobserved contact within a physical interval.',
            'Nonfoot/self-contact impulses and finite PhysX iteration history are not modeled.',
            'Recorded float32 state precision sets a nonzero physical-step comparison floor.',
            'This is a model diagnostic, not new recovery or a fitted-model performance claim.'])
    tmp=args.output.with_suffix('.tmp');tmp.write_text(json.dumps(plain(result),indent=2,allow_nan=False)+'\n');tmp.replace(args.output)
    print(json.dumps(summaries))


if __name__=='__main__':main()
