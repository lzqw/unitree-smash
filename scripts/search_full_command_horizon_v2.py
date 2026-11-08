"""Experimental short-horizon recovery search over full absolute commands.

No old-policy trust region or terminal-standing hard gate. All actuator and
collision path checks remain. A feasible 600 ms prediction is not a completed
landing, full-episode certificate, native result, or usable expert label.
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
from scipy.interpolate import PchipInterpolator

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.check_constrained_recovery_suffix_v1 import source_context,state_from_row
from scripts.search_preserved_early_support_v1 import initialize
from scripts.search_landing_support_tail_v3 import _WORKER
from scripts.predict_position_limited_spheres_v1 import predict_position_limited_spheres
from scripts.hip_mount_path_constraint_v1 import HipMountPathConstraintV1
from scripts.foot_separation_path_constraint_v1 import FootSeparationPathConstraintV1
from scripts.event_recovery_manager_v1 import authored_clearance_state
from scripts.feedback_tail_rollout_v1 import terminal_measures
from scripts.audit_joint_sequence_proposal_v1 import plain


def rollout(nodes,query,audit,substeps,deadline):
    w=_WORKER;c=w['controller'];interface=c.qp.interface;state=state_from_row(query)
    nodes=np.asarray(nodes,float)
    if nodes.shape!=(5,18) or not np.isfinite(nodes).all():raise ValueError('Five full-command knots required')
    limits=interface.limits
    targets=limits[:,0]+.5*(np.clip(nodes,-1.,1.)+1.)*np.diff(limits).ravel()
    curve=PchipInterpolator([0.,.08,.18,.30,.45,.60],np.vstack([state.previous_command_target,targets]),axis=0)
    geometry=FootSeparationPathConstraintV1(HipMountPathConstraintV1(w['geometry'],audit,margin_rad=0.),w['spheres'],margin_m=0.)
    cost=0.;rows=[];failure=None
    for k in range(60):
        if time.monotonic()>=deadline:failure=dict(step=k,reason='compute_budget');break
        previous=state.previous_command_target.copy()
        target=np.clip(curve(k*.01),limits[:,0],limits[:,1])
        target[12:]=np.clip(target[12:],previous[12:]-.1,previous[12:]+.1)
        try:
            pred=predict_position_limited_spheres(c.qp.model,interface,state,target,w['spheres'],geometry=geometry,substeps=substeps)
        except (ValueError,np.linalg.LinAlgError) as e:
            failure=dict(step=k,reason='prediction_rejected',error=repr(e),collision=getattr(geometry,'last_violation',None));break
        bad={name:value for name,value in pred['diagnostics'].items() if
            (name=='nonfoot_ground_violation_m' and value>1e-6) or
            (name in ['joint_position_violation_rad','joint_velocity_violation_rad_s','motor_torque_violation_nm'] and value>3e-4)}
        if bad:failure=dict(step=k,reason='physical_path_violation',violations=bad,first_ground_violation=pred['first_ground_violation']);break
        state=pred['state'];state.previous_command_target=target.copy()
        measures=terminal_measures(c,state)
        clearances=authored_clearance_state(w['geometry'],state)
        # Transient tilt/velocity are costs, not final-standing constraints.
        supported=bool(state.contacts.any())
        cost+=.01*((8. if supported else 2.)*(measures['roll']**2+measures['pitch']**2)
            +2.*measures['omega']**2+2.*measures['horizontal']**2
            +.02*np.sum(((target-previous)/np.diff(limits).ravel())**2))
        for name,item in clearances.items():
            gap=item['gap_m'];rate=item['rate_m_s'];head=name in ['Head_lower','Head_upper']
            approaching=max(0.,-rate-max(0.,gap-.005)/.15)
            cost+=.01*(40. if head else 10.)*(np.exp(-max(gap,0.)/.02)+approaching**2)
        rows.append(dict(step=k,q=state.q.copy(),v=state.v.copy(),contacts=state.contacts.copy(),
            target=target.copy(),head_lower=clearances['Head_lower'],head_upper=clearances['Head_upper'],measures=measures))
    accepted=failure is None and len(rows)==60
    terminal=rows[-1]['measures'] if rows else None
    if terminal:cost+=5.*(terminal['roll']**2+terminal['pitch']**2)+terminal['omega']**2+terminal['horizontal']**2
    return dict(nodes=nodes,horizon_model_feasible=accepted,steps=len(rows),required_steps=60,
        rank=[0 if accepted else 1,60-len(rows),float(cost)],cost=float(cost),failure=failure,
        terminal=terminal,trajectory=rows if substeps==160 else None,substeps=substeps,
        full_episode_certified=False,native_execution=False,valid_expert_labels=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--warm-search',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--max-seconds',type=float,default=900.)
    a=p.parse_args()
    if a.output.exists() or not 120<=a.max_seconds<=900:raise ValueError('Fresh bounded horizon experiment required')
    read=lambda f:json.loads(Path(f).read_text());sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    data=read(a.inputs);cfg=data['configuration'];query=data['query']
    source=read(read(cfg['model_plan'])['source']);_,_,_,audit,context=source_context(source['configuration']['source'])
    initialize(*context);limits=_WORKER['controller'].qp.interface.limits
    old=read(a.warm_search)['best']['trajectory']
    if len(old)<30:raise ValueError('Warm commands must cover this horizon')
    warm=np.asarray([old[min(i,len(old)-1)]['target'] for i in [7,17,29,44,59]])
    mean=np.clip(2*(warm-limits[:,0])/np.diff(limits).ravel()-1,-1,1)
    previous=2*(np.asarray(query['target'])-limits[:,0])/np.diff(limits).ravel()-1
    pins=dict(data['source_sha256'])
    for f in [a.inputs,a.warm_search,Path(__file__)]:pins[str(f.resolve())]=sha(f)
    for name,module in list(sys.modules.items()):
        if name.startswith('scripts.') and getattr(module,'__file__',None):
            f=str(Path(module.__file__).resolve());h=sha(f)
            if f in pins and pins[f]!=h:raise ValueError('Changed dependency '+f)
            pins[f]=h
    for f,h in pins.items():
        if sha(f)!=h:raise ValueError('Changed input '+f)
    a.output.mkdir(parents=True)
    def write(name,value):
        path=a.output/name;tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(plain(value),indent=2,allow_nan=False)+'\n');tmp.replace(path)
    write('inputs.json',dict(source_sha256=pins,query=query,source_control_tick=int(cfg['tick']),
        horizon_seconds=.6,command_lower=limits[:,0],command_upper=limits[:,1],
        local_residual_bound=None,terminal_standing_is_hard_constraint=False,
        arm_command_rate_rad_s=10.,max_seconds=a.max_seconds,workers=2))
    start=time.monotonic();deadline=start+a.max_seconds;search_end=start+.72*a.max_seconds
    rng=np.random.default_rng(882204);std=np.full((5,18),.18);records=[];fine=[]
    write('status.json',dict(state='RUNNING_FULL_COMMAND_HORIZON'))
    try:
        with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=context) as pool:
            for generation in range(6):
                if time.monotonic()>=search_end:break
                candidates=np.clip(rng.normal(mean,std,(24,5,18)),-1,1)
                candidates[0]=mean;candidates[1]=np.tile(previous,(5,1))
                rows=[f.result() for f in [pool.submit(rollout,n,query,audit,32,search_end) for n in candidates]]
                rows.sort(key=lambda x:x['rank']);elite=np.asarray([x['nodes'] for x in rows[:5]])
                mean=elite.mean(axis=0);std=np.maximum(elite.std(axis=0),.04)
                verified=[f.result() for f in [pool.submit(rollout,r['nodes'],query,audit,160,deadline) for r in rows[:2]]]
                fine.extend(verified);records.append(dict(generation=generation,coarse=rows,fine=verified))
                best=min(fine,key=lambda x:x['rank'])
                write('partial.json',dict(records=records,best=best))
                print(json.dumps(dict(generation=generation,best_steps=best['steps'],horizon_feasible=best['horizon_model_feasible'],cost=best['cost'])),flush=True)
        for f,h in pins.items():
            if sha(f)!=h:raise ValueError('Changed source during run '+f)
        write('result.json',dict(scope=__doc__,source_sha256=pins,records=records,
            best=min(fine,key=lambda x:x['rank']) if fine else None,wall_seconds=time.monotonic()-start,
            native_execution=False,full_episode_certified=False,valid_expert_labels=False))
        write('status.json',dict(state='COMPLETED'))
    except BaseException as e:write('status.json',dict(state='FAILED',error=repr(e)));raise


if __name__=='__main__':main()
