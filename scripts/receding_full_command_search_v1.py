"""Receding full-command shooting with coupled PD/contact/ground/net prediction.

This CPU experiment optimizes a complete future command sequence and applies
only its first 100 ms before solving again from the resulting state. The state
transition here is a model rollout, NOT native Isaac execution. No RL, fixed
TRAIN trigger, fake contact force, or full-episode certification is involved.
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
from scripts.check_constrained_recovery_suffix_v1 import source_context
from scripts.search_preserved_early_support_v1 import initialize
from scripts.search_landing_support_tail_v3 import _WORKER
from scripts.search_full_command_horizon_v2 import rollout as original_rollout
from scripts.authored_net_clearance_v1 import AuthoredNetClearanceV1,CourtCollisionPathConstraintV1
from scripts.audit_joint_sequence_proposal_v1 import plain
from quadsmash.physics.court_v4 import load_court


def initialize_court(*context):
    initialize(*context)
    ground=_WORKER['geometry']
    _WORKER['geometry']=CourtCollisionPathConstraintV1(ground,AuthoredNetClearanceV1(ground,load_court()))


def rollout(nodes,query,audit,substeps,deadline):
    result=original_rollout(nodes,query,audit,substeps,deadline)
    terminal=result['terminal']
    if terminal:
        # Rest/support is a cost, not an immediate standing hard constraint.
        margin=terminal['support_margin_m']
        extra=20.*(terminal['horizontal']**2+terminal['vertical_speed']**2)+10.*terminal['omega']**2
        extra+=10.*(terminal['roll']**2+terminal['pitch']**2)+.1*terminal['joint_speed']**2
        extra+=10.*max(0.,-(margin if margin is not None else -.2)/.1)**2
        result['original_cost']=result['cost'];result['cost']+=extra;result['rank'][-1]=result['cost']
    return result


def warm_knots(commands,previous,limits):
    commands=np.asarray(commands,float)
    if commands.ndim!=2 or commands.shape[1]!=18 or not len(commands):raise ValueError('Explicit command warm start required')
    curve=PchipInterpolator(np.arange(len(commands)+1)*.01,np.vstack([previous,commands]),axis=0)
    times=np.minimum(np.array([.08,.18,.30,.45,.60]),len(commands)*.01)
    targets=curve(times)
    return np.clip(2*(targets-limits[:,0])/np.diff(limits).ravel()-1,-1,1)


class FullCommandShootingMPCV1:
    def __init__(self,pool,audit,limits,seed=882211):
        self.pool=pool;self.audit=audit;self.limits=limits;self.rng=np.random.default_rng(seed)

    def solve(self,query,commands,deadline):
        started=time.monotonic();mean=warm_knots(commands,np.asarray(query['target']),self.limits)
        records=[];fine=[];std=np.full((5,18),.12);std[:,12:]=.06
        # Include the shifted previous candidate; never silently assume that
        # its old certificate remains valid at a new state or contact mode.
        first=self.pool.submit(rollout,mean,query,self.audit,160,min(deadline,started+90.)).result()
        fine.append(first)
        search_end=max(started,deadline-80.)
        for generation in range(3):
            if time.monotonic()>=search_end:break
            candidates=np.clip(self.rng.normal(mean,std,(12,5,18)),-1,1)
            candidates[0]=mean
            previous=2*(np.asarray(query['target'])-self.limits[:,0])/np.diff(self.limits).ravel()-1
            candidates[1]=np.tile(previous,(5,1))
            coarse=[f.result() for f in [self.pool.submit(rollout,n,query,self.audit,16,search_end) for n in candidates]]
            coarse.sort(key=lambda r:r['rank']);elite=np.asarray([r['nodes'] for r in coarse[:3]])
            mean=elite.mean(0);std=np.maximum(elite.std(0),.025)
            if time.monotonic()<deadline:
                verified=[f.result() for f in [self.pool.submit(rollout,r['nodes'],query,self.audit,160,deadline) for r in coarse[:2]]]
                fine.extend(verified)
            else:verified=[]
            records.append(dict(generation=generation,coarse=coarse,fine=verified))
        best=min(fine,key=lambda r:r['rank'])
        return dict(best=best,records=records,wall_seconds=time.monotonic()-started,
            accepted_full_horizon=best['horizon_model_feasible'],
            experimental_prefix_available=best['steps']>=10,
            native_execution=False,valid_expert_labels=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['inputs','warm-search','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--max-seconds',type=float,default=1200.)
    p.add_argument('--max-replans',type=int,default=5)
    a=p.parse_args();read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if a.output.exists() or not 120<=a.max_seconds<=1800 or not 1<=a.max_replans<=10:raise ValueError('Fresh bounded model experiment required')
    x=read(a.inputs);s=read(read(x['configuration']['model_plan'])['source']);*_,audit,context=source_context(s['configuration']['source'])
    initialize_court(*context);limits=_WORKER['controller'].qp.interface.limits
    query=dict(x['query']);warm=read(a.warm_search)['best']['trajectory']
    commands=np.asarray([r['target'] for r in warm]);pins=dict(x['source_sha256'])
    for path in [a.inputs,a.warm_search,Path(__file__),ROOT/'configs/court_net_v4.json']:pins[str(path.resolve())]=sha(path)
    for name,module in list(sys.modules.items()):
        if name.startswith(('scripts.','quadsmash.physics.court')) and getattr(module,'__file__',None):
            f=str(Path(module.__file__).resolve());h=sha(f)
            if f in pins and pins[f]!=h:raise ValueError('Changed dependency '+f)
            pins[f]=h
    for f,h in pins.items():
        if sha(f)!=h:raise ValueError('Changed input '+f)
    a.output.mkdir(parents=True)
    def write(name,value):
        path=a.output/name;tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(plain(value),indent=2,allow_nan=False)+'\n');tmp.replace(path)
    write('inputs.json',dict(source_sha256=pins,query=query,horizon_seconds=.6,replan_period_s=.1,
        actual_native_clock_used_only_for_diagnostic=x['configuration']['tick'],max_seconds=a.max_seconds,max_replans=a.max_replans,
        native_execution=False,valid_expert_labels=False))
    start=time.monotonic();deadline=start+a.max_seconds;history=[];executed=[];stopped=None
    write('status.json',dict(state='RUNNING_RECEDING_MODEL_MPC'))
    try:
        with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn'),initializer=initialize_court,initargs=context) as pool:
            mpc=FullCommandShootingMPCV1(pool,audit,limits)
            for iteration in range(a.max_replans):
                if deadline-time.monotonic()<90:stopped='total_compute_budget';break
                result=mpc.solve(query,commands,min(deadline,time.monotonic()+240.))
                history.append(result);best=result['best']
                if not result['experimental_prefix_available']:stopped='no_verified_100ms_prefix';break
                rows=best['trajectory'];executed.extend(rows[:10]);last=rows[9]
                query=dict(q=last['q'],v=last['v'],contacts=last['contacts'],target=last['target'])
                commands=np.asarray([r['target'] for r in rows[10:]])
                if not len(commands):commands=np.asarray([last['target']])
                write('partial.json',dict(history=history,executed=executed,next_query=query,stopped=stopped))
                print(json.dumps(dict(replan=iteration,executed_model_steps=len(executed),horizon_steps=best['steps'],
                    full_horizon=best['horizon_model_feasible'],terminal=best['terminal'],native_execution=False)),flush=True)
        for f,h in pins.items():
            if sha(f)!=h:raise ValueError('Changed source during run '+f)
        write('result.json',dict(history=history,executed=executed,next_query=query,stopped=stopped,
            wall_seconds=time.monotonic()-start,source_sha256=pins,native_execution=False,valid_expert_labels=False,full_episode_certified=False))
        write('status.json',dict(state='COMPLETED_MODEL_DIAGNOSTIC',executed_steps=len(executed),stopped=stopped))
    except BaseException as e:write('status.json',dict(state='FAILED',error=repr(e)));raise


if __name__=='__main__':main()
