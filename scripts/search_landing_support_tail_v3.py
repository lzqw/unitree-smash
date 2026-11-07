"""Preserve a constrained landing prefix and optimize its sustained support tail.

This offline prediction search restarts every candidate at the verified actual
post-release observation. It never resets native physics to a model state.
Acceptance still requires the entire original episode remainder and trailing
stability. Model candidates are not native successes or expert labels.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
import numpy as np
import pinocchio as pin

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.parallel_landing_solver_v1 import build_controller
from scripts.loaded_pinocchio_model_v1 import PublicState
from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance
from scripts.search_early_landing_trajectory_v1 import evaluate
from scripts.search_whole_body_early_landing_v2 import WholeBodyEarlyReference
from scripts.audit_joint_sequence_proposal_v1 import plain


class SupportTailReference:
    def __init__(self,model,interface,initial,knots,*,fixed_knots):
        self.model=model;self.interface=interface;self.initial=initial
        self.base=WholeBodyEarlyReference(model,interface,initial,fixed_knots)
        self.knots=np.asarray(knots,float)
        if self.knots.shape!=(3,18) or not np.isfinite(self.knots).all():
            raise ValueError('Three finite full-body support corrections required')
        self.times=np.array([.8,1.3,2.2,3.5])
        self.values=np.vstack([np.zeros(18),self.knots])

    def correction(self,t):
        if t<=self.times[0]:return np.zeros(18)
        if t>=self.times[-1]:return self.values[-1].copy()
        index=int(np.searchsorted(self.times,t,side='right')-1)
        u=(t-self.times[index])/(self.times[index+1]-self.times[index])
        blend=u*u*u*(10.-15.*u+6.*u*u)
        return (1.-blend)*self.values[index]+blend*self.values[index+1]

    def action(self,offset,state,previous):
        target,mode=self.base.action(offset,state,previous)
        t=.01*(offset-80);delta=self.correction(t)
        # Preserve the old command path exactly when the correction vanishes.
        if not np.any(delta):return target,mode
        # Combine nominal and tail requests BEFORE the original actuator
        # clipping/rate map, so the arm slew limiter is applied only once.
        u=float(np.clip(t/.04,0.,1.));blend=u*u*(3.-2.*u)
        raw=self.initial.previous_command_target.copy()
        raw[:12]=self.base.legs(min(t,.50))
        raw[12:]=(1.-blend)*raw[12:]+blend*self.initial.joint_position[12:]+self.base.arms(min(t,.50))
        target=np.clip(raw+delta,self.interface.limits[:,0],self.interface.limits[:,1])
        target[12:]=np.clip(target[12:],previous[12:]-.01*self.interface.arm_rate,
                           previous[12:]+.01*self.interface.arm_rate)
        return target,'continuous_landing_with_optimized_support_tail'


_WORKER={}


def initialize_worker(configuration,query,fixed_knots,sphere_path):
    controller=build_controller(configuration);model=controller.qp.model
    q=np.asarray(query['initial_q']);v=np.asarray(query['initial_v'])
    state=PublicState(q,v,pin.Quaternion(q[3:7]).matrix(),q[:3].copy(),q[model.q_indices].copy(),
        v[model.v_indices].copy(),np.asarray(query['initial_contacts'],bool),0.)
    state.previous_command_target=np.asarray(query['actual_previous_command'])
    geometry=AuthoredGroundClearance(ROOT/'configs/go2_piper_authored_ground_geometry_v1.json',model,margin_m=0.)
    spheres=json.loads(Path(sphere_path).read_text())['spheres']
    protocol=json.loads((ROOT/'configs/evaluation_protocol_v2.json').read_text())['recovery']
    _WORKER.update(controller=controller,initial=state,geometry=geometry,spheres=spheres,protocol=protocol,
        factory=partial(SupportTailReference,fixed_knots=np.asarray(fixed_knots)),tick=query['tick'])


def evaluate_job(job):
    index,knots,substeps,deadline,save=job;w=_WORKER
    result=evaluate(w['controller'],w['initial'],knots,w['spheres'],w['geometry'],w['protocol'],
        substeps=substeps,deadline=deadline,save=save,policy_factory=w['factory'],control_tick=w['tick'])
    return index,knots,result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-search',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--population',type=int,default=32)
    parser.add_argument('--generations',type=int,default=6)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--max-seconds',type=float,default=900.)
    parser.add_argument('--seed',type=int,default=882112)
    a=parser.parse_args()
    if a.output.exists() or not 8<=a.population<=64 or not 1<=a.generations<=10 or not 1<=a.workers<=4 or not 120<=a.max_seconds<=1800:
        raise ValueError('Fresh output and finite worker/search budget required')
    read=lambda f:json.loads(Path(f).read_text());sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    source=read(a.source_search)
    for f,h in source['source_sha256'].items():
        if sha(f)!=h:raise ValueError('Changed source search provenance '+f)
    selected=min(source['fine_candidates'],key=lambda x:x['fine']['rank'])
    if source['case']!=3 or source['tick']!=280 or selected['fine']['completed_model_steps']<=80:
        raise ValueError('Recorded landing with a verified 0.8s model prefix required')
    cfg=source['configuration'];query=read(cfg['query']);manifest=read(Path(cfg['bundle'])/'manifest.json')
    initargs=(manifest['configuration'],query,selected['knots_rad'],cfg['sphere_audit'])
    initialize_worker(*initargs)
    hashes=dict(source['source_sha256']);hashes[str(a.source_search.resolve())]=sha(a.source_search)
    hashes[str(Path(__file__).resolve())]=sha(__file__)
    rng=np.random.default_rng(a.seed);mean=np.zeros((3,18))
    bound=np.tile(np.r_[np.full(12,.5),np.full(6,.25)],(3,1))
    std=np.tile(np.r_[np.full(12,.14),np.full(6,.06)],(3,1))
    floor=np.tile(np.r_[np.full(12,.025),np.full(6,.015)],(3,1))
    started=time.monotonic();search_deadline=started+.65*a.max_seconds;deadline=started+a.max_seconds
    ledger=[];candidates=[];best=None;best_knots=None
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn'),
                             initializer=initialize_worker,initargs=initargs) as pool:
        for generation in range(a.generations):
            if time.monotonic()>=search_deadline:break
            population=np.clip(rng.normal(mean,std,(a.population,3,18)),-bound,bound)
            population[0]=mean if best_knots is None else best_knots
            population[1]=np.zeros((3,18))
            jobs=[(i,k,16,search_deadline,False) for i,k in enumerate(population)]
            ranked=[]
            for index,knots,result in pool.map(evaluate_job,jobs,chunksize=1):
                ledger.append(dict(generation=generation,index=index,knots_rad=knots,result=result))
                ranked.append((result['rank'],knots));candidates.append((result['rank'],knots))
                if best is None or result['rank']<best['rank']:
                    best=result;best_knots=knots.copy()
                    print(json.dumps(plain(dict(generation=generation,index=index,best=best))),flush=True)
            a.output.with_suffix('.partial.json').write_text(json.dumps(plain(ledger),allow_nan=False)+'\n')
            ranked.sort(key=lambda x:x[0]);elite=np.stack([x[1] for x in ranked[:max(2,len(ranked)//4)]])
            mean=elite.mean(0);std=np.maximum(elite.std(0),floor)
        unique=[];seen=set()
        for rank,knots in sorted(candidates,key=lambda x:x[0]):
            if knots.tobytes() in seen:continue
            seen.add(knots.tobytes());unique.append(knots)
            if len(unique)==2:break
        fine=[]
        if time.monotonic()<deadline:
            jobs=[(i,k,160,deadline,True) for i,k in enumerate(unique)]
            for index,knots,result in pool.map(evaluate_job,jobs,chunksize=1):
                # Actual early-state trajectories must be identical, not just
                # the nominal reference at a hypothetical fixed observation.
                saved=selected['fine']['trajectory'][:80]
                observed=result['trajectory'][:80]
                if len(observed)!=80:raise ValueError('Preserved prefix did not replay')
                error=max(float(np.max(abs(np.asarray(x[k])-np.asarray(y[k]))))
                    for x,y in zip(observed,saved) for k in ['target','q','v'])
                if error>1e-12:raise ValueError('Support search altered preserved landing prefix')
                fine.append(dict(knots_rad=knots,prefix_maximum_difference=error,fine=result))
                print(json.dumps(plain(dict(fine={k:v for k,v in result.items() if k!='trajectory'},prefix_error=error))),flush=True)
    for f,h in hashes.items():
        if sha(f)!=h:raise ValueError('Changed source during support search '+f)
    output=dict(scope=__doc__,case=3,tick=280,configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        fixed_early_knots_rad=selected['knots_rad'],tail_times_s=[.8,1.3,2.2,3.5],tail_bounds_rad=bound,
        candidate_count=len(ledger),ledger=ledger,fine_candidates=fine,final_mean=mean,final_std=std,
        any_model_feasible=any(x['fine']['accepted'] for x in fine),wall_seconds=time.monotonic()-started,
        source_sha256=hashes,native_execution=False,valid_expert_labels=False,
        limitations=['All candidates predict the original episode; no native state teleport or successful teacher is claimed.',
            'Preserving a landing prefix does not guarantee recovery or ball-side legality in native physics.',
            'Finite search cannot prove infeasibility; self-collision and unmodeled contact remain separate native checks.'])
    a.output.write_text(json.dumps(plain(output),indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
