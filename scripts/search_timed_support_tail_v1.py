"""Keep the improved timed landing prefix and optimize its sustained support.

Only the three full-body transition knots (starting at 0.35 s) change. Each
candidate is predicted from the original measured ground-executed query to
the original end time. No joint-stop impulse or relaxed constraint is added.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.check_constrained_recovery_suffix_v1 import source_context
from scripts.search_preserved_early_support_v1 import initialize
from scripts.search_recovery_timing_v1 import evaluate
from scripts.audit_joint_sequence_proposal_v1 import plain
from scripts.parallel_landing_solver_v1 import build_controller


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['model-plan','generation','bundle','output']:
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Fresh evidence required')
    read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    setup=read(a.model_plan);generation=read(a.generation);m=read(a.bundle/'manifest.json')
    eligible=[r for r in generation['fine'] if (r['result']['failure'] or {}).get('reason')!='declared_compute_budget']
    old=min(eligible,key=lambda r:r['result']['rank'])
    if (old['plan']['contract']!='timed_foot_space_recovery_v1'
            or old['result']['accepted'] or old['result']['completed_model_steps']<36):
        raise ValueError('A still-failing timed proposal with a full early prefix required')
    source=read(setup['source']);_,_,_,geometry,context=source_context(source['configuration']['source'])
    context=list(copy.deepcopy(context));query=m['query']
    if context[0]!=m['configuration']:raise ValueError('Physical interface changed')
    for key,field in [('initial_q','q'),('initial_v','v'),('initial_contacts','contacts'),('initial_previous_command','target')]:
        if not np.array_equal(old['result'][key],query[field]):raise ValueError('Initial query differs')
    context[1].update(initial_q=query['q'],initial_v=query['v'],initial_contacts=query['contacts'],
        actual_previous_command=query['target'],tick=280)
    with np.load(a.bundle/m['bundle_name'],allow_pickle=False) as z:
        context[4]=z['tracking_reference_q'].copy();context[5]=z['tracking_reference_v'].copy()
    context[6]=m['tracking_gains'];context=tuple(context)
    joint_order=build_controller(context[0]).qp.model.joint_order
    rear_calf=joint_order.index('RL_calf_joint')
    pins=dict(m['source_sha256'])
    for path in [a.model_plan,a.generation,a.bundle/'manifest.json',a.bundle/m['bundle_name'],Path(__file__).resolve()]:
        pins[str(path.resolve())]=sha(path)
    for name,module in list(sys.modules.items()):
        if name.startswith('scripts.') and getattr(module,'__file__',None):
            path=str(Path(module.__file__).resolve());digest=sha(path)
            if path in pins and pins[path]!=digest:raise ValueError('Implementation changed '+path)
            pins[path]=digest
    for path,digest in pins.items():
        if sha(path)!=digest:raise ValueError('Evidence changed '+path)
    a.output.mkdir(parents=True)
    def write(name,data):
        path=a.output/name;tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(plain(data),indent=2,allow_nan=False)+'\n');tmp.replace(path)
    write('owner.json',dict(pid=os.getpid()))
    write('inputs.json',dict(configuration={k:str(v) for k,v in vars(a).items()},source_sha256=pins,
        original_measured_state=True,physical_constraints_changed=False,max_seconds=1200.,workers=2))
    stem=copy.deepcopy(old['plan']);original=np.asarray(stem['transition_knots_rad'])
    if original.shape!=(3,18):raise ValueError('Three full-body support knots required')
    bound=np.r_[np.full(12,.3),np.full(6,.12)]
    mean=original.copy();std=np.tile(np.r_[np.full(12,.07),np.full(6,.025)],(3,1))
    floor=np.r_[np.full(12,.012),np.full(6,.006)];rng=np.random.default_rng(882160)
    began=time.monotonic();deadline=began+1200.;search_end=began+900.;ledger=[];verified=None
    def candidate(knots):
        plan=copy.deepcopy(stem);plan['transition_knots_rad']=np.clip(knots,original-bound,original+bound);return plan
    def prefix_same(row):
        rows=row['result']['trajectory'];reference=old['result']['trajectory']
        return len(rows)>=35 and all(np.array_equal(x[k],y[k]) for x,y in zip(rows[:35],reference[:35])
            for k in ['q','v','target','contacts'])
    try:
        with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=context) as pool:
            def batch(plans,resolution,limit):
                return list(pool.map(evaluate,[dict(plan=plan,query=query,geometry=geometry,substeps=resolution,deadline=limit) for plan in plans]))
            baseline=batch([stem],160,deadline)[0]
            for key in ['accepted','completed_model_steps','failure','rank','terminal']:
                if baseline['result'][key]!=old['result'][key]:raise ValueError('Baseline changed '+key)
            if not prefix_same(baseline):raise ValueError('Baseline prefix changed')
            write('baseline.json',baseline);best=baseline
            for generation_index in range(5):
                if time.monotonic()>=search_end:break
                population=[np.clip(mean+std*(.75*rng.normal(size=(1,18))+.5*rng.normal(size=(3,18))),original-bound,original+bound) for _ in range(24)]
                population[0]=original.copy();population[1]=mean.copy();population[2]=np.asarray(best['plan']['transition_knots_rad'])
                if generation_index==0:
                    # Seed modest left-rear knee flexion before the observed limit event.
                    for i,delta in enumerate([-.06,-.12,-.20],start=3):
                        population[i]=original.copy();population[i][:,rear_calf]+=delta
                coarse=batch([candidate(v) for v in population],32,search_end)
                ranked=sorted([r for r in coarse if (r['result']['failure'] or {}).get('reason')!='declared_compute_budget'],key=lambda r:r['result']['rank'])
                if not ranked:break
                elite=np.stack([r['plan']['transition_knots_rad'] for r in ranked[:6]])
                mean=.3*mean+.7*elite.mean(0);std=np.maximum(floor,.3*std+.7*elite.std(0))
                fine=batch([r['plan'] for r in ranked[:4]],160,deadline)
                for row in fine:
                    if (row['result']['failure'] or {}).get('reason')=='declared_compute_budget':continue
                    if not prefix_same(row):raise ValueError('Late search altered protected early model prefix')
                    if row['result']['rank']<best['result']['rank']:best=row
                write(f'generation_{generation_index:02d}.json',dict(coarse=coarse,fine=fine))
                ledger.append(dict(generation=generation_index,completed=best['result']['completed_model_steps'],
                    failure=best['result']['failure'],prefix_exact=True))
                write('status.json',dict(state='SEARCHING',**ledger[-1]));print(json.dumps(plain(ledger[-1])),flush=True)
                if best['result']['accepted']:
                    replay=batch([best['plan']],160,deadline)[0]
                    if replay['result']['accepted'] and len(replay['result']['trajectory'])==520 and all(
                            np.array_equal(x[k],y[k]) for x,y in zip(best['result']['trajectory'],replay['result']['trajectory'])
                            for k in ['q','v','target','contacts']):verified=replay
                    break
        for path,digest in pins.items():
            if sha(path)!=digest:raise ValueError('Source changed during search '+path)
        write('result.json',dict(scope=__doc__,case=m['case'],query=query,best=best,verified=verified,ledger=ledger,
            baseline_reproduced=True,protected_prefix_control_ticks=list(range(280,315)),source_sha256=pins,
            wall_seconds=time.monotonic()-began,native_execution=False,valid_expert_labels=False,
            physical_constraints_changed=False,physical_impossibility_claim=False))
        write('status.json',dict(state='COMPLETED',accepted=verified is not None,best_completed=best['result']['completed_model_steps']))
    except BaseException as error:
        write('status.json',dict(state='FAILED',error=repr(error)));raise


if __name__=='__main__':main()
