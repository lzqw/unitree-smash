"""Full-remainder search of reference timing after a native-confirmed head impact.

Measured initial state, model, action limits and acceptance stay fixed. The
search varies phase timing, not contact thresholds or the task's time limit.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.check_constrained_recovery_suffix_v1 import source_context, state_from_row
from scripts.search_preserved_early_support_v1 import initialize
from scripts.search_landing_support_tail_v3 import _WORKER
from scripts.hip_mount_path_constraint_v1 import HipMountPathConstraintV1
from scripts.speed_limited_recovery_suffix_v1 import evaluate_suffix
from scripts.timed_whole_body_recovery_v1 import build_timed_plan
from scripts.audit_joint_sequence_proposal_v1 import plain


def evaluate(job):
    w = _WORKER
    policy = build_timed_plan(w, job['plan'])
    result = evaluate_suffix(w['controller'], state_from_row(job['query']), policy,
        w['spheres'], HipMountPathConstraintV1(w['geometry'], job['geometry'], margin_rad=.02),
        w['protocol'], control_tick=280, substeps=job['substeps'], deadline=job['deadline'],
        save=job['substeps']==160)
    return dict(plan=job['plan'], result=result)


def bounded(x):
    y = np.asarray(x, float).copy()
    y[0] = np.clip(y[0], .02, .12)
    y[1] = np.clip(y[1], y[0]+.02, .30)
    y[2] = np.clip(y[2], y[1]+.03, .75)
    y[3] = np.clip(y[3], .02, .15)
    return y


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['model-plan','search','bundle','native-audit','output']:
        p.add_argument('--'+key,type=Path,required=True)
    a = p.parse_args()
    if a.output.exists():raise ValueError('Fresh search evidence required')
    read=lambda p:json.loads(Path(p).read_text())
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    setup=read(a.model_plan);search=read(a.search);m=read(a.bundle/'manifest.json');audit=read(a.native_audit)
    if (search['case']!=m['case'] or m['case']!=audit['case']
            or audit['selected_case_outcome']['recovery'] or not audit['passed_dispatch_integrity']
            or search['best']['plan']!=m['proposal']):
        raise ValueError('Same actual failed recovery proposal required')
    source=read(setup['source']);_,_,_,geometry,context=source_context(source['configuration']['source'])
    context=list(copy.deepcopy(context))
    if context[0]!=m['configuration']:raise ValueError('Physical/controller configuration differs')
    query=m['query']
    context[1].update(initial_q=query['q'],initial_v=query['v'],initial_contacts=query['contacts'],
        actual_previous_command=query['target'],tick=280)
    with np.load(a.bundle/m['bundle_name'],allow_pickle=False) as z:
        context[4]=z['tracking_reference_q'].copy();context[5]=z['tracking_reference_v'].copy()
    context[6]=m['tracking_gains'];context=tuple(context)
    pins=dict(m['source_sha256'])
    for path in [a.model_plan,a.search,a.bundle/'manifest.json',a.bundle/m['bundle_name'],a.native_audit,Path(__file__).resolve()]:
        pins[str(path.resolve())]=sha(path)
    for name,module in list(sys.modules.items()):
        if name.startswith('scripts.') and getattr(module,'__file__',None):
            path=str(Path(module.__file__).resolve());digest=sha(path)
            if path in pins and pins[path]!=digest:raise ValueError('Changed implementation '+path)
            pins[path]=digest
    for path,digest in pins.items():
        if sha(path)!=digest:raise ValueError('Changed input '+path)
    a.output.mkdir(parents=True)
    def write(name,value):
        path=a.output/name;tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(plain(value),indent=2,allow_nan=False)+'\n');tmp.replace(path)
    write('owner.json',dict(pid=os.getpid()))
    write('inputs.json',dict(source_sha256=pins,case=m['case'],configuration={k:str(v) for k,v in vars(a).items()},
        native_execution=False,valid_expert_labels=False,physical_constraints_changed=False))
    began=time.monotonic();deadline=began+900.;search_end=began+720.
    default=np.array([.08,.20,.50,.04]);mean=default.copy();std=np.array([.035,.055,.13,.035])
    floor=np.array([.008,.012,.025,.008]);rng=np.random.default_rng(882159)
    stem=copy.deepcopy(search['best']['plan']);stem['contract']='timed_foot_space_recovery_v1'
    ledger=[];verified=None
    def proposal(timing):
        plan=copy.deepcopy(stem);plan['early_timing_s']=bounded(timing);return plan
    try:
        with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=context) as pool:
            def batch(plans,resolution,limit):
                jobs=[dict(plan=plan,query=query,geometry=geometry,substeps=resolution,deadline=limit) for plan in plans]
                return list(pool.map(evaluate,jobs))
            baseline=batch([proposal(default)],160,deadline)[0];old=search['best']['result']
            for key in ['accepted','completed_model_steps','failure','rank','terminal']:
                if baseline['result'][key]!=old[key]:raise ValueError('Default timing changed baseline '+key)
            if len(baseline['result']['trajectory'])!=len(old['trajectory']) or not all(
                    np.array_equal(x[k],y[k]) for x,y in zip(baseline['result']['trajectory'],old['trajectory'])
                    for k in ['q','v','target','contacts']):
                raise ValueError('Default timing changed trajectory')
            best=baseline;write('baseline.json',baseline)
            for generation in range(6):
                if time.monotonic()>=search_end:break
                population=[bounded(mean+std*rng.normal(size=4)) for _ in range(32)]
                population[0]=default;population[1]=mean;population[2]=best['plan']['early_timing_s']
                if generation==0:
                    population[3:7]=[bounded(t) for t in [[.04,.10,.25,.04],[.04,.14,.35,.08],
                        [.10,.28,.65,.04],[.06,.18,.40,.12]]]
                coarse=batch([proposal(t) for t in population],32,search_end)
                ranked=sorted([r for r in coarse if (r['result']['failure'] or {}).get('reason')!='declared_compute_budget'],
                    key=lambda r:r['result']['rank'])
                if not ranked:break
                elite=np.stack([r['plan']['early_timing_s'] for r in ranked[:6]])
                mean=.3*mean+.7*elite.mean(0);std=np.maximum(floor,.3*std+.7*elite.std(0))
                fine=batch([r['plan'] for r in ranked[:4]],160,deadline)
                for row in fine:
                    if ((row['result']['failure'] or {}).get('reason')!='declared_compute_budget'
                            and row['result']['rank']<best['result']['rank']):best=row
                write(f'generation_{generation:02d}.json',dict(coarse=coarse,fine=fine))
                ledger.append(dict(generation=generation,timing=best['plan']['early_timing_s'],
                    completed=best['result']['completed_model_steps'],failure=best['result']['failure']))
                write('status.json',dict(state='SEARCHING',**ledger[-1]))
                print(json.dumps(plain(ledger[-1])),flush=True)
                if best['result']['accepted']:
                    replay=batch([best['plan']],160,deadline)[0]
                    if (replay['result']['accepted'] and len(replay['result']['trajectory'])==520 and all(
                            np.array_equal(x[k],y[k]) for x,y in zip(best['result']['trajectory'],replay['result']['trajectory'])
                            for k in ['q','v','target','contacts'])):verified=replay
                    break
        for path,digest in pins.items():
            if sha(path)!=digest:raise ValueError('Source changed during search '+path)
        write('result.json',dict(scope=__doc__,case=m['case'],query=query,best=best,verified=verified,
            ledger=ledger,baseline_reproduced=True,source_sha256=pins,wall_seconds=time.monotonic()-began,
            native_execution=False,valid_expert_labels=False,physical_constraints_changed=False,
            physical_impossibility_claim=False))
        write('status.json',dict(state='COMPLETED',verified=verified is not None,
            best_completed=best['result']['completed_model_steps']))
    except BaseException as error:
        write('status.json',dict(state='FAILED',error=repr(error)));raise


if __name__=='__main__':main()
