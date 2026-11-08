"""Search absolute early whole-body references after local correction fails.

The first 3x18 reference knots are optimized directly; later support and
measured-state feedback remain in the full-episode rollout. This expands
internal candidate authority, never physical joint, torque or task limits.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.check_constrained_recovery_suffix_v1 import source_context,state_from_row
from scripts.recovery_family_queries_v1 import transfer_reference
from scripts.search_preserved_early_support_v1 import initialize
from scripts.search_landing_support_tail_v3 import _WORKER
from scripts.current_state_recovery_planner_v3 import build_feedback_plan,predict_plan
from scripts.audit_joint_sequence_proposal_v1 import plain


def evaluate(job):
    return predict_plan(job['plan'],job['query'],280,0,job['geometry'],job['substeps'],job['deadline'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['plan','search','failed-local','output']:
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Fresh output required')
    args.output.mkdir(parents=True)
    read=lambda p:json.loads(Path(p).read_text())
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def write(name,data):
        path=args.output/name;tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(plain(data),indent=2,allow_nan=False)+'\n');tmp.replace(path)
    failed=read(args.failed_local);setup=read(args.plan);source=read(setup['source']);search=read(args.search)
    if failed['accepted'] or not failed['query_is_native'] or failed['control_tick']!=280:
        raise ValueError('Require the actual failed local correction query')
    _,_,_,geometry,context=source_context(source['configuration']['source'])
    initialize(*context)
    context,_=transfer_reference(context,state_from_row(search['query']))
    context,_=transfer_reference(context,state_from_row(failed['query']))
    initialize(*context)
    warm=copy.deepcopy(failed['records'][0]['row']['plan'])
    if warm['corrections']:raise ValueError('Unmodified complete warm controller required')
    original=np.asarray(warm['original_knots_rad'],float)
    state=state_from_row(failed['query']);interface=_WORKER['controller'].qp.interface
    lower=np.empty((3,18));upper=lower.copy()
    lower[:,:12]=np.maximum(interface.limits[:12,0],original[:3,:12]-1.)
    upper[:,:12]=np.minimum(interface.limits[:12,1],original[:3,:12]+1.)
    lower[:,12:]=np.maximum(interface.limits[12:,0]-state.joint_position[12:],-.6)
    upper[:,12:]=np.minimum(interface.limits[12:,1]-state.joint_position[12:],.6)
    mean=np.clip(original[:3],lower,upper)
    std=np.tile(np.r_[np.full(12,.25),np.full(6,.10)],(3,1))
    floor=np.r_[np.full(12,.04),np.full(6,.02)]
    rng=np.random.default_rng(882143);pins=dict(failed['source_sha256'])
    for path in [args.plan,args.search,args.failed_local,Path(__file__).resolve()]:pins[str(path.resolve())]=sha(path)
    for path,digest in pins.items():
        if sha(path)!=digest:raise ValueError('Changed source '+path)
    began=time.monotonic();deadline=began+900.;search_end=began+720.
    best=None;verified=None;ledger=[]
    def candidate(early):
        plan=copy.deepcopy(warm);knots=original.copy();knots[:3]=early
        plan['original_knots_rad']=knots
        return plan
    try:
        with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn'),
                initializer=initialize,initargs=context) as pool:
            def batch(plans,substeps,limit):
                futures={pool.submit(evaluate,dict(plan=p,query=failed['query'],geometry=geometry,
                    substeps=substeps,deadline=limit)):i for i,p in enumerate(plans)}
                rows=[None]*len(plans)
                for future in as_completed(futures):rows[futures[future]]=future.result()
                return rows
            baseline=batch([warm],160,deadline)[0]
            old=failed['records'][0]['row']['result']
            for key in ['accepted','completed_model_steps','failure','rank','terminal']:
                if baseline['result'][key]!=old[key]:raise ValueError('Warm prediction changed: '+key)
            best=baseline
            for generation in range(6):
                if time.monotonic()>=search_end:break
                population=np.clip(mean+std*(.75*rng.normal(size=(48,1,18))+
                    .5*rng.normal(size=(48,3,18))),lower,upper)
                population[0]=mean;population[1]=original[:3]
                population[2]=np.clip(np.tile(np.r_[state.joint_position[:12],np.zeros(6)],(3,1)),lower,upper)
                population[3]=np.asarray(best['plan']['original_knots_rad'])[:3]
                coarse=batch([candidate(k) for k in population],32,search_end)
                ranked=[r for r in coarse if (r['result']['failure'] or {}).get('reason')!='declared_compute_budget']
                ranked.sort(key=lambda r:r['result']['rank'])
                if not ranked:break
                elite=np.stack([np.asarray(r['plan']['original_knots_rad'])[:3] for r in ranked[:8]])
                mean=.3*mean+.7*elite.mean(0);std=np.maximum(floor,.3*std+.7*elite.std(0))
                fine=batch([r['plan'] for r in ranked[:4]],160,deadline)
                for row in fine:
                    if (row['result']['failure'] or {}).get('reason')!='declared_compute_budget' and row['result']['rank']<best['result']['rank']:
                        best=row
                compact=lambda r:dict(plan=r['plan'],result={k:v for k,v in r['result'].items() if k!='trajectory'})
                write(f'generation_{generation:02d}.json',dict(coarse=coarse,fine=fine))
                ledger.append(dict(generation=generation,best=compact(best)))
                print(json.dumps(plain(dict(generation=generation,best_completed=best['result']['completed_model_steps'],
                    failure=best['result']['failure']))),flush=True)
                write('status.json',dict(state='SEARCHING',case=failed['case'],generation=generation,
                    best_completed=best['result']['completed_model_steps'],native_execution=False))
                if best['result']['accepted']:
                    replay=batch([best['plan']],160,deadline)[0]
                    a=best['result'];b=replay['result']
                    if b['accepted'] and len(a['trajectory'])==len(b['trajectory'])==520 and all(
                            np.array_equal(x[k],y[k]) for x,y in zip(a['trajectory'],b['trajectory']) for k in ['q','v','target','contacts']):
                        verified=replay;break
        for path,digest in pins.items():
            if sha(path)!=digest:raise ValueError('Source changed during search '+path)
        write('result.json',dict(scope=__doc__,case=failed['case'],query=failed['query'],baseline_reproduced=True,
            best=best,verified=verified,ledger=ledger,source_sha256=pins,wall_seconds=time.monotonic()-began,
            candidate_lower=lower,candidate_upper=upper,original_physical_constraints_changed=False,
            native_execution=False,valid_expert_labels=False,physical_impossibility_claim=False))
        write('status.json',dict(state='COMPLETED',case=failed['case'],verified=verified is not None,
            best_completed=best['result']['completed_model_steps'],native_execution=False))
    except BaseException as error:
        write('status.json',dict(state='FAILED',error=repr(error)));raise


if __name__=='__main__':main()
