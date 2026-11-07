"""Fine-model whole-body support search with paired initial-state constraints.

Preserve nominal feasibility first, then improve two previously failing
orientation probes. Every selection uses the full original episode and fine
dynamics. Recheck the selected candidate on all thirteen declared probes.
Neither finite sampling nor model acceptance certifies robust native recovery.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.evaluate_landing_reference_tracking_v1 import initialize
from scripts.search_landing_support_tail_v3 import _WORKER
from scripts.search_early_landing_trajectory_v1 import evaluate
from scripts.perturb_early_support_domain_v1 import perturb_state
from scripts.audit_joint_sequence_proposal_v1 import plain


def candidate_rank(results):
    if len(results)!=3:raise ValueError('Nominal and two declared probes required')
    if any((r.get('failure') or {}).get('reason')=='declared_compute_budget' for r in results):return None
    # A gain on failed probes cannot compensate for losing nominal feasibility.
    return [0 if results[0]['accepted'] else 1,sum(not r['accepted'] for r in results),
        max(r['rank'][1] for r in results),sum(r['rank'][1] for r in results),
        sum(r['cost']/max(1,r['completed_model_steps']) for r in results)]


def job(index,probe,knots,deadline,save):
    w=_WORKER;state=perturb_state(w['initial'],probe)
    r=evaluate(w['controller'],state,knots,w['spheres'],w['geometry'],w['protocol'],
        substeps=160,deadline=deadline,save=save,policy_factory=w['factory'],control_tick=280)
    return dict(index=index,probe=probe,result=r)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--population',type=int,default=6);p.add_argument('--generations',type=int,default=2)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--max-seconds',type=float,default=1200.)
    p.add_argument('--seed',type=int,default=882115);a=p.parse_args()
    if a.output.exists() or not 4<=a.population<=16 or not 1<=a.generations<=4 or not 1<=a.workers<=4 or not 900<=a.max_seconds<=1800:
        raise ValueError('Fresh output and bounded multistate search required')
    read=lambda f:json.loads(Path(f).read_text());sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    source=read(a.source)
    if not source['nominal_preserved'] or source['compute_expired_probes']:raise ValueError('Completed paired source required')
    fine_paths=[Path(f) for f in source['source_sha256'] if Path(f).name=='refine_support_tail_fine160_v1.json']
    if len(fine_paths)!=1:raise ValueError('Unique nominal reference source required')
    fine=read(fine_paths[0]);tail=read(fine['configuration']['source']);parent=read(tail['configuration']['source_search'])
    cfg=parent['configuration'];query=read(cfg['query']);manifest=read(Path(cfg['bundle'])/'manifest.json')
    states=fine['independent_replay']['trajectory'];reference_q=np.asarray([query['initial_q']]+[x['q'] for x in states[:-1]])
    reference_v=np.asarray([query['initial_v']]+[x['v'] for x in states[:-1]])
    initargs=(manifest['configuration'],query,fine['fixed_early_knots_rad'],cfg['sphere_audit'],reference_q,reference_v,source['configuration'])
    hashes=dict(source['source_sha256']);hashes[str(a.source.resolve())]=sha(a.source);hashes[str(Path(__file__).resolve())]=sha(__file__)
    for f,h in hashes.items():
        if sha(f)!=h:raise ValueError('Changed source '+f)
    names=['nominal','orientation_world_0_+1','orientation_world_1_+1']
    old={r['probe']['name']:r for r in source['rows']};probes=[old[n]['probe'] for n in names]
    nominal_knots=np.asarray(fine['best_knots_rad']);best_knots=nominal_knots.copy()
    best_rank=candidate_rank([old[n]['result'] for n in names]);initial_rank=best_rank.copy()
    bounds=np.asarray(tail['tail_bounds_rad']);mean=nominal_knots.copy()
    std=np.tile(np.r_[np.full(12,.02),np.full(6,.01)],(3,1));floor=std*.2
    rng=np.random.default_rng(a.seed);started=time.monotonic();deadline=started+a.max_seconds
    search_deadline=started+.65*a.max_seconds;ledger=[];baseline_reproduced=False
    a.output.parent.mkdir(parents=True,exist_ok=True)
    def checkpoint():
        tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(plain(dict(ledger=ledger,best_rank=best_rank,best_knots=best_knots)),allow_nan=False)+'\n');tmp.replace(a.output.with_suffix('.partial.json'))
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=initargs) as pool:
        for generation in range(a.generations):
            if time.monotonic()>=search_deadline:break
            population=np.clip(rng.normal(mean,std,(a.population,3,18)),-bounds,bounds);population[0]=best_knots
            if generation==0:population[0]=nominal_knots
            futures=[pool.submit(job,i,probe,k,search_deadline,False) for i,k in enumerate(population) for probe in probes]
            groups={i:{} for i in range(len(population))};ranked=[]
            for future in as_completed(futures):
                row=future.result();i=row['index'];groups[i][row['probe']['name']]=row['result']
                if len(groups[i])<3:continue
                results=[groups[i][n] for n in names];rank=candidate_rank(results)
                if generation==0 and i==0:
                    baseline_reproduced=all(all(results[j][key]==old[n]['result'][key] for key in
                        ['accepted','completed_model_steps','failure','terminal','rank']) for j,n in enumerate(names))
                    if not baseline_reproduced:raise ValueError('Original paired model behavior did not reproduce')
                ledger.append(dict(generation=generation,index=i,knots_rad=population[i],results=results,rank=rank))
                if rank is not None:
                    ranked.append((rank,population[i]))
                    if rank<best_rank:best_rank=rank;best_knots=population[i].copy()
                print(json.dumps(plain(dict(generation=generation,index=i,rank=rank,best_rank=best_rank,
                    accepted=[r['accepted'] for r in results],steps=[r['completed_model_steps'] for r in results]))),flush=True)
                checkpoint()
            # Keep the distribution centered on candidates that retain the
            # original nominal solution; timed-out computations are excluded.
            eligible=sorted((r for r in ranked if r[0][0]==0),key=lambda x:x[0])
            if not eligible:break
            elite=np.stack([r[1] for r in eligible[:max(2,len(eligible)//2)]])
            mean=elite.mean(0);std=np.maximum(elite.std(0),floor)
        checks=[]
        if time.monotonic()<deadline:
            futures=[pool.submit(job,i,r['probe'],best_knots,deadline,True) for i,r in enumerate(source['rows'])]
            for future in as_completed(futures):
                row=future.result();checks.append(row)
                print(json.dumps(plain(dict(final_probe=row['probe'],accepted=row['result']['accepted'],steps=row['result']['completed_model_steps']))),flush=True)
        checks.sort(key=lambda x:x['index'])
    for f,h in hashes.items():
        if sha(f)!=h:raise ValueError('Changed search source '+f)
    check_lookup={r['probe']['name']:r for r in checks}
    complete_checks=len(checks)==13 and not any((r['result'].get('failure') or {}).get('reason')=='declared_compute_budget' for r in checks)
    paired=[dict(name=r['probe']['name'],old_accepted=old[r['probe']['name']]['result']['accepted'],new_accepted=r['result']['accepted']) for r in checks]
    result=dict(scope=__doc__,configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        search_probe_names=names,source_sha256=hashes,baseline_reproduced=baseline_reproduced,initial_rank=initial_rank,
        ledger=ledger,best_rank=best_rank,best_knots_rad=best_knots,changed_reference=not np.array_equal(best_knots,nominal_knots),
        final_checks=checks,all_thirteen_checks_completed=complete_checks,paired=paired,
        original_successes_retained=complete_checks and all(x['new_accepted'] for x in paired if x['old_accepted']),
        accepted_model_probes=sum(x['new_accepted'] for x in paired),
        nominal_feasible=bool('nominal' in check_lookup and check_lookup['nominal']['result']['accepted']),
        wall_seconds=time.monotonic()-started,native_execution=False,valid_expert_labels=False,
        limitations=['Three fit scenarios and thirteen previously inspected developer probes are not held-out validation.',
            'Feedback gains and nominal root-state reference stay fixed; only 54 whole-body support target knots are optimized.',
            'Original early prefix, all path limits and the original episode horizon stay fixed. No native input is changed.',
            'Failure to find a robust candidate is not a hardware impossibility proof.'])
    a.output.write_text(json.dumps(plain(result),indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
