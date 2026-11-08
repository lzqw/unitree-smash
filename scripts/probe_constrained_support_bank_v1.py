"""Compare feasible full-episode references from the same observed model state.

Selection uses predicted constraints and cost, not subsequent native outcomes.
This offline finite-candidate planning test is not online receding-horizon MPC,
a certified recovery region, or native teacher evidence.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.probe_constrained_support_domain_v1 import run_probe, initialize
from scripts.prepare_native_constrained_tracking_v1 import validated_candidate
from scripts.audit_joint_sequence_proposal_v1 import plain


def same_trajectory(first, second):
    if not first['accepted'] or not second['accepted']:
        return False
    a,b=first['trajectory'],second['trajectory']
    return len(a)==len(b)==520 and all(np.array_equal(x[k],y[k])
        for x,y in zip(a,b) for k in ['q','v','target','contacts'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['source','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--max-seconds',type=float,default=1800.)
    p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    if a.output.exists() or not 900<=a.max_seconds<=2400 or not 1<=a.workers<=4:
        raise ValueError('Fresh output and bounded prediction budget required')
    read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    domain=read(a.source)
    if not domain['nominal_exactly_reproduced'] or domain['probe_count']!=13 or domain['compute_expired_probes']:
        raise ValueError('Completed constrained initial-state probe set required')
    search_path=Path(domain['configuration']['source']);search=read(search_path);validated_candidate(search)
    previous=read(search['configuration']['source']);old=read(previous['configuration']['search'])
    source=read(old['configuration']['source']);audit=read(previous['configuration']['collision_audit'])
    paths=[Path(p) for p in source['source_sha256'] if Path(p).name=='refine_support_tail_fine160_v1.json']
    if len(paths)!=1:raise ValueError('Unique frozen reference required')
    fine=read(paths[0]);tail=read(fine['configuration']['source']);parent=read(tail['configuration']['source_search'])
    cfg=parent['configuration'];query=read(cfg['query']);manifest=read(Path(cfg['bundle'])/'manifest.json')
    states=fine['independent_replay']['trajectory']
    rq=np.asarray([query['initial_q']]+[x['q'] for x in states[:-1]])
    rv=np.asarray([query['initial_v']]+[x['v'] for x in states[:-1]])
    initargs=(manifest['configuration'],query,fine['fixed_early_knots_rad'],cfg['sphere_audit'],rq,rv,source['configuration'])
    hashes=dict(domain['source_sha256'])
    for path in [a.source.resolve(),Path(__file__).resolve()]:hashes[str(path)]=sha(path)
    for path,h in hashes.items():
        if sha(path)!=h:raise ValueError('Changed source '+path)
    candidates={};seen=set()
    for row in search['fine_rows']:
        if not row['result']['accepted']:continue
        knots=np.asarray(row['knots_rad']);key=knots.tobytes()
        if key in seen:continue
        seen.add(key);candidates[row['label']]=knots
    best=search['best']['label']
    if best not in candidates or len(candidates)<2:raise ValueError('Distinct complete references required')
    baseline={r['probe']['name']:r for r in domain['rows']}
    rows=[dict(candidate=best,source='completed_domain_evidence',**r) for r in domain['rows']]
    a.output.parent.mkdir(parents=True,exist_ok=True)
    began=time.monotonic();deadline=began+a.max_seconds;prediction_end=began+.72*a.max_seconds
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=initargs) as pool:
        jobs={pool.submit(run_probe,r['probe'],knots,audit,prediction_end):label
            for label,knots in candidates.items() if label!=best for r in domain['rows']}
        for future in as_completed(jobs):
            row=dict(candidate=jobs[future],source='new_fine_prediction',**future.result())
            expected=baseline[row['probe']['name']]
            if not np.array_equal(row['initial_q'],expected['initial_q']) or not np.array_equal(row['initial_v'],expected['initial_v']):
                raise ValueError('Candidates did not use the same starting state')
            rows.append(row)
            print(json.dumps(plain(dict(candidate=row['candidate'],probe=row['probe']['name'],
                accepted=row['result']['accepted'],steps=row['result']['completed_model_steps'],failure=row['result']['failure']))),flush=True)
            temp=a.output.with_suffix('.tmp');temp.write_text(json.dumps(plain(rows),allow_nan=False)+'\n');temp.replace(a.output.with_suffix('.partial.json'))
        choices=[];verification=[]
        for name,base in baseline.items():
            feasible=[r for r in rows if r['probe']['name']==name and r['result']['accepted']]
            if not feasible:
                choices.append(dict(probe=name,candidate=None,model_feasible=False,independently_reproduced=False));continue
            selected=min(feasible,key=lambda r:r['result']['cost']/r['result']['required_model_steps'])
            choices.append(dict(probe=name,candidate=selected['candidate'],model_feasible=True,
                independently_reproduced=selected['candidate']==best))
            if selected['candidate']!=best:
                verification.append((len(choices)-1,selected,pool.submit(run_probe,selected['probe'],
                    candidates[selected['candidate']],audit,deadline)))
        checks=[]
        for index,selected,future in verification:
            replay=future.result();same=same_trajectory(selected['result'],replay['result'])
            choices[index]['independently_reproduced']=same
            checks.append(dict(probe=selected['probe'],candidate=selected['candidate'],same_trajectory=same,replay=replay))
    for path,h in hashes.items():
        if sha(path)!=h:raise ValueError('Changed source during bank predictions '+path)
    output=dict(scope=__doc__,source_sha256=hashes,configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        candidates={k:v for k,v in candidates.items()},rows=rows,choices=choices,independent_replays=checks,
        baseline_accepted=domain['accepted_model_probes'],bank_verified_accepted=sum(c['model_feasible'] and c['independently_reproduced'] for c in choices),
        probe_count=13,reused_predictions=13,new_predictions=len(rows)-13,
        expired_new_predictions=sum((r['result']['failure'] or {}).get('reason')=='declared_compute_budget' for r in rows if r['source']=='new_fine_prediction'),
        wall_seconds=time.monotonic()-began,native_execution=False,valid_expert_labels=False,
        limitations=['Selection is based only on forward-model feasibility and cost at the supplied current state.',
            'All five references come from one nominal recovery family, not arbitrary recovery plans.',
            'This offline comparison does not demonstrate online replanning or native robustness.'])
    a.output.write_text(json.dumps(plain(output),indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
