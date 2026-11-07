"""Optimize sustained support after preserving a collision-constrained prefix.

Coarse predictions propose changes; only full 160-substep model evaluations can
replace the fine incumbent. Every rollout starts at the original recorded
flight state and runs to the original episode boundary. Native state is never
teleported. No model result is promoted to an expert demonstration.
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
from scripts.search_preserved_early_support_v1 import initialize
from scripts.search_landing_support_tail_v3 import _WORKER
from scripts.search_early_landing_trajectory_v1 import evaluate
from scripts.hip_mount_path_constraint_v1 import HipMountPathConstraintV1
from scripts.audit_joint_sequence_proposal_v1 import plain


def job(label,knots,audit,substeps,deadline):
    w=_WORKER;ground=HipMountPathConstraintV1(w['geometry'],audit,margin_rad=.02)
    result=evaluate(w['controller'],w['initial'],knots,w['spheres'],ground,w['protocol'],
        substeps=substeps,deadline=deadline,save=substeps==160,policy_factory=w['factory'],control_tick=280)
    if ground.last_violation is not None:
        result['failure'].update(reason='authored_hip_mount_constraint',collision=ground.last_violation)
    return dict(label=label,knots_rad=knots,result=result,substeps=substeps,
        eligible_for_fine_incumbent=substeps==160,valid_expert_labels=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-seconds',type=float,default=1800.);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    if a.output.exists() or not 900<=a.max_seconds<=3600 or not 1<=a.workers<=4:
        raise ValueError('Fresh output and bounded computation required')
    read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    previous=read(a.source);search_path=Path(previous['configuration']['search']);search=read(search_path)
    audit=read(previous['configuration']['collision_audit']);source=read(search['configuration']['source'])
    paths=[Path(f) for f in source['source_sha256'] if Path(f).name=='refine_support_tail_fine160_v1.json']
    if len(paths)!=1:raise ValueError('Unique nominal source required')
    fine=read(paths[0]);tail=read(fine['configuration']['source']);parent=read(tail['configuration']['source_search'])
    cfg=parent['configuration'];query=read(cfg['query']);manifest=read(Path(cfg['bundle'])/'manifest.json')
    states=fine['independent_replay']['trajectory']
    rq=np.asarray([query['initial_q']]+[r['q'] for r in states[:-1]])
    rv=np.asarray([query['initial_v']]+[r['v'] for r in states[:-1]])
    initargs=(manifest['configuration'],query,fine['fixed_early_knots_rad'],cfg['sphere_audit'],rq,rv,source['configuration'])
    hashes=dict(previous['source_sha256'])
    for f in [a.source,Path(__file__)]:hashes[str(f.resolve())]=sha(f)
    for f,h in hashes.items():
        if sha(f)!=h:raise ValueError('Changed input '+f)
    original=np.asarray(previous['best']['knots_rad']);fixed=original[:3].copy()
    rng=np.random.default_rng(882120);mean=original[3:].copy()
    std=np.tile(np.r_[np.full(12,.07),np.full(6,.02)],(3,1))
    minimum=np.tile(np.r_[np.full(12,.012),np.full(6,.004)],(3,1))
    trust=np.r_[np.full(12,.4),np.full(6,.12)]
    lower,upper=mean-trust,mean+trust
    began=time.monotonic();deadline=began+a.max_seconds;search_end=began+.88*a.max_seconds
    ledger=[];fine_rows=[];generations=[];seen=set();best=None
    a.output.parent.mkdir(parents=True,exist_ok=True)
    def progress():
        tmp=a.output.with_suffix('.tmp')
        tmp.write_text(json.dumps(plain(dict(coarse_evaluations=len(ledger),fine_evaluations=len(fine_rows),
            best_label=best['label'],best_result={k:v for k,v in best['result'].items() if k!='trajectory'},
            completed_generations=len(generations))),allow_nan=False)+'\n')
        tmp.replace(a.output.with_suffix('.partial.json'))
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=initargs) as pool:
        best=pool.submit(job,'reproduced_constrained_prefix',original,audit,160,deadline).result()
        for k in ['accepted','completed_model_steps','terminal','terminal_violations','rank']:
            if best['result'][k]!=previous['best']['result'][k]:raise ValueError('Fine incumbent did not reproduce '+k)
        fine_rows.append(best);seen.add(original.tobytes());progress()
        for generation in range(6):
            if time.monotonic()>=search_end:break
            proposals=[]
            for i in range(40):
                center=mean if i<20 else np.asarray(best['knots_rad'])[3:]
                # Correlated temporal movement plus local knot freedom.
                shared=rng.normal(size=(1,18));local=rng.normal(size=(3,18))
                tail=np.clip(center+std*(.75*shared+.5*local),lower,upper)
                knots=np.vstack([fixed,tail]);proposals.append(knots)
            proposals[0]=np.asarray(best['knots_rad']).copy();proposals[1]=np.vstack([fixed,mean])
            rows=[]
            futures=[pool.submit(job,f'g{generation}_coarse_{i}',knots,audit,32,search_end) for i,knots in enumerate(proposals)]
            for future in as_completed(futures):rows.append(future.result())
            rows.sort(key=lambda r:r['result']['rank']);ledger.extend(rows)
            complete=[r for r in rows if (r['result']['failure'] or {}).get('reason')!='declared_compute_budget']
            if not complete:break
            elites=np.stack([r['knots_rad'][3:] for r in complete[:8]])
            mean=.3*mean+.7*elites.mean(0);std=np.maximum(minimum,.3*std+.7*elites.std(0))
            selected=[]
            for row in complete:
                key=np.asarray(row['knots_rad']).tobytes()
                if key in seen:continue
                seen.add(key);selected.append(row)
                if len(selected)==4:break
            futures=[pool.submit(job,r['label'].replace('coarse','fine'),r['knots_rad'],audit,160,search_end) for r in selected]
            for future in as_completed(futures):
                row=future.result();fine_rows.append(row)
                if row['result']['rank']<best['result']['rank']:best=row
                print(json.dumps(plain(dict(label=row['label'],steps=row['result']['completed_model_steps'],
                    accepted=row['result']['accepted'],failure=row['result']['failure'],best=best['label']))),flush=True)
            generations.append(dict(generation=generation,coarse_best_label=rows[0]['label'],fine_best_label=best['label'],
                std=std.copy(),mean=mean.copy()));progress()
        replay=None
        if best['result']['accepted'] and time.monotonic()<deadline:
            replay=pool.submit(job,'independent_fine_replay',best['knots_rad'],audit,160,deadline).result()
    for f,h in hashes.items():
        if sha(f)!=h:raise ValueError('Input changed during search '+f)
    result=dict(scope=__doc__,source_sha256=hashes,configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        baseline_reproduced=True,fixed_early_knots_rad=fixed,coarse_rows=ledger,fine_rows=fine_rows,generations=generations,
        best=best,independent_replay=replay,full_model_candidate_found=bool(replay and replay['result']['accepted']),
        wall_seconds=time.monotonic()-began,native_execution=False,valid_expert_labels=False,
        limitations=['Coarse feasibility is never used for acceptance or expert labels.',
            'Fine model checks include only audited self-collision pairs, not complete physical equivalence.',
            'All candidates share the selected prefix; failure does not rule out alternative earlier motions.'])
    a.output.write_text(json.dumps(plain(result),indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
