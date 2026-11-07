"""Full-episode trajectory search with the omitted hip/mount constraint added.

Every candidate uses original whole-body dynamics and 160 microsteps, with
microstep configuration rejection for the audited pair. This is not native
validation or a complete all-pairs self-collision certificate.
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


def job(label,knots,audit,deadline):
    w=_WORKER;ground=HipMountPathConstraintV1(w['geometry'],audit,margin_rad=.02)
    result=evaluate(w['controller'],w['initial'],knots,w['spheres'],ground,w['protocol'],
        substeps=160,deadline=deadline,save=True,policy_factory=w['factory'],control_tick=280)
    if ground.last_violation is not None:
        result['failure'].update(reason='authored_hip_mount_constraint',collision=ground.last_violation)
    return dict(label=label,knots_rad=knots,result=result,
                microstep_geometry_checks=ground.calls,physical_joint_limits_changed=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['search','collision-audit','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--max-seconds',type=float,default=1200.);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    if a.output.exists() or not 300<=a.max_seconds<=1800 or not 1<=a.workers<=4:
        raise ValueError('Fresh output and bounded full-resolution search required')
    read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    search=read(a.search);audit=read(a.collision_audit);source=read(search['configuration']['source'])
    paths=[Path(f) for f in source['source_sha256'] if Path(f).name=='refine_support_tail_fine160_v1.json']
    if len(paths)!=1:raise ValueError('Unique nominal source required')
    fine=read(paths[0]);tail=read(fine['configuration']['source']);parent=read(tail['configuration']['source_search'])
    cfg=parent['configuration'];query=read(cfg['query']);manifest=read(Path(cfg['bundle'])/'manifest.json')
    states=fine['independent_replay']['trajectory'];rq=np.asarray([query['initial_q']]+[r['q'] for r in states[:-1]])
    rv=np.asarray([query['initial_v']]+[r['v'] for r in states[:-1]])
    initargs=(manifest['configuration'],query,fine['fixed_early_knots_rad'],cfg['sphere_audit'],rq,rv,source['configuration'])
    hashes=dict(search['source_sha256']);hashes.update(audit['source_sha256'])
    for f in [a.search,a.collision_audit,Path(__file__),ROOT/'scripts/hip_mount_path_constraint_v1.py']:
        hashes[str(f.resolve())]=sha(f)
    for f,h in hashes.items():
        if sha(f)!=h:raise ValueError('Changed input '+f)
    original=np.asarray(search['best_knots_rad']);rng=np.random.default_rng(882119)
    began=time.monotonic();deadline=began+a.max_seconds;rows=[];best=None
    a.output.parent.mkdir(parents=True,exist_ok=True)
    def record(row):
        nonlocal best
        rows.append(row)
        if best is None or row['result']['rank']<best['result']['rank']:best=row
        print(json.dumps(plain(dict(label=row['label'],accepted=row['result']['accepted'],
            steps=row['result']['completed_model_steps'],failure=row['result']['failure'],best=best['label']))),flush=True)
        temp=a.output.with_suffix('.tmp');temp.write_text(json.dumps(plain(dict(completed=len(rows),best_label=best['label'],
            best_result={k:v for k,v in best['result'].items() if k!='trajectory'})),allow_nan=False)+'\n')
        temp.replace(a.output.with_suffix('.partial.json'))
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=initargs) as pool:
        candidates=[('old_reference_new_constraints',original.copy())]
        for amount in [.04,.08,.12,.18,.24]:
            for late in [False,True]:
                knots=original.copy();knots[:3,0]-=amount*np.array([.15,.5,1.])
                if late:knots[3:,0]-=amount*.5
                candidates.append((f'early_hip_{amount}_tail_{late}',knots))
        for generation in range(3):
            futures=[pool.submit(job,label,knots,audit,deadline) for label,knots in candidates]
            for future in as_completed(futures):record(future.result())
            if time.monotonic()>began+.8*a.max_seconds:break
            if generation==2:break
            ranked=sorted(rows,key=lambda r:r['result']['rank'])[:4]
            candidates=[]
            for i in range(12):
                anchor=np.asarray(ranked[i%len(ranked)]['knots_rad'])
                shared=rng.normal(0.,np.r_[np.full(12,.025),np.full(6,.008)],(2,1,18))
                noise=rng.normal(0.,np.r_[np.full(12,.008),np.full(6,.003)],(2,3,18))
                knots=anchor+(shared+noise).reshape(6,18)
                # Bounds limit this offline search; they do not change hardware.
                trust=np.r_[np.full(12,.35),np.full(6,.06)]
                knots=np.clip(knots,original-trust,original+trust)
                limits=np.asarray(manifest['configuration']['interface']['limits'])
                knots[:3]=np.clip(knots[:3],limits[:,0],limits[:,1])
                candidates.append((f'fullbody_generation_{generation+1}_{i}',knots))
        replay=None
        if best is not None and best['result']['accepted'] and time.monotonic()<deadline:
            replay=pool.submit(job,'independent_fine_replay',best['knots_rad'],audit,deadline).result()
    for f,h in hashes.items():
        if sha(f)!=h:raise ValueError('Input changed during search '+f)
    output=dict(scope=__doc__,source_sha256=hashes,configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        rows=rows,best=best,independent_replay=replay,wall_seconds=time.monotonic()-began,
        full_model_candidate_found=bool(replay and replay['result']['accepted']),
        margin_rad=.02,substeps=160,native_execution=False,valid_expert_labels=False,
        limitations=['Only audited hip/mount pairs added; remaining self-collision pairs still unmodeled.',
            'Finite search failure does not establish physical impossibility.',
            'Old 7/13 checks used a weaker constraint set and are not preservation targets under the added constraint.'])
    a.output.write_text(json.dumps(plain(output),indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
