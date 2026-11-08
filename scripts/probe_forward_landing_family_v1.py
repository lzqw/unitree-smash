"""Test forward foothold proposals motivated by observed forward COM motion.

Reuses measured queries, original constraints and the exact foot-space IK and
rollout worker. The new search range is internal candidate authority only.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.probe_task_space_landing_v1 import evaluate
from scripts.check_constrained_recovery_suffix_v1 import source_context,state_from_row
from scripts.search_preserved_early_support_v1 import initialize
from scripts.recovery_family_queries_v1 import transfer_reference
from scripts.audit_joint_sequence_proposal_v1 import plain


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['plan','previous','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Fresh output required')
    a.output.mkdir(parents=True)
    read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def write(name,value):
        path=a.output/name;tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(plain(value),indent=2,allow_nan=False)+'\n');tmp.replace(path)
    try:
        previous=read(a.previous/'result.json');pins=dict(previous['source_sha256'])
        plan=read(a.plan);source=read(plan['source']);_,knots,_,geometry,context=source_context(source['configuration']['source'])
        initialize(*context)
        placements=[dict(front_down=0.,rear_down=0.,front_forward=0.)]+[
            dict(front_down=d,rear_down=0.,front_forward=x) for x in [.06,.09,.12,.15] for d in [0.,.025]]
        jobs=[]
        for case in [87,85,96,71]:
            original=next(r for r in previous['rows'] if r['case']==case and all(x==0 for x in r['placement'].values()))
            path=a.previous/original['file'];r=read(path)['result'];pins[str(path.resolve())]=sha(path)
            query=dict(q=r['initial_q'],v=r['initial_v'],contacts=r['initial_contacts'],target=r['initial_previous_command'])
            aligned,_=transfer_reference(context,state_from_row(query))
            for placement in placements:jobs.append(dict(case=case,context=aligned,query=query,
                original_knots=knots,geometry=geometry,placement=placement))
        for name,module in list(sys.modules.items()):
            if name.startswith('scripts.') and getattr(module,'__file__',None):
                path=str(Path(module.__file__).resolve());pins[path]=sha(path)
        pins[str(Path(__file__).resolve())]=sha(__file__)
        for path,h in pins.items():
            if sha(path)!=h:raise ValueError('Changed source '+path)
        deadline=time.monotonic()+900.
        for job in jobs:job['deadline']=deadline
        write('status.json',dict(state='MODEL_SEARCH',candidates=len(jobs),maximum_seconds=900,native_execution=False))
        rows=[]
        with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
            futures={pool.submit(evaluate,job):i for i,job in enumerate(jobs)}
            for future in as_completed(futures):
                row=future.result();name=f'candidate_{futures[future]:03d}.json';write(name,row)
                r=row['result'];compact={k:v for k,v in row.items() if k not in ['result','knots_rad']}
                compact.update(file=name,sha256=sha(a.output/name),result={k:v for k,v in r.items() if k not in
                    ['trajectory','initial_q','initial_v','initial_contacts','initial_previous_command']} if r else None)
                rows.append(compact);print(json.dumps(plain(dict(case=row['case'],placement=row['placement'],
                    completed=r['completed_model_steps'] if r else None,verified=row['independently_verified']))),flush=True)
        for path,h in pins.items():
            if sha(path)!=h:raise ValueError('Changed source '+path)
        write('result.json',dict(scope=__doc__,rows=rows,source_sha256=pins,native_execution=False,valid_expert_labels=False))
        write('status.json',dict(state='COMPLETED_MODEL_SEARCH',candidates=len(rows),
            verified=sum(r['independently_verified'] for r in rows),native_execution=False))
    except BaseException as e:
        write('status.json',dict(state='FAILED',error=repr(e)));raise


if __name__=='__main__':main()
