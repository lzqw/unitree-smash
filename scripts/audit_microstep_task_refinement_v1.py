"""Refine true ball-side recovery failures after substep geometry restoration."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import scripts.landing_sequential_recovery_search_v1 as search
from scripts.microstep_ground_sequence_v1 import make_proposal
from scripts.audit_geometry_recovery_task_v1 import query as refine_query
from scripts.audit_microstep_recovery_v1 import query as continuation_query


def query(args):
    bundle,geometry,row=args;started=time.perf_counter()
    old_propose=search.propose_sequence;old_rollout=search.rollout_for_search
    def weighted_rollout(*args):
        result=old_rollout(*args);H=len(result['profile'])
        result['cost']+=sum((10. if i==H-1 else 1./H)*np.sum(n['following'].joint_velocity**2)
            for i,n in enumerate(result['profile']))
        return result
    seed=dict(row,newly_model_feasible=not row['after']['accepted'])
    search.propose_sequence=make_proposal(2.);search.rollout_for_search=weighted_rollout
    try:refined=refine_query((bundle,geometry,seed))
    finally:search.propose_sequence=old_propose;search.rollout_for_search=old_rollout
    continued=continuation_query((bundle,geometry,refined,1))
    return dict(case=row['case'],ball5_except_recovery=True,native_execution=False,
        refinement=refined,continuation_check=continued,wall_seconds=time.perf_counter()-started)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['bundle','geometry','start-report','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve evidence')
    start=json.loads(a.start_report.read_text())
    for name,digest in start['source_sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:raise ValueError('Changed seed '+name)
    manifest=json.loads((a.bundle/'manifest.json').read_text())
    metrics=Path(manifest['baseline'])/'metrics.json'
    outcomes={r['training_index']:r for r in json.loads(metrics.read_text())['complete5_case_outcomes']}
    rows=[r for r in start['records'] if r['case'] in [3,18]]
    if len(rows)!=2 or any(not outcomes[r['case']]['ball5_except_recovery'] or outcomes[r['case']]['complete5'] for r in rows):
        raise ValueError('Require original ball-side-qualified recovery failures')
    a.output.parent.mkdir(parents=True,exist_ok=True);results=[]
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(query,(a.bundle,a.geometry,r)) for r in rows]
        for future in as_completed(futures):
            r=future.result();results.append(r);c=r['continuation_check']
            print(json.dumps(dict(case=r['case'],accepted=c['after']['accepted'],
                terminal_risk=c['after']['terminal_risk'],ground_violation_m=c['after']['ground_violation_m'],
                verified_continuation_s=c['verified_continuation_s'])),flush=True)
            a.output.with_suffix('.partial.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    files=[Path(__file__),a.start_report,a.bundle/'manifest.json',a.geometry,metrics]
    files += [Path(m.__file__) for n,m in sys.modules.items() if n.startswith('scripts.') and getattr(m,'__file__',None)]
    result=dict(scope=__doc__,native_execution=False,valid_expert_labels=False,
        records=sorted(results,key=lambda r:r['case']),
        source_sha256={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
