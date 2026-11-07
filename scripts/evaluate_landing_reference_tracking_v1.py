"""Paired fine-model evaluation of sustained support tracking on nearby states."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from functools import partial
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.search_landing_support_tail_v3 import initialize_worker,_WORKER,SupportTailReference
from scripts.perturb_early_support_domain_v1 import run_probe
from scripts.landing_reference_tracking_v1 import LandingReferenceTrackingV1
from scripts.audit_joint_sequence_proposal_v1 import plain


def factory(model,interface,initial,knots,*,fixed_knots,reference_q,reference_v,gains):
    base=SupportTailReference(model,interface,initial,knots,fixed_knots=fixed_knots)
    return LandingReferenceTrackingV1(base,reference_q,reference_v,**gains)


def initialize(configuration,query,fixed_knots,spheres,reference_q,reference_v,gains):
    initialize_worker(configuration,query,fixed_knots,spheres)
    _WORKER['factory']=partial(factory,fixed_knots=fixed_knots,reference_q=reference_q,reference_v=reference_v,gains=gains)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe-source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-seconds',type=float,default=900.);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    if a.output.exists() or not 300<=a.max_seconds<=1800 or not 1<=a.workers<=4:
        raise ValueError('Fresh output and bounded CPU evaluation required')
    read=lambda f:json.loads(Path(f).read_text());sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    probes=read(a.probe_source)
    if probes['compute_expired_probes'] or probes['nominal_maximum_difference']!=0:
        raise ValueError('Completed reproducible paired baseline required')
    # Provenance identifies the already verified nominal full-episode search.
    sources=[Path(f) for f in probes['source_sha256'] if Path(f).name=='refine_support_tail_fine160_v1.json']
    if len(sources)!=1:raise ValueError('One immutable nominal source required')
    source=read(sources[0]);tail=read(source['configuration']['source']);parent=read(tail['configuration']['source_search'])
    cfg=parent['configuration'];query=read(cfg['query']);manifest=read(Path(cfg['bundle'])/'manifest.json')
    states=source['independent_replay']['trajectory']
    reference_q=np.asarray([query['initial_q']]+[s['q'] for s in states[:-1]])
    reference_v=np.asarray([query['initial_v']]+[s['v'] for s in states[:-1]])
    gains=dict(position_gain=.4,velocity_gain_s=.06,orientation_gain=.2,angular_gain_s=.02,
               max_translation_m=.025,max_rotation_rad=.04,max_joint_delta_rad=.08)
    hashes=dict(probes['source_sha256'])
    for f in [a.probe_source,Path(__file__),ROOT/'scripts/landing_reference_tracking_v1.py']:
        hashes[str(f.resolve())]=sha(f)
    for f,h in hashes.items():
        if sha(f)!=h:raise ValueError('Changed source '+f)
    initargs=(manifest['configuration'],query,source['fixed_early_knots_rad'],cfg['sphere_audit'],reference_q,reference_v,gains)
    start=time.monotonic();deadline=start+a.max_seconds;rows=[]
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=initargs) as pool:
        jobs=[pool.submit(run_probe,(s['probe'],source['best_knots_rad'],deadline)) for s in probes['rows']]
        for future in as_completed(jobs):
            row=future.result();rows.append(row)
            print(json.dumps(plain(dict(probe=row['probe'],result={k:v for k,v in row['result'].items() if k!='trajectory'}))),flush=True)
            tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(plain(rows),allow_nan=False)+'\n');tmp.replace(a.output.with_suffix('.partial.json'))
    lookup={r['probe']['name']:r for r in rows};rows=[lookup[r['probe']['name']] for r in probes['rows']]
    nominal=rows[0]['result'];expected=source['independent_replay']
    nominal_error=None
    if len(nominal['trajectory'])==520:
        nominal_error=max(float(np.max(abs(np.asarray(x[k])-y[k]))) for x,y in zip(nominal['trajectory'],expected['trajectory']) for k in ['target','q','v'])
    nominal_preserved=bool(nominal['accepted'] and nominal_error is not None and nominal_error<=1e-12)
    for f,h in hashes.items():
        if sha(f)!=h:raise ValueError('Changed source during evaluation '+f)
    paired=[dict(name=x['probe']['name'],baseline_accepted=b['result']['accepted'],candidate_accepted=x['result']['accepted'],
        baseline_completed_steps=b['result']['completed_model_steps'],candidate_completed_steps=x['result']['completed_model_steps']) for b,x in zip(probes['rows'],rows)]
    output=dict(scope=__doc__,source_sha256=hashes,configuration=gains,rows=rows,paired=paired,
        nominal_preserved=nominal_preserved,nominal_maximum_difference=nominal_error,
        accepted_model_probes=sum(x['result']['accepted'] for x in rows),probe_count=len(rows),
        compute_expired_probes=sum((x['result'].get('failure') or {}).get('reason')=='declared_compute_budget' for x in rows),
        wall_seconds=time.monotonic()-start,substeps=160,native_execution=False,valid_expert_labels=False,
        limitations=['Fixed deterministic axis probes are not a success-rate estimate or invariant-region proof.',
            'This is a candidate feedback execution layer, not a new online MPC.',
            'Reference states are planned nominal states; no future native observations are queried.'])
    a.output.write_text(json.dumps(plain(output),indent=2,allow_nan=False)+'\n')
    if not nominal_preserved:raise ValueError('Candidate did not preserve nominal behavior; inspect negative result')


if __name__=='__main__':main()
