"""Compare a model-rejected proposal with its full-ground native execution.

Actual command and measured state initialize every prediction. Physics and
task labels remain unchanged; model geometric clearance is not a measured
contact event. Strict isolation failures are reported, never waived.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.check_constrained_recovery_suffix_v1 import source_context
from scripts.search_preserved_early_support_v1 import initialize
from scripts.search_landing_support_tail_v3 import _WORKER
from scripts.audit_native_feedback_prediction_v1 import predict
from scripts.audit_joint_sequence_proposal_v1 import plain


def check(job):
    result=predict(job)
    distances,_=_WORKER['geometry'].evaluate(np.asarray(job['after']['q']))
    result['actual_final_model_geometry_gap_m']=float(distances.min())
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['model-plan','run','bundle','native-audit','head-audit','output']:
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Fresh result required')
    read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    audit=read(args.native_audit);head=read(args.head_audit);job=read(args.run/'job.json')
    native=read(args.run/'sequence_native_result.json');manifest=read(args.bundle/'manifest.json')
    if (job['status']!='SUCCEEDED' or not audit['passed_dispatch_integrity'] or
            not native['query_match']['passed'] or native['model_accepted'] or manifest['model_accepted']):
        raise ValueError('Completed explicit model-rejected native diagnostic required')
    model_plan=read(args.model_plan);source=read(model_plan['source'])
    _,_,_,_,context=source_context(source['configuration']['source'])
    initialize(*context)
    if context[0]!=manifest['configuration']:raise ValueError('Prediction/native interface differs')
    with np.load(args.run/'sequence_native_trace.npz',allow_pickle=False) as z:
        trace={k:z[k].copy() for k in z.files}
    ticks=np.asarray([r['tick'] for r in native['records']],int);n=len(ticks)
    if n<2 or any(len(v)!=n for v in trace.values()) or not np.all(np.diff(ticks)==1):
        raise ValueError('Aligned consecutive native first-episode records required')
    if np.max(abs(trace['previous_target_rad'][1:]-trace['final_target_rad'][:-1]))>5e-6:
        raise ValueError('Actual command history differs')
    # Fixed sampling schedule chosen before inspecting candidate outcomes.
    indices=sorted(set(range(min(50,n-1)))|set(range(50,n-1,10)))
    def state(i):
        return dict(q=trace['q'][i],v=trace['v'][i],contacts=trace['contacts'][i],
            target=trace['previous_target_rad'][i])
    jobs=[dict(tick=int(ticks[i]),before=state(i),after=state(i+1),
        target=trace['final_target_rad'][i],foot_force=trace['foot_force_world_n'][i]) for i in indices]
    pins=dict(audit['source_sha256'])
    for evidence in [model_plan,head]:
        for p,h in evidence['source_sha256'].items():
            if p in pins and pins[p]!=h:raise ValueError('Evidence disagrees '+p)
            pins[p]=h
    for p in [args.model_plan,args.native_audit,args.head_audit,args.bundle/'manifest.json',Path(__file__).resolve()]:
        pins[str(p.resolve())]=sha(p)
    for name,module in list(sys.modules.items()):
        if name.startswith('scripts.') and getattr(module,'__file__',None):
            p=str(Path(module.__file__).resolve());h=sha(p)
            if p in pins and pins[p]!=h:raise ValueError('Pinned implementation changed '+p)
            pins[p]=h
    for p,h in pins.items():
        if sha(p)!=h:raise ValueError('Changed source '+p)
    began=time.monotonic()
    with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=context) as pool:
        rows=list(pool.map(check,jobs))
    summaries={}
    completed=[r for r in rows if r['completed']]
    clear=[r for r in completed if min(r['actual_initial_nonfoot_gap_m'],r['actual_final_model_geometry_gap_m'])>.001]
    for label,subset in [('all_completed',completed),('both_endpoints_model_geometry_clear_1mm',clear)]:
        keys=[k for k in subset[0] if '_error_' in k] if subset else []
        summaries[label]=dict(count=len(subset),errors={k:dict(maximum=max(r[k] for r in subset),
            median=float(np.median([r[k] for r in subset]))) for k in keys})
    for p,h in pins.items():
        if sha(p)!=h:raise ValueError('Source changed during diagnostic '+p)
    case=manifest['case'];selected_head=next(r for r in head['records'] if r['case']==case)
    result=dict(scope=__doc__,case=case,intervals=len(rows),completed=len(completed),rows=rows,
        summary=summaries,wall_seconds=time.monotonic()-began,
        original_model_rejection=manifest['model_rejection'],original_model_completed_steps=manifest['model_completed_steps'],
        native_outcome=audit['selected_case_outcome'],complete5=audit['complete5'],
        actual_first_scored_nonfoot_s=audit['selected_first_scored_nonfoot_s'],head_observation=selected_head,
        strict_untouched_isolation_passed=audit['strict_untouched_isolation_passed'],
        source_sha256=pins,valid_expert_labels=False,automatic_promotion=False,new_native_execution=False,
        limitations=['Offline one-step predictions are not a new recovery execution.',
            'Nonfoot/self-contact impulses remain outside this model.',
            'Model geometric endpoint clearance does not exclude a contact within the interval.',
            'Head-force observations and original task scoring are reported separately.'])
    tmp=args.output.with_suffix('.tmp');tmp.write_text(json.dumps(plain(result),indent=2,allow_nan=False)+'\n');tmp.replace(args.output)
    print(json.dumps(plain(dict(case=case,completed=len(completed),summary=summaries,native_outcome=result['native_outcome']))))


if __name__=='__main__':main()
