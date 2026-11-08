"""One-step model checks on the actual student prefix of recovery families.

The next sample's previous target is the command actually executed over the
interval. All intervals keep actual state and command history; no reference
trajectory or newly optimized action is substituted.
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
from scripts.check_constrained_recovery_suffix_v1 import source_context,state_from_row
from scripts.search_preserved_early_support_v1 import initialize
from scripts.search_landing_support_tail_v3 import _WORKER
from scripts.native_recovery_query_bank_v1 import state_from_sample
from scripts.audit_native_feedback_prediction_v1 import predict
from scripts.audit_joint_sequence_proposal_v1 import plain


def check(job):
    result=predict(job)
    after=state_from_row(job['after'])
    gap,_=_WORKER['geometry'].evaluate(after.q)
    result.update(case=job['case'],actual_final_nonfoot_gap_m=float(gap.min()))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Fresh audit output required')
    read=lambda p:json.loads(Path(p).read_text())
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    plan=read(args.plan);audit=read(Path(plan['bank_followup'])/'query_bank_audit.json')
    bank=Path(plan['bank_run']);meta=read(bank/'recovery_query_bank.json')
    if not audit['passed'] or not meta['complete_window']:raise ValueError('Passive complete native bank required')
    source=read(plan['source']);_,_,_,_,context=source_context(source['configuration']['source'])
    initialize(*context);model=_WORKER['controller'].qp.model
    with np.load(bank/'recovery_query_bank.npz',allow_pickle=False) as z:
        arrays={k:z[k].copy() for k in z.files}
    def actual(tick,case):
        indices=np.flatnonzero(arrays['control_ticks']==tick)
        if len(indices)!=1:raise ValueError('Unique actual tick required')
        i=int(indices[0]);sample={k:v[i,case] for k,v in arrays.items() if k!='control_ticks'}
        if sample['episode_index']!=0 or not sample['unfinished_first_episode']:
            raise ValueError('Cannot cross reset or terminated native interval')
        state=state_from_sample(model,sample,meta['origins_world'][case])
        return dict(q=state.q,v=state.v,contacts=state.contacts,target=state.previous_command_target),sample
    jobs=[]
    for case in [87,85]:
        for tick in range(270,330):
            before,sample=actual(tick,case);after,_=actual(tick+1,case)
            jobs.append(dict(case=case,tick=tick,before=before,after=after,
                target=after['target'],foot_force=sample['foot_force_world']))
    pins=dict(audit['source_sha256'])
    for path in [args.plan,bank/'recovery_query_bank.npz',bank/'recovery_query_bank.json',Path(__file__).resolve()]:
        pins[str(path.resolve())]=sha(path)
    for name,module in list(sys.modules.items()):
        if name.startswith('scripts.') and getattr(module,'__file__',None):
            path=str(Path(module.__file__).resolve());pins[path]=sha(path)
    for path,digest in pins.items():
        if sha(path)!=digest:raise ValueError('Changed source '+path)
    began=time.monotonic()
    with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=context) as pool:
        rows=list(pool.map(check,jobs))
    summaries={}
    for case in [87,85]:
        chosen=[r for r in rows if r['case']==case]
        subsets={'all_completed':[r for r in chosen if r['completed']],
            'both_endpoints_clear_1mm':[r for r in chosen if r['completed'] and
                min(r['actual_initial_nonfoot_gap_m'],r['actual_final_nonfoot_gap_m'])>.001]}
        summaries[str(case)]={}
        for name,subset in subsets.items():
            metrics=[k for k in subset[0] if '_error_' in k] if subset else []
            summaries[str(case)][name]=dict(count=len(subset),metrics={k:dict(
                maximum=max(r[k] for r in subset),median=float(np.median([r[k] for r in subset]))) for k in metrics})
    for path,digest in pins.items():
        if sha(path)!=digest:raise ValueError('Source changed during audit '+path)
    result=dict(scope=__doc__,rows=rows,summaries=summaries,source_sha256=pins,
        wall_seconds=time.monotonic()-began,new_native_execution=False,valid_expert_labels=False,
        limits=['Nonfoot/self-contact impulses are not modeled.',
            'Endpoint clearance is not a guarantee of no contact within the interval.',
            'This checks actual student commands, not the newly searched controller or its robustness.'])
    tmp=args.output.with_suffix('.tmp');tmp.write_text(json.dumps(plain(result),indent=2,allow_nan=False)+'\n');tmp.replace(args.output)
    print(json.dumps(summaries))


if __name__=='__main__':main()
