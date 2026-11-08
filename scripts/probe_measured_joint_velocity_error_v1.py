"""Probe fixed recovery feedback against a measured one-step joint error.

The three queries are the actual endpoint state and plus/minus its preceding
observed prediction residual in joint velocity. This is model sensitivity,
not a probabilistic error bound, native disturbance test or robust guarantee.
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

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.check_constrained_recovery_suffix_v1 import source_context
from scripts.search_preserved_early_support_v1 import initialize
from scripts.search_landing_support_tail_v3 import _WORKER
from scripts.current_state_recovery_planner_v1 import predict_plan
from scripts.prepare_native_replanned_suffix_v1 import validate
from scripts.audit_joint_sequence_proposal_v1 import plain


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['attribution','source','run','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if a.output.exists():raise ValueError('Fresh sensitivity output required')
    attr=json.loads(a.attribution.read_text());source=json.loads(a.source.read_text())
    tick,_=validate(source)
    if attr['maximum_original_error_reproduction_difference']!=0.:
        raise ValueError('Reproduced native/model response evidence required')
    worst=max(attr['rows'],key=lambda r:r['errors']['joint_velocity_max_rad_s'])
    now=worst['control_tick']+1
    _,knots,_,audit,context=source_context(source['configuration']['source']);initialize(*context)
    model=_WORKER['controller'].qp.model
    run=a.run.resolve();meta=json.loads((run/'sequence_native_result.json').read_text())
    index={r['control_tick']:i for i,r in enumerate(meta['records'])};i=index[now]
    pins=dict(attr['source_sha256'])
    for name in ['sequence_native_result.json','sequence_native_trace.npz']:
        if pins.get(str(run/name))!=sha(run/name):raise ValueError('Native source differs')
    for path in [a.source,a.attribution,Path(__file__)]:pins[str(path.resolve())]=sha(path)
    for name,module in list(sys.modules.items()):
        if name.startswith('scripts.') and getattr(module,'__file__',None):
            path=Path(module.__file__).resolve();pins[str(path)]=sha(path)
    for path,h in pins.items():
        if sha(path)!=h:raise ValueError('Changed input: '+path)
    with np.load(run/'sequence_native_trace.npz',allow_pickle=False) as z:
        query=dict(q=z['q'][i].copy(),v=z['v'][i].copy(),contacts=z['contacts'][i].copy(),
            target=z['previous_target_rad'][i].copy())
    plan=dict(original_knots_rad=knots,corrections=[dict(control_tick=tick,
        previous_command=source['query']['target'],knots_rad=source['best']['corrections_rad'])])
    error=np.asarray(worst['per_joint_velocity_error_rad_s'])
    if error.shape!=(18,) or not np.isfinite(error).all():raise ValueError('Finite measured joint residual required')
    queries=[]
    for sign,label in [(0,'actual_native_endpoint'),(1,'plus_measured_joint_velocity_residual'),(-1,'minus_measured_joint_velocity_residual')]:
        q=copy.deepcopy(query);q['v'][model.v_indices]+=sign*error
        if not all(np.array_equal(q[k],query[k]) for k in ['q','contacts','target']):
            raise ValueError('Probe changed pose, contact or command history')
        queries.append((label,q))
    began=time.monotonic();deadline=began+600;rows=[]
    with ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=context) as pool:
        jobs={pool.submit(predict_plan,plan,q,now,0,audit,160,deadline):(label,q) for label,q in queries}
        for future in as_completed(jobs):
            label,q=jobs[future];result=future.result()['result']
            rows.append(dict(probe=label,query=q,result=result))
            print(json.dumps(dict(probe=label,accepted=result['accepted'],completed=result['completed_model_steps'],
                failure=result['failure'])),flush=True)
    for path,h in pins.items():
        if sha(path)!=h:raise ValueError('Source changed during sensitivity check')
    result=dict(scope=__doc__,source_sha256=pins,source_interval_tick=worst['control_tick'],
        control_tick=now,joint_order=model.joint_order,measured_velocity_residual_rad_s=error,
        rows=rows,wall_seconds=time.monotonic()-began,native_execution=False,valid_expert_labels=False,
        reoptimization=False,limitations=['This residual is one observation, not a validated uncertainty set.',
            'Only joint velocity is perturbed; correlated pose/contact/model errors are not covered.',
            'The fixed plan is followed by its feedback law; no failed result is silently reoptimized.'])
    a.output.write_text(json.dumps(plain(result),indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
