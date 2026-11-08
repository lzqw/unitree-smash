"""Test coupled position-stop predictions from measured recovery states.

Offline diagnosis of horizon drift, not a new native MPC execution. The
reference clock, executed command history and full remaining recovery checks
are retained. No optimizer may replace the supplied measured starting state.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
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
from scripts.hip_mount_path_constraint_v1 import HipMountPathConstraintV1
from scripts.position_limited_recovery_suffix_v1 import evaluate_suffix
from scripts.timed_whole_body_recovery_v1 import build_timed_plan
from scripts.audit_joint_sequence_proposal_v1 import plain


def evaluate(job):
    w=_WORKER
    def run():
        return evaluate_suffix(w['controller'],state_from_row(job['query']),build_timed_plan(w,job['plan']),
            w['spheres'],HipMountPathConstraintV1(w['geometry'],job['geometry'],margin_rad=.02),w['protocol'],
            control_tick=job['tick'],substeps=160,deadline=job['deadline'])
    result=run();replay_exact=False
    if result['accepted']:
        replay=run()
        replay_exact=(replay['accepted'] and len(replay['trajectory'])==len(result['trajectory']) and all(
            np.array_equal(x[k],y[k]) for x,y in zip(result['trajectory'],replay['trajectory'])
            for k in ['q','v','target','contacts']))
    return dict(tick=job['tick'],result=result,independent_replay_exact=replay_exact)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['model-plan','run','bundle','native-audit','output']:
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Fresh evidence required')
    read=lambda path:json.loads(Path(path).read_text());sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    setup=read(a.model_plan);m=read(a.bundle/'manifest.json');audit=read(a.native_audit)
    native=read(a.run/'sequence_native_result.json')
    if read(a.run/'job.json')['status']!='SUCCEEDED' or not audit['passed_dispatch_integrity'] or not native['query_match']['passed']:
        raise ValueError('Completed matching actual run required')
    source=read(setup['source']);_,_,_,geometry,context=source_context(source['configuration']['source'])
    context=list(copy.deepcopy(context));initial=m['query']
    if context[0]!=m['configuration']:raise ValueError('Physical interface differs')
    context[1].update(initial_q=initial['q'],initial_v=initial['v'],initial_contacts=initial['contacts'],
        actual_previous_command=initial['target'],tick=280)
    with np.load(a.bundle/m['bundle_name'],allow_pickle=False) as z:
        context[4]=z['tracking_reference_q'].copy();context[5]=z['tracking_reference_v'].copy()
    context[6]=m['tracking_gains']
    with np.load(a.run/'sequence_native_trace.npz',allow_pickle=False) as z:trace={k:z[k].copy() for k in z.files}
    ticks=np.asarray([r['tick'] for r in native['records']],int)
    if not np.array_equal(ticks,np.arange(280,800)):raise ValueError('Complete actual suffix required')
    if np.max(abs(trace['previous_target_rad'][1:]-trace['final_target_rad'][:-1]))>5e-6:
        raise ValueError('Actual command history differs')
    pins=dict(audit['source_sha256'])
    for path in [a.model_plan,a.bundle/'manifest.json',a.bundle/m['bundle_name'],a.native_audit,Path(__file__).resolve()]:
        pins[str(path.resolve())]=sha(path)
    for name,module in list(sys.modules.items()):
        if name.startswith('scripts.') and getattr(module,'__file__',None):
            path=str(Path(module.__file__).resolve());digest=sha(path)
            if path in pins and pins[path]!=digest:raise ValueError('Pinned source changed '+path)
            pins[path]=digest
    for path,digest in pins.items():
        if sha(path)!=digest:raise ValueError('Input changed '+path)
    a.output.mkdir(parents=True)
    def write(name,value):
        path=a.output/name;tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(plain(value),indent=2,allow_nan=False)+'\n');tmp.replace(path)
    write('inputs.json',dict(configuration={k:str(v) for k,v in vars(a).items()},source_sha256=pins,
        fixed_query_ticks=[280,320,400,500,600,700],max_seconds=900,workers=2,physical_constraints_changed=False))
    began=time.monotonic();deadline=began+900;jobs=[]
    for tick in [280,320,400,500,600,700]:
        i=tick-280
        query=dict(q=trace['q'][i],v=trace['v'][i],contacts=trace['contacts'][i],target=trace['previous_target_rad'][i])
        jobs.append(dict(tick=tick,query=query,plan=m['proposal'],geometry=geometry,deadline=deadline))
    try:
        write('status.json',dict(state='PREDICTING_POSITION_LIMITED_SUFFIXES'))
        with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=tuple(context)) as pool:
            rows=[]
            for row in pool.map(evaluate,jobs):
                rows.append(row);write(f"tick_{row['tick']}.json",row)
                print(json.dumps(dict(tick=row['tick'],accepted=row['result']['accepted'],
                    completed=row['result']['completed_model_steps'],failure=row['result']['failure'])),flush=True)
        for path,digest in pins.items():
            if sha(path)!=digest:raise ValueError('Input changed during diagnosis '+path)
        write('result.json',dict(scope=__doc__,rows=rows,source_sha256=pins,wall_seconds=time.monotonic()-began,
            native_outcome=audit['selected_case_outcome'],strict_untouched_isolation_passed=audit['strict_untouched_isolation_passed'],
            new_native_execution=False,valid_expert_labels=False,physical_constraints_changed=False))
        write('status.json',dict(state='COMPLETED',accepted_ticks=[r['tick'] for r in rows if r['independent_replay_exact']]))
    except BaseException as error:
        write('status.json',dict(state='FAILED',error=repr(error)));raise


if __name__=='__main__':main()
