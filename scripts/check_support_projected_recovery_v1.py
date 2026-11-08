"""Compare support-projected feedback using the unchanged full-episode model."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.check_constrained_recovery_suffix_v1 import source_context, state_from_row
from scripts.search_preserved_early_support_v1 import initialize
from scripts.search_landing_support_tail_v3 import _WORKER
from scripts.hip_mount_path_constraint_v1 import HipMountPathConstraintV1
from scripts.speed_limited_recovery_suffix_v1 import evaluate_suffix
from scripts.timed_whole_body_recovery_v1 import build_timed_plan
from scripts.support_projected_tracking_v1 import build_support_projected_plan
from scripts.audit_joint_sequence_proposal_v1 import plain


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['model-plan','generation','bundle','output']:
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Fresh evidence required')
    read=lambda path:json.loads(Path(path).read_text())
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    setup=read(a.model_plan);g=read(a.generation);m=read(a.bundle/'manifest.json')
    old=min(g['fine'],key=lambda r:r['result']['rank'])
    source=read(setup['source']);_,_,_,geometry,context=source_context(source['configuration']['source'])
    context=list(copy.deepcopy(context));query=m['query']
    if context[0]!=m['configuration']:raise ValueError('Physical configuration differs')
    for key,field in [('initial_q','q'),('initial_v','v'),('initial_contacts','contacts'),('initial_previous_command','target')]:
        if not np.array_equal(old['result'][key],query[field]):raise ValueError('Initial state differs')
    context[1].update(initial_q=query['q'],initial_v=query['v'],initial_contacts=query['contacts'],
        actual_previous_command=query['target'],tick=280)
    with np.load(a.bundle/m['bundle_name'],allow_pickle=False) as z:
        context[4]=z['tracking_reference_q'].copy();context[5]=z['tracking_reference_v'].copy()
    context[6]=m['tracking_gains'];initialize(*context)
    pins=dict(m['source_sha256'])
    for path in [a.model_plan,a.generation,a.bundle/'manifest.json',a.bundle/m['bundle_name'],Path(__file__).resolve()]:
        pins[str(path.resolve())]=sha(path)
    for name,module in list(sys.modules.items()):
        if name.startswith('scripts.') and getattr(module,'__file__',None):
            path=str(Path(module.__file__).resolve());digest=sha(path)
            if path in pins and pins[path]!=digest:raise ValueError('Source changed '+path)
            pins[path]=digest
    for path,digest in pins.items():
        if sha(path)!=digest:raise ValueError('Pinned input changed '+path)
    a.output.mkdir(parents=True)
    def write(name,value):
        path=a.output/name;tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(plain(value),indent=2,allow_nan=False)+'\n');tmp.replace(path)
    write('inputs.json',dict(configuration={k:str(v) for k,v in vars(a).items()},source_sha256=pins,
        physical_constraints_changed=False,max_seconds=600))
    deadline=time.monotonic()+600
    def rollout(plan,builder):
        w=_WORKER
        return evaluate_suffix(w['controller'],state_from_row(query),builder(w,plan),w['spheres'],
            HipMountPathConstraintV1(w['geometry'],geometry,margin_rad=.02),w['protocol'],
            control_tick=280,substeps=160,deadline=deadline)
    try:
        write('status.json',dict(state='BASELINE_REPLAY'))
        baseline=rollout(old['plan'],build_timed_plan)
        for key in ['accepted','completed_model_steps','failure','rank','terminal']:
            if baseline[key]!=old['result'][key]:raise ValueError('Baseline changed '+key)
        def exact(rows,reference):
            return len(rows)==len(reference) and all(np.array_equal(x[k],y[k]) for x,y in zip(rows,reference)
                for k in ['q','v','target','contacts'])
        if not exact(baseline['trajectory'],old['result']['trajectory']):raise ValueError('Baseline trajectory differs')
        write('baseline.json',dict(plan=old['plan'],result=baseline))
        plan=copy.deepcopy(old['plan']);plan['contract']='support_projected_timed_recovery_v1'
        write('status.json',dict(state='PROJECTED_REFERENCE_ROLLOUT'))
        candidate=rollout(plan,build_support_projected_plan)
        if not exact(candidate['trajectory'][:35],baseline['trajectory'][:35]):raise ValueError('Protected early prefix changed')
        verified=None
        if candidate['accepted']:
            replay=rollout(plan,build_support_projected_plan)
            if replay['accepted'] and exact(replay['trajectory'],candidate['trajectory']):verified=replay
        for path,digest in pins.items():
            if sha(path)!=digest:raise ValueError('Pinned input changed during execution '+path)
        write('result.json',dict(baseline=dict(plan=old['plan'],result=baseline),candidate=dict(plan=plan,result=candidate),
            verified=verified,source_sha256=pins,protected_prefix_exact=True,native_execution=False,
            valid_expert_labels=False,physical_constraints_changed=False))
        summary=dict(state='COMPLETED',baseline_completed=baseline['completed_model_steps'],
            candidate_completed=candidate['completed_model_steps'],candidate_accepted=candidate['accepted'],
            candidate_failure=candidate['failure'],verified=verified is not None)
        write('status.json',summary);print(json.dumps(plain(summary)),flush=True)
    except BaseException as error:
        write('status.json',dict(state='FAILED',error=repr(error)));raise


if __name__=='__main__':main()
