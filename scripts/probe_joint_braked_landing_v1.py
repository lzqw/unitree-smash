"""Check joint-braking feedback on the four foot-space candidate families.

Finite offline model trials with measured initial state and full original
remainder. No short-horizon success, native success, or expert-label claim.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import itertools
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.check_constrained_recovery_suffix_v1 import source_context, state_from_row
from scripts.search_preserved_early_support_v1 import initialize
from scripts.search_landing_support_tail_v3 import _WORKER
from scripts.recovery_family_queries_v1 import transfer_reference
from scripts.hip_mount_path_constraint_v1 import HipMountPathConstraintV1
from scripts.terminal_attitude_feedback_v1 import TerminalAttitudeFeedbackV1
from scripts.speed_limited_recovery_suffix_v1 import evaluate_suffix
from scripts.landing_joint_braking_reference_v1 import LandingJointBrakingReferenceV1
from scripts.audit_joint_sequence_proposal_v1 import plain


def evaluate(job):
    initialize(*job['context']); w=_WORKER; controller=w['controller']; state=state_from_row(job['query'])
    def rollout():
        base=w['factory'](controller.qp.model,controller.qp.interface,state,np.asarray(job['knots']))
        tracking=TerminalAttitudeFeedbackV1(base,proportional=.3,damping_s=.04,maximum_joint_delta_rad=.08)
        policy=LandingJointBrakingReferenceV1(tracking,**job['braking'])
        target,_=policy.action(80,state,state.previous_command_target)
        if not np.array_equal(target,state.previous_command_target):raise ValueError('Discontinuous first command')
        return evaluate_suffix(controller,state,policy,w['spheres'],
            HipMountPathConstraintV1(w['geometry'],job['geometry'],margin_rad=.02),w['protocol'],
            control_tick=280,substeps=160,deadline=job['deadline'],save=True)
    result=rollout(); verified=False
    if result['accepted']:
        replay=rollout()
        verified=replay['accepted'] and len(replay['trajectory'])==len(result['trajectory'])==520 and all(
            np.array_equal(a[k],b[k]) for a,b in zip(result['trajectory'],replay['trajectory'])
            for k in ['q','v','contacts','target'])
    return dict(case=job['case'],braking=job['braking'],knots_rad=job['knots'],result=result,
        independently_verified=bool(verified),native_execution=False,valid_expert_labels=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['plan','foot-search','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Fresh output required')
    a.output.mkdir(parents=True)
    read=lambda p:json.loads(Path(p).read_text()); sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def write(name,value):
        path=a.output/name; tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(plain(value),indent=2,allow_nan=False)+'\n');tmp.replace(path)
    try:
        search=read(a.foot_search/'result.json'); pins=dict(search['source_sha256'])
        for path,h in pins.items():
            if sha(path)!=h:raise ValueError('Changed source '+path)
        plan=read(a.plan);source=read(plan['source']);_,_,_,geometry,context=source_context(source['configuration']['source'])
        initialize(*context);jobs=[]
        controls=[dict(gain_s=0.,lookahead_s=0.)]+[dict(gain_s=g,lookahead_s=t)
            for g,t in itertools.product([.04,.08,.16],[0.,.03,.06])]
        for case in [87,85,96,71]:
            rows=[r for r in search['rows'] if r['case']==case and r['result']]
            best=min(rows,key=lambda r:r['result']['rank']); path=a.foot_search/best['file']; parent=read(path)
            pins[str(path.resolve())]=sha(path)
            result=parent['result'];query=dict(q=result['initial_q'],v=result['initial_v'],
                contacts=result['initial_contacts'],target=result['initial_previous_command'])
            aligned,_=transfer_reference(context,state_from_row(query))
            for braking in controls:
                jobs.append(dict(case=case,context=aligned,query=query,geometry=geometry,
                    knots=parent['knots_rad'],braking=braking))
        for name,module in list(sys.modules.items()):
            if name.startswith('scripts.') and getattr(module,'__file__',None):
                path=str(Path(module.__file__).resolve());pins[path]=sha(path)
        pins[str(Path(__file__).resolve())]=sha(__file__)
        deadline=time.monotonic()+900.
        for job in jobs:job['deadline']=deadline
        write('status.json',dict(state='MODEL_SEARCH',candidates=len(jobs),maximum_seconds=900,native_execution=False))
        rows=[]
        with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
            futures={pool.submit(evaluate,job):i for i,job in enumerate(jobs)}
            for future in as_completed(futures):
                row=future.result(); name=f'candidate_{futures[future]:03d}.json';write(name,row)
                compact={k:v for k,v in row.items() if k not in ['result','knots_rad']}
                compact['file']=name;compact['sha256']=sha(a.output/name)
                compact['result']={k:v for k,v in row['result'].items() if k not in
                    ['trajectory','initial_q','initial_v','initial_contacts','initial_previous_command']}
                rows.append(compact);print(json.dumps(plain(dict(case=row['case'],braking=row['braking'],
                    completed=row['result']['completed_model_steps'],accepted=row['result']['accepted']))),flush=True)
        for path,h in pins.items():
            if sha(path)!=h:raise ValueError('Source changed '+path)
        write('result.json',dict(scope=__doc__,rows=rows,source_sha256=pins,native_execution=False,valid_expert_labels=False))
        write('status.json',dict(state='COMPLETED_MODEL_SEARCH',candidates=len(rows),
            verified=sum(r['independently_verified'] for r in rows),native_execution=False))
    except BaseException as e:
        write('status.json',dict(state='FAILED',error=repr(e)));raise


if __name__=='__main__':main()
