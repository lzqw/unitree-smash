"""Optimize the support transition after a constrained landing prefix.

All rollouts start at the original measured airborne state, run to tick 800,
and keep hard constraints unchanged. The first 0.35 s is preserved. Search
is offline full-remainder shooting, not yet a native MPC or expert teacher.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np
from scipy.interpolate import PchipInterpolator

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.check_constrained_recovery_suffix_v1 import source_context, state_from_row
from scripts.search_preserved_early_support_v1 import initialize
from scripts.search_landing_support_tail_v3 import _WORKER
from scripts.recovery_family_queries_v1 import transfer_reference
from scripts.terminal_attitude_feedback_v1 import TerminalAttitudeFeedbackV1
from scripts.landing_joint_braking_reference_v1 import LandingJointBrakingReferenceV1
from scripts.hip_mount_path_constraint_v1 import HipMountPathConstraintV1
from scripts.speed_limited_recovery_suffix_v1 import evaluate_suffix
from scripts.audit_joint_sequence_proposal_v1 import plain


class SupportTransitionV1:
    def __init__(self,base,knots):
        self.base=base;self.model=base.model;self.interface=base.interface
        knots=np.asarray(knots,float)
        if knots.shape!=(3,18) or not np.isfinite(knots).all():raise ValueError('Three finite transition knots required')
        self.curve=PchipInterpolator([.35,.60,1.20,2.50],np.vstack([np.zeros(18),knots]),axis=0)

    def action(self,offset,state,previous):
        target,mode=self.base.action(offset,state,previous);t=.01*(offset-80)
        if t<=.35:return target,mode
        result=np.clip(target+self.curve(min(t,2.5)),self.interface.limits[:,0],self.interface.limits[:,1])
        result[12:]=np.clip(result[12:],previous[12:]-.01*self.interface.arm_rate,previous[12:]+.01*self.interface.arm_rate)
        return result,'optimized_early_support_transition'


def evaluate(job):
    w=_WORKER;c=w['controller'];state=state_from_row(job['query'])
    base=w['factory'](c.qp.model,c.qp.interface,state,np.asarray(job['original_knots']))
    tracking=TerminalAttitudeFeedbackV1(base,proportional=.3,damping_s=.04,maximum_joint_delta_rad=.08)
    transition=SupportTransitionV1(tracking,job['transition_knots'])
    policy=LandingJointBrakingReferenceV1(transition,**job['braking'])
    result=evaluate_suffix(c,state,policy,w['spheres'],
        HipMountPathConstraintV1(w['geometry'],job['geometry'],margin_rad=.02),w['protocol'],
        control_tick=280,substeps=job['substeps'],deadline=job['deadline'],save=job['substeps']==160)
    if job['substeps']==160:
        prefix=result['trajectory'][:35]
        if len(prefix)!=35 or any(not np.array_equal(a[k],b[k]) for a,b in zip(prefix,job['prefix']) for k in ['q','v','target','contacts']):
            raise ValueError('Candidate altered preserved landing prefix')
    return dict(transition_knots=job['transition_knots'],result=result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['plan','candidate','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Fresh output required')
    a.output.mkdir(parents=True)
    read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def write(name,value):
        path=a.output/name;tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(plain(value),indent=2,allow_nan=False)+'\n');tmp.replace(path)
    try:
        parent=read(a.candidate);r=parent['result']
        if parent['case']!=87 or r['substeps']!=160 or r['completed_model_steps']<60 or r['accepted']:
            raise ValueError('Measured-state candidate with a useful but incomplete landing prefix required')
        plan=read(a.plan);source=read(plan['source']);_,_,_,geometry,context=source_context(source['configuration']['source'])
        initialize(*context);query=dict(q=r['initial_q'],v=r['initial_v'],contacts=r['initial_contacts'],target=r['initial_previous_command'])
        context,_=transfer_reference(context,state_from_row(query))
        pins=dict(plan['source_sha256']);pins[str(a.candidate.resolve())]=sha(a.candidate)
        for name,module in list(sys.modules.items()):
            if name.startswith('scripts.') and getattr(module,'__file__',None):
                path=str(Path(module.__file__).resolve());pins[path]=sha(path)
        pins[str(Path(__file__).resolve())]=sha(__file__)
        for path,h in pins.items():
            if sha(path)!=h:raise ValueError('Changed source '+path)
        began=time.monotonic();deadline=began+900.;search_end=began+720.
        template=dict(query=query,original_knots=parent['knots_rad'],braking=parent['braking'],
            geometry=geometry,prefix=r['trajectory'][:35])
        rng=np.random.default_rng(882140);mean=np.zeros((3,18))
        std=np.tile(np.r_[np.full(12,.12),np.full(6,.04)],(3,1))
        bounds=np.r_[np.full(12,.6),np.full(6,.3)]
        floor=np.r_[np.full(12,.025),np.full(6,.012)]
        records=[];best=None;verified=None
        with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=context) as pool:
            def batch(knots,substeps,limit):
                jobs={pool.submit(evaluate,dict(template,transition_knots=k,substeps=substeps,deadline=limit)):i for i,k in enumerate(knots)}
                rows=[None]*len(knots)
                for future in as_completed(jobs):rows[jobs[future]]=future.result()
                return rows
            for generation in range(6):
                if time.monotonic()>=search_end:break
                write('status.json',dict(state='MODEL_SEARCH',generation=generation,case=87,maximum_seconds=900,native_execution=False))
                population=np.clip(mean+std*(.65*rng.normal(size=(24,1,18))+.65*rng.normal(size=(24,3,18))),-bounds,bounds)
                population[0]=mean;population[1]=0.
                if best is not None:population[2]=best['transition_knots']
                coarse=batch(population,32,search_end)
                ranked=sorted(coarse,key=lambda x:x['result']['rank'])
                records.append(dict(generation=generation,coarse=coarse))
                elite=np.stack([row['transition_knots'] for row in ranked[:6]])
                mean=.3*mean+.7*elite.mean(0);std=np.maximum(floor,.3*std+.7*elite.std(0))
                fine=batch([row['transition_knots'] for row in ranked[:4]],160,deadline)
                for row in fine:
                    if best is None or row['result']['rank']<best['result']['rank']:best=row
                write(f'generation_{generation:02d}.json',dict(coarse=coarse,fine=fine))
                print(json.dumps(plain(dict(generation=generation,best_completed=best['result']['completed_model_steps'],
                    best_failure=best['result']['failure']))),flush=True)
                accepted=[row for row in fine if row['result']['accepted']]
                if accepted:
                    winner=min(accepted,key=lambda x:x['result']['rank']);replay=batch([winner['transition_knots']],160,deadline)[0]
                    x=winner['result'];y=replay['result']
                    if y['accepted'] and len(x['trajectory'])==len(y['trajectory'])==520 and all(
                        np.array_equal(a[k],b[k]) for a,b in zip(x['trajectory'],y['trajectory']) for k in ['q','v','target','contacts']):
                        verified=dict(winner=winner,replay=replay);break
        for path,h in pins.items():
            if sha(path)!=h:raise ValueError('Source changed during search '+path)
        write('result.json',dict(scope=__doc__,case=87,query=query,original_knots=parent['knots_rad'],braking=parent['braking'],
            best=best,verified=verified,source_sha256=pins,preserved_prefix_steps=35,wall_seconds=time.monotonic()-began,
            native_execution=False,valid_expert_labels=False))
        write('status.json',dict(state='COMPLETED_MODEL_SEARCH',verified=verified is not None,native_execution=False))
    except BaseException as error:
        write('status.json',dict(state='FAILED',error=repr(error)));raise


if __name__=='__main__':main()
