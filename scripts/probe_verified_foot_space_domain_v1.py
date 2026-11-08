"""Evaluate one verified policy on recorded recovery failures, without retuning.

Initial states and command histories come from a strictly passive native bank.
Only reference XY/yaw is aligned. This is model coverage, never native rescue.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
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
from scripts.recovery_family_queries_v1 import transfer_reference
from scripts.native_recovery_query_bank_v1 import state_from_sample
from scripts.terminal_attitude_feedback_v1 import TerminalAttitudeFeedbackV1
from scripts.search_braked_support_transition_v1 import SupportTransitionV1
from scripts.landing_joint_braking_reference_v1 import LandingJointBrakingReferenceV1
from scripts.hip_mount_path_constraint_v1 import HipMountPathConstraintV1
from scripts.speed_limited_recovery_suffix_v1 import evaluate_suffix
from scripts.audit_joint_sequence_proposal_v1 import plain


def evaluate(job):
    initialize(*job['context']);w=_WORKER;c=w['controller'];state=state_from_row(job['query'])
    def rollout():
        base=w['factory'](c.qp.model,c.qp.interface,state,np.asarray(job['knots']))
        tracking=TerminalAttitudeFeedbackV1(base,proportional=.3,damping_s=.04,maximum_joint_delta_rad=.08)
        policy=LandingJointBrakingReferenceV1(SupportTransitionV1(tracking,job['transition']),**job['braking'])
        return evaluate_suffix(c,state,policy,w['spheres'],HipMountPathConstraintV1(w['geometry'],job['geometry'],margin_rad=.02),
            w['protocol'],control_tick=280,substeps=160,deadline=job['deadline'],save=True)
    r=rollout();verified=False
    if r['accepted']:
        s=rollout();verified=s['accepted'] and len(s['trajectory'])==len(r['trajectory'])==520 and all(
            np.array_equal(a[k],b[k]) for a,b in zip(r['trajectory'],s['trajectory']) for k in ['q','v','contacts','target'])
    if job['case']==87:
        if not r['accepted'] or not verified or any(not np.array_equal(a[k],b[k])
                for a,b in zip(r['trajectory'],job['nominal']) for k in ['q','v','contacts','target']):
            raise ValueError('Nominal policy/state replay changed')
    return dict(case=job['case'],query=job['query'],result=r,verified=bool(verified),
        native_execution=False,valid_expert_labels=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['plan','search','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Fresh output required')
    a.output.mkdir(parents=True)
    read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def write(name,value):
        path=a.output/name;tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(plain(value),indent=2,allow_nan=False)+'\n');tmp.replace(path)
    try:
        plan=read(a.plan);search=read(a.search)
        if not search['verified']:raise ValueError('Verified nominal model solution required')
        audit=read(Path(plan['bank_followup'])/'query_bank_audit.json');meta=read(Path(plan['bank_run'])/'recovery_query_bank.json')
        if not audit['passed'] or not meta['complete_window']:raise ValueError('Verified passive complete state bank required')
        pins=dict(search['source_sha256']);pins.update(audit['source_sha256']);pins[str(a.search.resolve())]=sha(a.search)
        for path,h in pins.items():
            if sha(path)!=h:raise ValueError('Changed source '+path)
        source=read(plan['source']);_,_,_,geometry,context=source_context(source['configuration']['source'])
        initialize(*context);nominal_state=state_from_row(search['query'])
        nominal_context,_=transfer_reference(context,nominal_state);model=_WORKER['controller'].qp.model
        with np.load(Path(plan['bank_run'])/'recovery_query_bank.npz',allow_pickle=False) as archive:
            z={k:archive[k] for k in archive.files}
        index=int(np.flatnonzero(z['control_ticks']==280)[0]);jobs=[];skipped=[]
        for case in read(plan['selection'])['cohort']:
            sample={k:z[k][index,case].copy() for k in z if k!='control_ticks'}
            if not sample['unfinished_first_episode'] or sample['episode_index']!=0 or sample['phase'][3]<=.5:
                skipped.append(dict(case=case,reason='not_observed_first_episode_post_hit_flight_at_tick280'));continue
            measured=state_from_sample(model,sample,meta['origins_world'][case])
            query=dict(q=measured.q,v=measured.v,contacts=measured.contacts,target=measured.previous_command_target)
            if case==87:
                for k in query:
                    if not np.array_equal(query[k],search['query'][k]):raise ValueError('Nominal bank query differs')
                ctx=nominal_context
            else:ctx,_=transfer_reference(nominal_context,measured)
            jobs.append(dict(case=case,query=query,context=ctx,geometry=geometry,
                knots=search['original_knots'],transition=search['verified']['replay']['transition_knots'],
                braking=search['braking'],nominal=search['verified']['replay']['result']['trajectory'] if case==87 else None))
        if not any(j['case']==87 for j in jobs):raise ValueError('Missing nominal regression')
        for name,module in list(sys.modules.items()):
            if name.startswith('scripts.') and getattr(module,'__file__',None):
                path=str(Path(module.__file__).resolve());pins[path]=sha(path)
        pins[str(Path(__file__).resolve())]=sha(__file__)
        deadline=time.monotonic()+1800.
        for j in jobs:j['deadline']=deadline
        write('status.json',dict(state='MODEL_DOMAIN_PROBES',cases=len(jobs),workers=2,maximum_seconds=1800,native_execution=False))
        rows=[]
        with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn')) as pool:
            for future in as_completed([pool.submit(evaluate,j) for j in jobs]):
                row=future.result();name=f"case_{row['case']:03d}.json";write(name,row)
                compact=dict(case=row['case'],verified=row['verified'],file=name,sha256=sha(a.output/name),
                    result={k:v for k,v in row['result'].items() if k not in ['trajectory','initial_q','initial_v','initial_contacts','initial_previous_command']})
                rows.append(compact);print(json.dumps(dict(case=row['case'],verified=row['verified'],completed=row['result']['completed_model_steps'])),flush=True)
        for path,h in pins.items():
            if sha(path)!=h:raise ValueError('Source changed '+path)
        write('result.json',dict(scope=__doc__,rows=rows,skipped=skipped,source_sha256=pins,native_execution=False,valid_expert_labels=False))
        write('status.json',dict(state='COMPLETED_MODEL_DOMAIN_PROBES',verified=sum(r['verified'] for r in rows),cases=len(rows),skipped=len(skipped),native_execution=False))
    except BaseException as e:
        write('status.json',dict(state='FAILED',error=repr(e)));raise


if __name__=='__main__':main()
