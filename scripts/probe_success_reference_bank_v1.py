"""Test distinct successful command proposals on preregistered failure states.

Uses passive native inputs and full-remainder model execution. No simulated
state reset, native control change, expert label, or real-time claim follows.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.native_recovery_query_bank_v1 import state_from_sample
from scripts.check_constrained_recovery_suffix_v1 import source_context, state_from_row
from scripts.search_preserved_early_support_v1 import initialize
from scripts.search_landing_support_tail_v3 import _WORKER
from scripts.recovery_family_queries_v1 import transfer_reference
from scripts.recorded_recovery_reference_v1 import RecordedRecoveryReferenceV1, proposal_targets, initial_distance
from scripts.landing_reference_tracking_v1 import LandingReferenceTrackingV1
from scripts.terminal_attitude_feedback_v1 import TerminalAttitudeFeedbackV1
from scripts.hip_mount_path_constraint_v1 import HipMountPathConstraintV1
from scripts.constrained_recovery_suffix_v1 import evaluate_suffix
from scripts.audit_joint_sequence_proposal_v1 import plain
from scripts.queue_native_terminal_support_v1 import identity


def evaluate(job):
    initialize(*job['context']); w = _WORKER; controller = w['controller']
    state = state_from_row(job['query'])
    def rollout():
        base = RecordedRecoveryReferenceV1(controller.qp.model, controller.qp.interface,
            w['initial'], job['targets'])
        tracking = LandingReferenceTrackingV1(base, job['context'][4], job['context'][5], **job['context'][6])
        policy = TerminalAttitudeFeedbackV1(tracking, proportional=.3, damping_s=.04,
            maximum_joint_delta_rad=.08)
        geometry = HipMountPathConstraintV1(w['geometry'], job['geometry'], margin_rad=.02)
        return evaluate_suffix(controller, state, policy, w['spheres'], geometry, w['protocol'],
            control_tick=280, substeps=160, deadline=job['deadline'], save=True)
    result = rollout(); replay_exact = False
    if result['accepted']:
        replay = rollout()
        replay_exact = (replay['accepted'] and len(replay['trajectory']) == len(result['trajectory']) == 520
            and all(np.array_equal(a[k], b[k]) for a,b in zip(result['trajectory'], replay['trajectory'])
                    for k in ['q','v','target','contacts']))
    return dict(case=job['case'], donor=job['donor'], similarity=job['similarity'],
        result=result, independently_verified_model_plan=bool(result['accepted'] and replay_exact),
        replay_exact=replay_exact, native_execution=False, valid_expert_labels=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    args=p.parse_args();plan=json.loads(args.plan.read_text());out=args.plan.parent
    read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def write(name,value):
        path=out/name;tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(plain(value),indent=2,allow_nan=False)+'\n');tmp.replace(path)
    def check(pins):
        for name,h in pins.items():
            if sha(name)!=h:raise ValueError('Changed input '+name)
    with (out/'owner.json').open('x') as f:json.dump(dict(pid=os.getpid(),plan_sha256=sha(args.plan)),f)
    try:
        check(plan['source_sha256']);owner=plan['wait_process'];began=time.monotonic()
        write('status.json',dict(state='WAITING_FOR_CPU_OWNER',pid=os.getpid(),owner=owner))
        while True:
            current=identity(owner['pid'])
            if current is None:break
            if current!=owner['start_ticks']:raise ValueError('CPU owner identity changed')
            if time.monotonic()-began>7200:raise TimeoutError('Bounded CPU wait expired')
            time.sleep(5)
        check(plan['source_sha256'])
        bank=Path(plan['bank_run']);follow=Path(plan['bank_followup'])
        if read(follow/'status.json')['state']!='COMPLETED_DIAGNOSTIC':raise ValueError('Bank strict audit did not pass')
        audit=read(follow/'query_bank_audit.json');meta=read(bank/'recovery_query_bank.json')
        if not audit['passed'] or not meta['complete_window']:raise ValueError('Incomplete or nonpassive bank')
        check(audit['source_sha256'])
        if sha(bank/'recovery_query_bank.npz')!=meta['data_sha256']:raise ValueError('Changed state bank')
        comparison=read(follow/'comparison.json')
        donors=[r['training_index'] for r in comparison['after']['stages'] if r['complete5']]
        if len(donors)!=27:raise ValueError('Expected original 27 successful donors')
        selection=read(plan['selection']);source=read(plan['source'])
        _,_,_,geometry,context=source_context(source['configuration']['source'])
        initialize(*context);model=_WORKER['controller'].qp.model
        if list(model.joint_order)!=meta['joint_names']:raise ValueError('Joint order mismatch')
        pins=dict(plan['source_sha256']);pins.update(audit['source_sha256'])
        pins[str(follow/'comparison.json')]=sha(follow/'comparison.json')
        for name,module in list(sys.modules.items()):
            if name.startswith('scripts.') and getattr(module,'__file__',None):
                path=str(Path(module.__file__).resolve());pins[path]=sha(path)
        jobs=[];selection_rows=[]
        with np.load(bank/'recovery_query_bank.npz',allow_pickle=False) as archive:
            # NPZ indexing decompresses a whole member. Materialize this
            # bounded bank once, not once for every donor/frame/field.
            z={k:archive[k] for k in archive.files}
            if sum(x.nbytes for x in z.values())>64*1024**2:raise ValueError('State-bank memory bound exceeded')
            indices=np.flatnonzero((z['control_ticks']>=280)&(z['control_ticks']<800))
            if not np.array_equal(z['control_ticks'][indices],np.arange(280,800)):raise ValueError('Incomplete suffix clock')
            def state_at(lane,index):
                sample={k:z[k][index,lane].copy() for k in z if k!='control_ticks'}
                return state_from_sample(model,sample,meta['origins_world'][lane])
            donor_states={lane:state_at(lane,indices[0]) for lane in donors}
            for choice in selection['selected']:
                lane=choice['case'];i=indices[0]
                if not z['unfinished_first_episode'][i,lane] or z['episode_index'][i,lane]!=0 or z['phase'][i,lane,3]<=.5:
                    raise ValueError('Failure input is not original post-hit flight')
                state=state_at(lane,i);query=dict(q=state.q.copy(),v=state.v.copy(),contacts=state.contacts.copy(),target=state.previous_command_target.copy())
                ranked=sorted([(initial_distance(state,s),d) for d,s in donor_states.items()],key=lambda r:(r[0]['score'],r[1]))
                selection_rows.append(dict(case=lane,ranked_donors=[dict(donor=d,similarity=s) for s,d in ranked],selected=[d for _,d in ranked[:3]]))
                for similarity,donor in ranked[:3]:
                    if not z['unfinished_first_episode'][indices,donor].all() or np.any(z['episode_index'][indices,donor]!=0):
                        raise ValueError('Donor suffix contains reset states')
                    states=[state_at(donor,j) for j in indices]
                    donor_context=list(copy.deepcopy(context));donor_context[1]['initial_q']=states[0].q.copy()
                    donor_context[4]=np.stack([s.q for s in states]);donor_context[5]=np.stack([s.v for s in states])
                    aligned,_=transfer_reference(tuple(donor_context),state)
                    jobs.append(dict(case=lane,donor=donor,similarity=similarity,context=aligned,query=query,
                        targets=proposal_targets(z['previous_target'][indices,donor]),geometry=geometry))
        write('donor_selection.json',dict(rows=selection_rows,uses_candidate_outcomes=False,
            last_command='Tick799 proposal holds measured tick798 command; not a measured tick799 action.',source_sha256=pins))
        deadline=time.monotonic()+1800
        for job in jobs:job['deadline']=deadline
        write('status.json',dict(state='CPU_FULL_REMAINDER_PROBES',pid=os.getpid(),candidates=len(jobs),workers=4))
        rows=[]
        with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
            futures=[pool.submit(evaluate,job) for job in jobs]
            for future in as_completed(futures):
                row=future.result();name=f"case_{row['case']:03d}_donor_{row['donor']:03d}.json";write(name,row)
                rows.append(dict(case=row['case'],donor=row['donor'],verified=row['independently_verified_model_plan'],
                    completed=row['result']['completed_model_steps'],failure=row['result']['failure'],file=name,sha256=sha(out/name)))
                print(json.dumps(plain(rows[-1])),flush=True)
        check(pins);write('result.json',dict(rows=rows,source_sha256=pins,native_execution=False,valid_expert_labels=False))
        write('status.json',dict(state='COMPLETED_MODEL_PROBES',candidates=len(rows),verified=sum(r['verified'] for r in rows),native_execution=False))
    except BaseException as error:
        write('status.json',dict(state='FAILED',error=repr(error),pid=os.getpid()));raise


if __name__=='__main__':main()
