"""Try a load-balanced continuation seed after true ball-side recovery queries."""
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import scripts.landing_sequential_recovery_search_v1 as search
from scripts.parallel_landing_solver_v1 import build_controller
from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance
from scripts.terminal_recovery_constraints_v1 import make_proposal
from scripts.audit_terminal_gravity_balance_v1 import gravity_target
from scripts.audit_microstep_recovery_v1 import characterize
from scripts.audit_joint_sequence_proposal_v1 import plain


def query(args):
    bundle,geometry,record=args;started=time.perf_counter();row=record['continuation_check']
    m=json.loads((bundle/'manifest.json').read_text());cfg=m['configuration'];c=build_controller(cfg)
    c.sequence_geometry=AuthoredGroundClearance(geometry,c.qp.model,margin_m=0.)
    key=f"case_{row['case']:03d}";limits=np.asarray(cfg['interface']['limits'])
    with np.load(bundle/m['bundle_name'],allow_pickle=False) as z:
        initial=c.qp.model.state_from_public(z[key+'_query_public'],np.asarray(cfg['posture_target']))
        initial.previous_command_target=limits[:,0]+.5*(z[key+'_previous_action']+1.)*np.diff(limits).ravel()
        reference=limits[:,0]+.5*(z[key+'_query_student_action']+1.)*np.diff(limits).ravel()
    prefix,state=characterize(c,initial,row['commands'],reference)
    if not prefix['accepted']:raise ValueError('Original prefix failed independent recheck')
    old_propose=search.propose_sequence;old_rollout=search.rollout_for_search
    def weighted_rollout(*args):
        r=old_rollout(*args);H=len(r['profile'])
        r['cost']+=sum((10. if i==H-1 else 1./H)*np.sum(n['following'].joint_velocity**2)
            for i,n in enumerate(r['profile']));return r
    search.propose_sequence=make_proposal(2.);search.rollout_for_search=weighted_rollout
    blocks=[]
    try:
        for block in range(2):
            reference=state.previous_command_target.copy();proposal=gravity_target(c,state)
            if not proposal['solved']:
                blocks.append(dict(block=block,proposal=proposal,accepted=False));break
            seed=np.tile(proposal['target'],(40,1));before,following=characterize(c,state,seed,reference)
            task=None;repair=None;commands=seed
            if not before['accepted'] or not before['joint_settling_diagnostic']:
                task=search.sequential_recovery_search(c,state,reference,seed,iterations=12,time_limit_s=300.)
                repair=search.restore_feasibility(c,state,reference,task['final_search_commands'],iterations=12)
                commands=repair['commands'] if repair['accepted'] else task['commands']
                if commands is None:commands=repair['final_search_commands']
            after,following=characterize(c,state,commands,reference)
            accepted=bool(after['accepted'] and after['joint_settling_diagnostic'])
            blocks.append(dict(block=block,proposal=proposal,before=before,after=after,accepted=accepted,
                task=task,repair=repair,commands=commands if accepted else None))
            if not accepted:break
            state=copy.deepcopy(following)
    finally:search.propose_sequence=old_propose;search.rollout_for_search=old_rollout
    return plain(dict(case=row['case'],ball5_except_recovery=row['ball5_except_recovery'],
        native_execution=False,prefix=prefix,prefix_commands=row['commands'],blocks=blocks,
        verified_continuation_s=.4*sum(b['accepted'] for b in blocks),wall_seconds=time.perf_counter()-started))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['bundle','geometry','start-report','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve evidence')
    start=json.loads(a.start_report.read_text())
    for name,digest in start['source_sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:raise ValueError('Changed seed '+name)
    rows=[r for r in start['records'] if r['continuation_check']['after']['accepted']]
    a.output.parent.mkdir(parents=True,exist_ok=True);results=[]
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        for future in as_completed([pool.submit(query,(a.bundle,a.geometry,r)) for r in rows]):
            r=future.result();results.append(r);print(json.dumps(dict(case=r['case'],verified_continuation_s=r['verified_continuation_s'])),flush=True)
            a.output.with_suffix('.partial.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    files=[Path(__file__),a.start_report,a.bundle/'manifest.json',a.geometry]
    files += [Path(m.__file__) for n,m in sys.modules.items() if n.startswith('scripts.') and getattr(m,'__file__',None)]
    result=dict(scope=__doc__,native_execution=False,valid_expert_labels=False,records=sorted(results,key=lambda r:r['case']),
        source_sha256={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
