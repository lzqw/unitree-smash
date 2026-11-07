"""Repair substep clearance and test continued model support, never native proof."""
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
from scripts.landing_joint_sequence_proposal_v1 import verify_sequence
from scripts.microstep_ground_sequence_v1 import make_proposal,generated_source
from scripts.parallel_landing_solver_v1 import build_controller
from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance
from scripts.audit_joint_sequence_proposal_v1 import plain


def characterize(controller,state,commands,reference):
    rollout=search.rollout_for_search(controller,state,commands,reference)
    verification=verify_sequence(controller,state,commands)
    last=rollout['profile'][-1]['following']
    leg=float(np.max(abs(last.joint_velocity[:12])));arm=float(np.max(abs(last.joint_velocity[12:])))
    vertical=float((last.root_rotation_world@last.v[:3])[2])
    accepted=bool(verification['accepted'] and rollout['physically_feasible'])
    return dict(accepted=accepted,maximum_physical_violation=rollout['maximum_physical_violation'],
        ground_violation_m=max(x['physical']['authored_ground_clearance_m'] for x in rollout['checks']),
        terminal_risk=rollout['terminal_risk'],maximum_leg_speed_rad_s=leg,
        maximum_arm_speed_rad_s=arm,vertical_speed_mps=vertical,
        # Additional diagnostic, never a change to formal recovery criteria.
        joint_settling_diagnostic=bool(max(leg,arm)<=.5 and abs(vertical)<=.1),
        verification=verification),last


def query(args):
    bundle,geometry,row,blocks=args;started=time.perf_counter()
    m=json.loads((bundle/'manifest.json').read_text());cfg=m['configuration'];c=build_controller(cfg)
    c.sequence_geometry=AuthoredGroundClearance(geometry,c.qp.model,margin_m=0.)
    key=f"case_{row['case']:03d}";limits=np.asarray(cfg['interface']['limits'])
    with np.load(bundle/m['bundle_name'],allow_pickle=False) as z:
        state=c.qp.model.state_from_public(z[key+'_query_public'],np.asarray(cfg['posture_target']))
        state.previous_command_target=limits[:,0]+.5*(z[key+'_previous_action']+1.)*np.diff(limits).ravel()
        reference=limits[:,0]+.5*(z[key+'_query_student_action']+1.)*np.diff(limits).ravel()
    old_propose=search.propose_sequence;old_rollout=search.rollout_for_search
    def weighted_rollout(*args):
        result=old_rollout(*args);H=len(result['profile'])
        result['cost']+=sum((10. if i==H-1 else 1./H)*np.sum(n['following'].joint_velocity**2)
            for i,n in enumerate(result['profile']))
        return result
    search.propose_sequence=make_proposal(2.);search.rollout_for_search=weighted_rollout
    try:
        before,_=characterize(c,state,row['final_search_commands'],reference)
        repair=search.restore_feasibility(c,state,reference,row['final_search_commands'],iterations=12)
        commands=repair['commands'] if repair['accepted'] else repair['final_search_commands']
        after,last=characterize(c,state,commands,reference);tails=[]
        if after['accepted'] and after['joint_settling_diagnostic']:
            for block in range(blocks):
                initial=copy.deepcopy(last);ref=initial.previous_command_target.copy()
                nominal=np.tile(ref,(c.settings.horizon_steps,1))
                held,held_last=characterize(c,initial,nominal,ref)
                task=None;tail_repair=None
                if held['accepted'] and held['joint_settling_diagnostic']:
                    tail_commands=nominal;tail_after=held;last=held_last
                else:
                    task=search.sequential_recovery_search(c,initial,ref,nominal,iterations=8,time_limit_s=240.)
                    tail_repair=search.restore_feasibility(c,initial,ref,task['final_search_commands'],iterations=12)
                    tail_commands=tail_repair['commands'] if tail_repair['accepted'] else task['commands']
                    if tail_commands is None:tail_commands=tail_repair['final_search_commands']
                    tail_after,last=characterize(c,initial,tail_commands,ref)
                tails.append(dict(block=block,held=held,task=task,repair=tail_repair,
                    after=tail_after,commands=tail_commands))
                if not tail_after['accepted'] or not tail_after['joint_settling_diagnostic']:break
        return plain(dict(case=row['case'],ball5_except_recovery=row['ball5_except_recovery'],
            native_execution=False,before=before,repair=repair,after=after,
            commands=commands if after['accepted'] else None,final_search_commands=commands,
            continuation=tails,verified_continuation_s=.4*sum(t['after']['accepted'] and
                t['after']['joint_settling_diagnostic'] for t in tails),wall_seconds=time.perf_counter()-started))
    finally:search.propose_sequence=old_propose;search.rollout_for_search=old_rollout


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['bundle','geometry','start-report','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--continuation-blocks',type=int,choices=[0,1,2],default=1);a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve previous evidence')
    start=json.loads(a.start_report.read_text())
    for name,digest in start['source_sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:raise ValueError('Changed seed source '+name)
    a.output.parent.mkdir(parents=True,exist_ok=True);generated=a.output.with_suffix('.proposal.py')
    generated.write_text(generated_source(2.));rows=[];started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs=[pool.submit(query,(a.bundle,a.geometry,r,a.continuation_blocks)) for r in start['records']]
        for future in as_completed(jobs):
            row=future.result();rows.append(row)
            print(json.dumps(dict(case=row['case'],accepted=row['after']['accepted'],
                ground_violation_m=row['after']['ground_violation_m'],
                joint_settling=row['after']['joint_settling_diagnostic'],
                verified_continuation_s=row['verified_continuation_s'])),flush=True)
            a.output.with_suffix('.partial.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    files=[Path(__file__),a.start_report,a.bundle/'manifest.json',a.geometry,generated]
    files += [Path(m.__file__) for n,m in sys.modules.items() if n.startswith('scripts.') and getattr(m,'__file__',None)]
    result=dict(scope=__doc__,native_execution=False,valid_expert_labels=False,
        formal_criteria_unchanged=True,geometry_tolerance_m=1e-6,
        linear_geometry_row_tolerance_m=1e-7,geometry_slack_upper_bound_m=0.,
        continuation_is_predicted_not_measured=True,
        joint_settling_diagnostic_thresholds=dict(joint_speed_rad_s=.5,vertical_speed_mps=.1),
        records=sorted(rows,key=lambda r:r['case']),wall_seconds=time.perf_counter()-started,
        source_sha256={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
