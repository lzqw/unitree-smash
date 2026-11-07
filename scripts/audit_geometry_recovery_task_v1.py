"""Joint task refinement after geometry restoration, without native promotion."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance
from scripts.parallel_landing_solver_v1 import build_controller
from scripts.landing_sequential_recovery_search_v1 import rollout_for_search,restore_feasibility,sequential_recovery_search
from scripts.landing_joint_sequence_proposal_v1 import verify_sequence
from scripts.audit_joint_sequence_proposal_v1 import plain


def query(args):
    bundle,geometry,row=args;start=time.perf_counter()
    manifest=json.loads((bundle/'manifest.json').read_text());cfg=manifest['configuration']
    c=build_controller(cfg);c.sequence_geometry=AuthoredGroundClearance(geometry,c.qp.model,margin_m=0.)
    key=f"case_{row['case']:03d}"
    with np.load(bundle/manifest['bundle_name'],allow_pickle=False) as z:
        state=c.qp.model.state_from_public(z[key+'_query_public'],np.asarray(cfg['posture_target']))
        limits=np.asarray(cfg['interface']['limits'])
        state.previous_command_target=limits[:,0]+.5*(z[key+'_previous_action']+1.)*np.diff(limits).ravel()
        reference=limits[:,0]+.5*(z[key+'_query_student_action']+1.)*np.diff(limits).ravel()
    task=sequential_recovery_search(c,state,reference,row['final_search_commands'],iterations=12,time_limit_s=300.)
    repair=restore_feasibility(c,state,reference,task['final_search_commands'],iterations=8)
    commands=repair['commands'] if repair['accepted'] else task['commands']
    accepted=commands is not None
    if accepted:
        checked=verify_sequence(c,state,commands)
        search=rollout_for_search(c,state,commands,reference)
        if not checked['accepted'] or not search['physically_feasible']:raise ValueError('Final independent rejection')
    else:
        commands=repair['final_search_commands'];checked=verify_sequence(c,state,commands)
        search=rollout_for_search(c,state,commands,reference)
    return plain(dict(case=row['case'],accepted=accepted,native_execution=False,
        newly_model_feasible=row['newly_model_feasible'],ball5_except_recovery=row['ball5_except_recovery'],
        commands=commands if accepted else None,final_search_commands=commands,
        task=task,repair=repair,verification=checked,
        final_ground_violation_m=max(r['physical']['authored_ground_clearance_m'] for r in search['checks']),
        maximum_physical_violation=search['maximum_physical_violation'],
        final_terminal_risk=search['terminal_risk'],wall_seconds=time.perf_counter()-start))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--geometry',type=Path,required=True)
    p.add_argument('--restoration-report',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve evidence')
    previous=json.loads(a.restoration_report.read_text())
    for name,digest in previous['source_sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:raise ValueError('Changed restoration source '+name)
    started=time.perf_counter();rows=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        for row in pool.map(query,[(a.bundle,a.geometry,r) for r in previous['records']]):
            rows.append(row);print(json.dumps({k:row[k] for k in ['case','accepted','final_ground_violation_m','maximum_physical_violation','wall_seconds']}),flush=True)
            a.output.with_suffix('.partial.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    files=[Path(__file__),a.restoration_report,a.bundle/'manifest.json',a.geometry]
    files += [Path(m.__file__) for n,m in sys.modules.items() if n.startswith('scripts.') and getattr(m,'__file__',None)]
    summary=dict(queries=len(rows),accepted_model_candidates=sum(r['accepted'] for r in rows),
        accepted_new_candidates=sum(r['accepted'] and r['newly_model_feasible'] for r in rows),
        wall_seconds=time.perf_counter()-started)
    result=dict(scope=__doc__,native_execution=False,summary=summary,records=rows,
        source_sha256={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(summary))

if __name__=='__main__':main()
