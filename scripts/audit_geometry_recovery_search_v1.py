"""Attempt geometry-constrained recovery of recorded whole-sequence candidates."""
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
from scripts.landing_sequential_recovery_search_v1 import rollout_for_search,restore_feasibility
from scripts.landing_joint_sequence_proposal_v1 import verify_sequence
from scripts.audit_joint_sequence_proposal_v1 import plain


def query(args):
    bundle,geometry,item,iterations=args;start=time.perf_counter()
    manifest=json.loads((bundle/'manifest.json').read_text());cfg=manifest['configuration']
    c=build_controller(cfg);g=AuthoredGroundClearance(geometry,c.qp.model,margin_m=0.)
    c.sequence_geometry=g;key=f"case_{item['case']:03d}"
    with np.load(bundle/manifest['bundle_name'],allow_pickle=False) as z:
        state=c.qp.model.state_from_public(z[key+'_query_public'],np.asarray(cfg['posture_target']))
        limits=np.asarray(cfg['interface']['limits'])
        state.previous_command_target=limits[:,0]+.5*(z[key+'_previous_action']+1.)*np.diff(limits).ravel()
        reference=limits[:,0]+.5*(z[key+'_query_student_action']+1.)*np.diff(limits).ravel()
        commands=z[key+'_commands_rad'].copy()
        if np.max(abs(state.q-z[key+'_predicted_q'][0]))>1e-9:raise ValueError('Query state mismatch')
    before=rollout_for_search(c,state,commands,reference)
    initial_min=float(min(g.evaluate(state.q)[0]))
    result=restore_feasibility(c,state,reference,commands,iterations=iterations)
    after=rollout_for_search(c,state,result['final_search_commands'],reference)
    checked=verify_sequence(c,state,result['final_search_commands'])
    return plain(dict(case=item['case'],query_tick=item['query_tick'],
        ball5_except_recovery=item['ball5_except_recovery'],newly_model_feasible=item['newly_model_feasible'],
        initial_query_minimum_clearance_m=initial_min,
        before_ground_violation_m=max(r['physical']['authored_ground_clearance_m'] for r in before['checks']),
        after_ground_violation_m=max(r['physical']['authored_ground_clearance_m'] for r in after['checks']),
        before_infeasibility=before['infeasibility'],final_checks=after['checks'],
        independent_verification=checked,wall_seconds=time.perf_counter()-start,**result))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--geometry',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--iterations',type=int,default=8);a=p.parse_args()
    if a.output.exists() or not 1<=a.workers<=4:raise ValueError('Preserve evidence and limit CPU workers')
    manifest=json.loads((a.bundle/'manifest.json').read_text());archive=a.bundle/manifest['bundle_name']
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=manifest['bundle_sha256']:raise ValueError('Bundle changed')
    # These are archived command proposals, not previously certified geometry
    # or current native controls. Reconstruct them from actual query observations.
    started=time.perf_counter();rows=[]
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for row in pool.map(query,[(a.bundle,a.geometry,r,a.iterations) for r in manifest['rows']]):
            rows.append(row);print(json.dumps({k:row[k] for k in ['case','accepted','before_ground_violation_m','after_ground_violation_m','wall_seconds']}),flush=True)
            a.output.with_suffix('.partial.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    files=[Path(__file__),a.bundle/'manifest.json',archive,a.geometry]
    files += [Path(m.__file__) for n,m in sys.modules.items() if n.startswith('scripts.') and getattr(m,'__file__',None)]
    summary=dict(queries=len(rows),accepted_model_candidates=sum(r['accepted'] for r in rows),
        accepted_new_candidates=sum(r['accepted'] and r['newly_model_feasible'] for r in rows),
        wall_seconds=time.perf_counter()-started)
    result=dict(scope=__doc__,native_execution=False,summary=summary,records=rows,
        geometry_margin_m=0.,geometry_tolerance_m=1e-6,
        limitations=['Authored ground geometry only; self-collision and cooked geometry not certified.',
            'Optimization at control boundaries, final checking at all 16 model microsteps.',
            'Search-only scalar geometry slack; no slack allowed for final acceptance.',
            'Known frozen-contact dynamics and native mismatch limitations remain.'],
        source_sha256={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(summary))

if __name__=='__main__':main()
