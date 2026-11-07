"""Probe continued model support beyond an accepted 0.4 s recovery endpoint."""
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time
import numpy as np
import pinocchio as pin
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance
from scripts.parallel_landing_solver_v1 import build_controller
from scripts.landing_joint_sequence_proposal_v1 import verify_sequence
from scripts.landing_sequential_recovery_search_v1 import rollout_for_search,sequential_recovery_search,restore_feasibility
from scripts.audit_joint_sequence_proposal_v1 import plain


def query(args):
    bundle,geometry,kind=args;started=time.perf_counter()
    m=json.loads((bundle/'manifest.json').read_text());cfg=m['configuration'];c=build_controller(cfg)
    c.sequence_geometry=AuthoredGroundClearance(geometry,c.qp.model,margin_m=0.)
    with np.load(bundle/m['bundle_name'],allow_pickle=False) as z:
        state=c.qp.model.state_from_public(z['case_015_query_public'],np.asarray(cfg['posture_target']))
        state.q=z['case_015_predicted_q'][-1].copy();state.v=z['case_015_predicted_v'][-1].copy()
        state.contacts=z['case_015_predicted_contacts'][-1].copy()
        state.previous_command_target=z['case_015_commands_rad'][-1].copy()
    model=c.qp.model;state.root_rotation_world=pin.Quaternion(state.q[3:7]).matrix()
    state.root_position_world=state.q[:3].copy();state.joint_position=state.q[model.q_indices].copy()
    state.joint_velocity=state.v[model.v_indices].copy()
    initial=copy.deepcopy(state);reference=state.previous_command_target.copy();blocks=[]
    for block in range(2):
        target=state.previous_command_target.copy() if kind=='hold_command_seed' else state.joint_position.copy()
        commands=[];previous=state.previous_command_target.copy()
        for _ in range(c.settings.horizon_steps):
            desired=target.copy();desired[12:]=np.clip(desired[12:],previous[12:]-.1,previous[12:]+.1)
            commands.append(desired);previous=desired
        nominal=np.asarray(commands);before=rollout_for_search(c,state,nominal,reference)
        check=verify_sequence(c,state,nominal);task=None;repair=None
        accepted=check['accepted'] and before['physically_feasible']
        if accepted:chosen=nominal
        else:
            task=sequential_recovery_search(c,state,reference,nominal,iterations=12,time_limit_s=300.)
            repair=restore_feasibility(c,state,reference,task['final_search_commands'],iterations=8)
            chosen=repair['commands'] if repair['accepted'] else task['commands']
            accepted=chosen is not None
        row=dict(block=block,offset_from_original_query_s=.4*(block+1),accepted=bool(accepted),
            nominal_accepted=bool(check['accepted'] and before['physically_feasible']),
            nominal_maximum_physical_violation=before['maximum_physical_violation'],
            nominal_ground_violation_m=max(x['physical']['authored_ground_clearance_m'] for x in before['checks']),
            nominal_terminal_risk=before['terminal_risk'],task=task,repair=repair,commands=chosen)
        blocks.append(row)
        if not accepted:break
        confirmed=verify_sequence(c,state,chosen);pred=rollout_for_search(c,state,chosen,reference)
        if not confirmed['accepted'] or not pred['physically_feasible']:raise ValueError('Continuation failed independent verification')
        row.update(terminal_risk=pred['terminal_risk'],maximum_physical_violation=pred['maximum_physical_violation'],
            predicted_q=np.stack([state.q]+[n['following'].q for n in pred['profile']]),
            predicted_v=np.stack([state.v]+[n['following'].v for n in pred['profile']]))
        state=copy.deepcopy(pred['profile'][-1]['following'])
    return plain(dict(seed=kind,case=15,native_execution=False,blocks=blocks,
        verified_continuation_seconds=.4*sum(r['accepted'] for r in blocks),
        initial_model_state_q=initial.q,initial_model_state_v=initial.v,
        wall_seconds=time.perf_counter()-started))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--geometry',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve existing continuation evidence')
    m=json.loads((a.bundle/'manifest.json').read_text());archive=a.bundle/m['bundle_name']
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=m['bundle_sha256']:raise ValueError('Changed candidate')
    for name,digest in m['source_sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:raise ValueError('Changed candidate source '+name)
    start=time.perf_counter();rows=[];a.output.parent.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        for row in pool.map(query,[(a.bundle,a.geometry,k) for k in ['hold_command_seed','measured_pose_seed']]):
            rows.append(row);print(json.dumps({k:row[k] for k in ['seed','verified_continuation_seconds','wall_seconds']}),flush=True)
            a.output.with_suffix('.partial.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    paths=[Path(__file__),a.bundle/'manifest.json',archive,a.geometry,ROOT/'configs/go2_piper_racket_landing_model_v1.json']
    paths += [Path(m.__file__) for n,m in sys.modules.items() if n.startswith('scripts.') and getattr(m,'__file__',None)]
    result=dict(scope=__doc__,native_execution=False,rows=rows,wall_seconds=time.perf_counter()-start,
        limitations=['Replanning from predicted endpoints, not measured feedback.',
            'Authored ground geometry only, no self-collision or native-contact certificate.',
            'May predict beyond the original 8 s boundary; original evaluation is unchanged.'],
        source_sha256={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths})
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
