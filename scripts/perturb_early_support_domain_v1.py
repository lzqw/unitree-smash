"""Probe nearby physical model states for the fixed early recovery controller.

Finite deterministic probes are not a certified region, sensor uncertainty
test, native perturbation experiment, or a change to the query admission gate.
"""
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
import numpy as np
import pinocchio as pin

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.search_landing_support_tail_v3 import initialize_worker,_WORKER
from scripts.search_early_landing_trajectory_v1 import evaluate
from scripts.audit_joint_sequence_proposal_v1 import plain


def perturb_state(initial,spec):
    state=copy.deepcopy(initial)
    kind=spec['kind']
    if kind=='nominal':return state
    axis=spec['world_axis'];amount=spec['amount']
    if axis not in (0,1) or not np.isfinite(amount):raise ValueError('Declared finite planar-axis probe required')
    delta=np.zeros(3);delta[axis]=amount
    rotation=initial.root_rotation_world
    if kind=='orientation':
        changed=pin.exp3(delta)@rotation
        state.q[3:7]=pin.Quaternion(changed).coeffs()
        state.root_rotation_world=changed
        # Keep BOTH world linear and angular velocity unchanged when changing
        # orientation; otherwise this would be an undeclared combined probe.
        state.v[:3]=changed.T@rotation@initial.v[:3]
        state.v[3:6]=changed.T@rotation@initial.v[3:6]
    elif kind=='linear_velocity':
        state.v[:3]+=rotation.T@delta
    elif kind=='angular_velocity':
        state.v[3:6]+=rotation.T@delta
    else:raise ValueError('Unknown physical perturbation')
    return state


def run_probe(job):
    spec,knots,deadline=job;w=_WORKER
    state=perturb_state(w['initial'],spec)
    result=evaluate(w['controller'],state,knots,w['spheres'],w['geometry'],w['protocol'],
        substeps=160,deadline=deadline,save=True,policy_factory=w['factory'],control_tick=280)
    return dict(probe=spec,initial_q=state.q.copy(),initial_v=state.v.copy(),result=result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-seconds',type=float,default=900.);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    if a.output.exists() or not 300<=a.max_seconds<=1800 or not 1<=a.workers<=4:
        raise ValueError('Fresh output and bounded probe budget required')
    read=lambda f:json.loads(Path(f).read_text());sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    source=read(a.source)
    if not source['any_model_feasible'] or not source['replay_reproduced']:raise ValueError('Reproduced nominal candidate required')
    hashes=dict(source['source_sha256']);hashes[str(a.source.resolve())]=sha(a.source)
    hashes[str(Path(__file__).resolve())]=sha(__file__)
    for f,h in hashes.items():
        if sha(f)!=h:raise ValueError('Changed probe source '+f)
    tail=read(source['configuration']['source']);parent=read(tail['configuration']['source_search'])
    cfg=parent['configuration'];query=read(cfg['query']);manifest=read(Path(cfg['bundle'])/'manifest.json')
    initargs=(manifest['configuration'],query,source['fixed_early_knots_rad'],cfg['sphere_audit'])
    probes=[dict(name='nominal',kind='nominal')]
    for kind,size,unit in [('orientation',np.deg2rad(.5),'rad'),('linear_velocity',.02,'m/s'),('angular_velocity',.02,'rad/s')]:
        for axis in (0,1):
            for sign in (-1,1):
                probes.append(dict(name=f'{kind}_world_{axis}_{sign:+d}',kind=kind,world_axis=axis,amount=sign*size,unit=unit))
    start=time.monotonic();deadline=start+a.max_seconds;rows=[]
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn'),
            initializer=initialize_worker,initargs=initargs) as pool:
        for row in pool.map(run_probe,[(s,source['best_knots_rad'],deadline) for s in probes],chunksize=1):
            rows.append(row)
            compact={k:v for k,v in row['result'].items() if k!='trajectory'}
            print(json.dumps(plain(dict(probe=row['probe'],result=compact))),flush=True)
            tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(plain(rows),allow_nan=False)+'\n');tmp.replace(a.output.with_suffix('.partial.json'))
    nominal=rows[0]['result'];expected=source['independent_replay']
    if nominal['accepted']!=expected['accepted'] or len(nominal['trajectory'])!=520:
        raise ValueError('Nominal probe failed to reproduce accepted reference')
    error=max(float(np.max(abs(np.asarray(x[k])-y[k]))) for x,y in zip(nominal['trajectory'],expected['trajectory']) for k in ['target','q','v'])
    if error>1e-12:raise ValueError('Nominal probe changed the saved controller')
    for f,h in hashes.items():
        if sha(f)!=h:raise ValueError('Changed source during probes '+f)
    expired=sum((x['result'].get('failure') or {}).get('reason')=='declared_compute_budget' for x in rows)
    output=dict(scope=__doc__,case=3,tick=280,substeps=160,probe_count=len(rows),rows=rows,
        accepted_model_probes=sum(x['result']['accepted'] for x in rows),compute_expired_probes=expired,
        nominal_maximum_difference=error,wall_seconds=time.monotonic()-start,source_sha256=hashes,
        original_reference_and_gains_fixed=True,original_command_history_fixed=True,
        native_execution=False,query_admission_widened=False,valid_expert_labels=False,
        limitations=['Twelve axis probes do not sample all joint, actuator, contact, ball or combined uncertainties.',
            'Model perturbations are physical-state counterfactuals, not native resets or teacher successes.',
            'Passing nominal and nearby probes cannot certify a robust invariant region.'])
    a.output.write_text(json.dumps(plain(output),indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
