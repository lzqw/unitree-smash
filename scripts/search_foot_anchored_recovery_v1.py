"""Search a full recovery remainder using measured foot anchors and body/arm timing.

Offline full-body shooting through the original position actuator model. IK
is only a proposal mechanism. No candidate is a native success or expert label.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.check_constrained_recovery_suffix_v1 import source_context, state_from_row
from scripts.search_preserved_early_support_v1 import initialize
from scripts.search_landing_support_tail_v3 import _WORKER
from scripts.foot_anchored_recovery_reference_v1 import (
    FootAnchoredRecoveryReferenceV1, FootAnchoredRecoveryFeedbackV1,
    PARAMETER_NAMES, LOWER, UPPER, NOMINAL,
)
from scripts.foot_separation_path_constraint_v1 import FootSeparationPathConstraintV1
from scripts.hip_mount_path_constraint_v1 import HipMountPathConstraintV1
from scripts.position_limited_recovery_suffix_v1 import evaluate_suffix
from scripts.audit_joint_sequence_proposal_v1 import plain


def reference(parameters, query, tick):
    w = _WORKER; state = state_from_row(query); controller = w['controller']
    return FootAnchoredRecoveryReferenceV1(controller.qp.model, controller.qp.interface,
        state, w['spheres'], parameters, control_tick=tick)


def evaluate(parameters, query, tick, audit, substeps, deadline):
    w = _WORKER; state = state_from_row(query)
    ref = reference(parameters, query, tick)
    geometry = FootSeparationPathConstraintV1(
        HipMountPathConstraintV1(w['geometry'], audit, margin_rad=.02), w['spheres'], margin_m=.005)
    result = evaluate_suffix(w['controller'], state, FootAnchoredRecoveryFeedbackV1(ref),
        w['spheres'], geometry, w['protocol'], control_tick=tick, substeps=substeps,
        deadline=deadline, save=substeps == 160)
    for key, field in [('initial_q','q'),('initial_v','v'),('initial_contacts','contacts'),('initial_previous_command','target')]:
        if not np.array_equal(result[key], query[field]):
            raise ValueError('Measured initial state/history changed')
    failure = result['failure']
    if failure and (failure.get('collision') or {}).get('constraint_type') == 'foot_pair_clearance':
        failure['reason'] = 'foot_pair_clearance'
        fraction = ((geometry.last_violation['prediction_geometry_call']-1) % substeps)/substeps
        result['rank'][1] = result['required_model_steps']-result['completed_model_steps']-fraction
    return dict(parameters=np.asarray(parameters), result=result,
        ik_diagnostics=ref.ik_diagnostics,
        foot_center_targets_world=ref.foot_center_targets_world,
        minimum_checked_foot_gap_m=geometry.minimum_gap_m if np.isfinite(geometry.minimum_gap_m) else None)


def check_reference(query, tick):
    ref = reference(NOMINAL, query, tick); state = state_from_row(query)
    model = ref.model
    _, analytical = ref.centers(state.q, jacobian=True)
    numerical = np.zeros_like(analytical)
    eps = 1e-6
    for axis in range(model.model.nv):
        delta = np.zeros(model.model.nv); delta[axis] = eps
        numerical[:, :, axis] = (ref.centers(pin.integrate(model.model,state.q,delta))-
            ref.centers(pin.integrate(model.model,state.q,-delta)))/(2*eps)
    error = float(np.max(abs(analytical-numerical)))
    if error > 1e-7: raise ValueError('Authored foot center Jacobian differs '+str(error))
    policy = FootAnchoredRecoveryFeedbackV1(ref)
    previous = state.previous_command_target.copy()
    maximum_arm_step = 0.
    for t in range(tick,800):
        target,_ = policy.action(80+t-280,state,previous)
        if t == tick and not np.array_equal(target,previous): raise ValueError('Handoff differs')
        if not np.isfinite(target).all() or np.any(target<ref.interface.limits[:,0]-1e-10) or np.any(target>ref.interface.limits[:,1]+1e-10):
            raise ValueError('Original joint command bounds violated')
        maximum_arm_step = max(maximum_arm_step,float(np.max(abs(target[12:]-previous[12:]))))
        previous=target
    if maximum_arm_step > .01*ref.interface.arm_rate+1e-10: raise ValueError('Original arm slew violated')
    return dict(passed=True, jacobian_max_error=error, command_samples=800-tick,
        maximum_arm_step_rad=maximum_arm_step, actual_first_command_exact=True,
        scope='Kinematics and interface on actual query, not dynamic/native recovery')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-seconds',type=float,default=900.)
    a=p.parse_args()
    if a.output.exists() or not 120<=a.max_seconds<=1800: raise ValueError('Fresh bounded search required')
    read=lambda f:json.loads(Path(f).read_text())
    sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    inputs=read(a.inputs); cfg=inputs['configuration']; query=inputs['query'];tick=int(cfg['tick'])
    manifest=read(Path(cfg['bundle'])/'manifest.json')
    source=read(read(cfg['model_plan'])['source'])
    _,_,_,audit,context=source_context(source['configuration']['source'])
    context=list(copy.deepcopy(context));initial=manifest['query']
    if context[0]!=manifest['configuration']:raise ValueError('Physical configuration differs')
    context[1].update(initial_q=initial['q'],initial_v=initial['v'],initial_contacts=initial['contacts'],
        actual_previous_command=initial['target'],tick=280)
    with np.load(Path(cfg['bundle'])/manifest['bundle_name'],allow_pickle=False) as z:
        context[4]=z['tracking_reference_q'].copy();context[5]=z['tracking_reference_v'].copy()
    context[6]=manifest['tracking_gains'];initialize(*context)
    pins=dict(inputs['source_sha256'])
    for f in [a.inputs,Path(__file__)]:pins[str(f.resolve())]=sha(f)
    for name,module in list(sys.modules.items()):
        if name.startswith('scripts.') and getattr(module,'__file__',None):
            f=str(Path(module.__file__).resolve());digest=sha(f)
            if f in pins and pins[f]!=digest:raise ValueError('Pinned source changed '+f)
            pins[f]=digest
    for f,h in pins.items():
        if sha(f)!=h:raise ValueError('Changed source '+f)
    a.output.mkdir(parents=True)
    def write(name,obj):
        path=a.output/name;tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(plain(obj),indent=2,allow_nan=False)+'\n');tmp.replace(path)
    checks=check_reference(query,tick);write('reference_check.json',checks)
    write('inputs.json',dict(source_inputs=str(a.inputs),source_sha256=pins,query=query,tick=tick,
        parameter_names=PARAMETER_NAMES,lower=LOWER,upper=UPPER,seed=882192,
        max_seconds=a.max_seconds,workers=2,physical_constraints_changed=False))
    began=time.monotonic();deadline=began+a.max_seconds;search_end=began+.7*a.max_seconds
    rng=np.random.default_rng(882192);mean=NOMINAL.copy();std=.22*(UPPER-LOWER)
    records=[];fine=[];best=None;repeat=None
    write('status.json',dict(state='RUNNING',tick=tick))
    try:
        with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=tuple(context)) as pool:
            for generation in range(6):
                if time.monotonic()>=search_end:break
                candidates=np.clip(rng.normal(mean,std,(16,len(PARAMETER_NAMES))),LOWER,UPPER)
                candidates[0]=mean;candidates[1]=NOMINAL
                jobs=[pool.submit(evaluate,x,query,tick,audit,32,search_end) for x in candidates]
                rows=[f.result() for f in jobs]
                rows.sort(key=lambda r:r['result']['rank'])
                elite=np.asarray([r['parameters'] for r in rows[:4]])
                mean=elite.mean(axis=0);std=np.maximum(elite.std(axis=0),.04*(UPPER-LOWER))
                verified=[]
                if time.monotonic()<deadline-30:
                    jobs=[pool.submit(evaluate,r['parameters'],query,tick,audit,160,deadline-20) for r in rows[:2]]
                    verified=[f.result() for f in jobs];fine.extend(verified)
                    best=min(fine,key=lambda r:r['result']['rank'])
                records.append(dict(generation=generation,coarse=rows,fine=verified))
                write('partial.json',dict(records=records,best=best,native_execution=False))
                print(json.dumps(dict(generation=generation,coarse_steps=rows[0]['result']['completed_model_steps'],
                    fine_steps=best['result']['completed_model_steps'] if best else None)),flush=True)
                if best and best['result']['accepted']:break
            if best and best['result']['accepted'] and time.monotonic()<deadline:
                repeat=pool.submit(evaluate,best['parameters'],query,tick,audit,160,deadline).result()
        exact=False
        if repeat and repeat['result']['accepted']:
            left=best['result']['trajectory'];right=repeat['result']['trajectory']
            exact=len(left)==len(right) and all(np.array_equal(x[k],y[k]) for x,y in zip(left,right) for k in ['q','v','target','contacts'])
        for f,h in pins.items():
            if sha(f)!=h:raise ValueError('Changed source during solve '+f)
        write('result.json',dict(scope=__doc__,source_sha256=pins,records=records,best=best,
            independent_repeat=repeat,model_candidate_accepted=bool(exact),wall_seconds=time.monotonic()-began,
            native_execution=False,valid_expert_labels=False,online_mpc=False,
            limitations=['Full coupled prediction; foot-pair clearance without pair impulse response.',
                'Reference IK is not a dynamics feasibility certificate.',
                'Measured state is a model input only; no native teleportation.']))
        write('status.json',dict(state='COMPLETED',model_candidate_accepted=bool(exact)))
    except BaseException as error:
        write('status.json',dict(state='FAILED',error=repr(error)));raise


if __name__=='__main__':main()
