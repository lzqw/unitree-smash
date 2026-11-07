"""Structured whole-body sequence search under the new frictional predictor.

Three interpolated full-joint correction knots, original command limits and
arm slew. Uses true model rollouts for all constraints, never a short-period
QP score as an acceptance certificate. This is offline model search only.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.parallel_landing_solver_v1 import build_controller
from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance
from scripts.predict_frictional_spheres_v1 import predict_frictional_spheres

CONTEXT=None


def init_worker(bundle,sphere_path,case):
    global CONTEXT
    bundle=Path(bundle);m=json.loads((bundle/'manifest.json').read_text());c=build_controller(m['configuration'])
    key=f'case_{case:03d}';limits=c.qp.interface.limits
    with np.load(bundle/m['bundle_name']) as z:
        state=c.qp.model.state_from_public(z[key+'_query_public'],np.asarray(m['configuration']['posture_target']))
        previous=limits[:,0]+.5*(z[key+'_previous_action']+1)*np.diff(limits).ravel()
        seed=z[key+'_commands_rad'][:40].copy()
    geometry=AuthoredGroundClearance(ROOT/'configs/go2_piper_authored_ground_geometry_v1.json',c.qp.model,margin_m=0.)
    CONTEXT=(c,state,previous,seed,json.loads(Path(sphere_path).read_text())['spheres'],geometry)


def admit_sequence(seed,knots,previous,interface):
    ticks=np.linspace(0,len(seed)-1,len(knots));target=seed.copy()
    for joint in range(18):target[:,joint]+=np.interp(np.arange(len(seed)),ticks,knots[:,joint])
    target=np.clip(target,interface.limits[:,0],interface.limits[:,1]);prior=previous.copy()
    for row in target:
        row[12:]=np.clip(row[12:],prior[12:]-.01*interface.arm_rate,prior[12:]+.01*interface.arm_rate)
        prior=row.copy()
    return target


def evaluate(knots,save_states=False):
    c,initial,previous,seed,spheres,geometry=CONTEXT
    state=copy.deepcopy(initial);commands=admit_sequence(seed,np.asarray(knots),previous,c.qp.interface)
    cost=0.;maximum=0.;raw={};states=[];first_failure=None
    try:
        for tick,command in enumerate(commands):
            result=predict_frictional_spheres(c.qp.model,c.qp.interface,state,command,spheres,geometry=geometry)
            state=result['state'];d=result['diagnostics']
            for key,value in d.items():raw[key]=max(raw.get(key,0.),value)
            violations=[max(0,d['nonfoot_ground_violation_m']-1e-6)/.005,
                max(0,d['joint_position_violation_rad']-3e-4)/.02,
                max(0,d['joint_velocity_violation_rad_s']-3e-4)/2.,
                max(0,d['motor_torque_violation_nm']-3e-4)]
            local=max(violations);maximum=max(maximum,local)
            if local>0 and first_failure is None:first_failure=tick
            R=state.root_rotation_world
            roll=np.arctan2(R[2,1],R[2,2]);pitch=np.arcsin(np.clip(-R[2,0],-1.,1.))
            omega=np.linalg.norm(state.v[3:6]);horizontal=np.linalg.norm((R@state.v[:3])[:2])
            joint_speed=np.max(abs(state.joint_velocity))
            cost+=(20*(roll*roll+pitch*pitch)+5*omega*omega+5*horizontal*horizontal+
                .05*np.sum(state.joint_velocity**2))/len(commands)
            if save_states:states.append(dict(q=state.q.tolist(),v=state.v.tolist(),contacts=state.contacts.tolist()))
            # Grossly invalid paths cannot be teacher trajectories. Stop wasting
            # computation there, but never call a truncated path accepted.
            if maximum>20:raise ValueError('Gross model constraint violation')
        margin=float(c.support_margin(state));vertical=float(abs((state.root_rotation_world@state.v[:3])[2]))
        support_deficit=1e3 if not np.isfinite(margin) else max(0.,(c.settings.terminal_support_margin_m-margin)/.01)
        terminal=max(0.,abs(roll)/np.deg2rad(15)-1,abs(pitch)/np.deg2rad(15)-1,
            omega-1,horizontal/.3-1,joint_speed-1,vertical/.1-1,support_deficit)
        maximum=max(maximum,terminal)
        cost+=10*(20*(roll*roll+pitch*pitch)+5*omega*omega+5*horizontal*horizontal+.05*np.sum(state.joint_velocity**2))
        cost+=np.mean((commands-seed)**2)
        return dict(solved=True,accepted=bool(maximum<=1e-10),maximum_scaled_violation=float(maximum),cost=float(cost),
            first_violation_tick=first_failure,diagnostics=raw,terminal=dict(roll=float(roll),pitch=float(pitch),
                omega=float(omega),horizontal=float(horizontal),joint_speed=float(joint_speed),vertical_speed=vertical,
                support_margin_m=margin if np.isfinite(margin) else None,support_feet=int(state.contacts.sum())),
            states=states if save_states else None,commands=commands.tolist() if save_states else None)
    except (ValueError,np.linalg.LinAlgError) as error:
        return dict(solved=False,accepted=False,maximum_scaled_violation=max(1e3,maximum),cost=1e6,
            error=repr(error),first_violation_tick=first_failure,diagnostics=raw)


def key(result):return (not result['accepted'],result['maximum_scaled_violation'],result['cost'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['bundle','sphere-audit','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--case',type=int,default=3);p.add_argument('--population',type=int,default=48)
    p.add_argument('--iterations',type=int,default=4);p.add_argument('--seed',type=int,default=882095);a=p.parse_args()
    if a.output.exists() or not 16<=a.population<=128 or not 1<=a.iterations<=8:raise ValueError('Preserve evidence and bounded search')
    sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    m=json.loads((a.bundle/'manifest.json').read_text());archive=a.bundle/m['bundle_name']
    if sha(archive)!=m['bundle_sha256']:raise ValueError('Changed seed')
    sphere=json.loads(a.sphere_audit.read_text())
    for f,digest in sphere['source_sha256'].items():
        if sha(f)!=digest:raise ValueError('Changed sphere inputs '+f)
    files=[Path(__file__),a.bundle/'manifest.json',archive,a.sphere_audit,
        ROOT/'scripts/predict_frictional_spheres_v1.py',ROOT/'scripts/frictional_impulse_step_v1.py',
        ROOT/'configs/go2_piper_authored_ground_geometry_v1.json']
    hashes={str(f.resolve()):sha(f) for f in files};rng=np.random.default_rng(a.seed)
    mean=np.zeros((3,18));std=np.tile(np.r_[np.full(12,.04),np.full(6,.03)],(3,1));best_knots=mean.copy();best=None;history=[]
    started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn'),
            initializer=init_worker,initargs=(str(a.bundle),str(a.sphere_audit),a.case)) as pool:
        baseline=pool.submit(evaluate,mean).result();best=baseline
        for iteration in range(a.iterations):
            candidates=mean+rng.normal(size=(a.population,3,18))*std
            candidates[0]=best_knots;candidates[1]=np.zeros((3,18))
            values=list(pool.map(evaluate,candidates));order=sorted(range(len(values)),key=lambda i:key(values[i]))
            elite=candidates[order[:max(6,a.population//6)]]
            mean=.2*mean+.8*elite.mean(0);std=np.maximum(.002,.3*std+.7*elite.std(0))
            winner=order[0]
            if key(values[winner])<key(best):best=values[winner];best_knots=candidates[winner].copy()
            row=dict(iteration=iteration,best=best,accepted=sum(v['accepted'] for v in values),solver_failures=sum(not v['solved'] for v in values))
            history.append(row);print(json.dumps(row),flush=True)
            a.output.with_suffix('.partial.json').write_text(json.dumps(history,indent=2,allow_nan=False)+'\n')
        independent=pool.submit(evaluate,best_knots,True).result()
    if key(independent)!=key(best):raise ValueError('Independent replay differs')
    for f,digest in hashes.items():
        if sha(f)!=digest:raise ValueError('Inputs changed during search')
    result=dict(scope=__doc__,native_execution=False,valid_teacher=False,case=a.case,seed=a.seed,
        population=a.population,iterations=a.iterations,baseline=baseline,history=history,
        final=independent,knots=best_knots.tolist(),wall_seconds=time.perf_counter()-started,source_sha256=hashes,
        limitation='A 0.4 s model-feasible candidate is not a complete native recovery or a full stable window')
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
