"""Bounded offline search for a continuous early landing and support trajectory.

All candidates start from a verified recorded ground-start state. Three leg
target knots are jointly searched and followed by persistent attitude feedback;
the arm brakes towards its measured starting configuration. Only a complete
fine-resolution constrained continuation may be called model-feasible. This
script neither deploys controls nor supplies positive expert labels.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.interpolate import PchipInterpolator

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.parallel_landing_solver_v1 import build_controller
from scripts.loaded_pinocchio_model_v1 import PublicState
from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance
from scripts.predict_frictional_spheres_v1 import predict_frictional_spheres
from scripts.terminal_attitude_feedback_v1 import TerminalAttitudeFeedbackV1
from scripts.feedback_tail_rollout_v1 import terminal_measures,terminal_violations,kinematically_stable
from scripts.audit_joint_sequence_proposal_v1 import plain


class EarlyReference:
    def __init__(self,model,interface,initial,knots):
        knots=np.asarray(knots,float)
        if knots.shape!=(3,12) or not np.isfinite(knots).all():
            raise ValueError('Three finite leg target knots required')
        self.model=model;self.interface=interface;self.initial=initial
        # PCHIP avoids component-wise overshoot between bounded targets. The
        # first requested control is exactly the preceding actual command.
        bounded=np.clip(knots,interface.limits[:12,0],interface.limits[:12,1])
        self.curve=PchipInterpolator([0.,.08,.20,.50],
            np.vstack([initial.previous_command_target[:12],bounded]),axis=0,extrapolate=False)

    def action(self,offset,state,previous):
        step=offset-80
        if type(step) is not int or not 0<=step<500:
            raise ValueError('Declared complete early-recovery clock required')
        t=.01*step;u=float(np.clip(t/.04,0.,1.));blend=u*u*(3.-2.*u)
        target=self.initial.previous_command_target.copy()
        target[:12]=self.curve(min(t,.50))
        target[12:]=(1.-blend)*target[12:]+blend*self.initial.joint_position[12:]
        target=np.clip(target,self.interface.limits[:,0],self.interface.limits[:,1])
        target[12:]=np.clip(target[12:],previous[12:]-.01*self.interface.arm_rate,
                           previous[12:]+.01*self.interface.arm_rate)
        return target,'continuous_early_landing_reference'


def evaluate(controller,initial,knots,spheres,geometry,protocol,*,substeps,deadline,save=False,
             policy_factory=EarlyReference,control_tick=300):
    if type(control_tick) is not int or not 280<=control_tick<=300:
        raise ValueError('Declared early query clock required')
    length=800-control_tick
    state=copy.deepcopy(initial);interface=controller.qp.interface
    base=policy_factory(controller.qp.model,interface,initial,knots)
    policy=TerminalAttitudeFeedbackV1(base,proportional=.3,damping_s=.04,maximum_joint_delta_rad=.08)
    first,_=policy.action(80,state,state.previous_command_target)
    if not np.array_equal(first,initial.previous_command_target):
        raise ValueError('First command must preserve actual dispatch continuity')
    start=time.monotonic();failure=None;trailing=longest=0;cost=0.;rows=[];end=None;completed=0
    for step in range(length):
        if time.monotonic()>=deadline:
            failure=dict(step=step,reason='declared_compute_budget');break
        previous=state.previous_command_target.copy()
        target,_=policy.action(80+step,state,previous)
        if (not np.isfinite(target).all() or np.max(abs(target[12:]-previous[12:]))>.1+1e-10):
            raise ValueError('Unexecutable command candidate')
        try:
            pred=predict_frictional_spheres(controller.qp.model,interface,state,target,spheres,
                                          geometry=geometry,substeps=substeps)
        except (ValueError,np.linalg.LinAlgError) as error:
            failure=dict(step=step,reason='model_solver_rejected',error=repr(error));break
        d=pred['diagnostics']
        bad={k:v for k,v in d.items() if (k=='nonfoot_ground_violation_m' and v>1e-6) or
            (k in ['joint_position_violation_rad','joint_velocity_violation_rad_s','motor_torque_violation_nm'] and v>3e-4)}
        if bad:
            failure=dict(step=step,reason='path_constraints',violations=bad,
                         first_ground_violation=pred['first_ground_violation']);break
        state=pred['state'];state.previous_command_target=target.copy();completed+=1
        end=terminal_measures(controller,state)
        trailing=trailing+1 if kinematically_stable(end,protocol) and state.q[2]>=.12 else 0
        longest=max(longest,trailing)
        cost+=.01*(20*(end['roll']**2+end['pitch']**2)+5*end['omega']**2+
                   5*end['horizontal']**2+.05*np.sum(state.joint_velocity**2)+.1*np.sum((target-previous)**2))
        if save:rows.append(dict(target=target.copy(),q=state.q.copy(),v=state.v.copy(),contacts=state.contacts.copy()))
    violations=terminal_violations(end,.01) if end else {'no_endpoint':True}
    if trailing<80:violations['trailing_window_s']=trailing*.01
    if state.q[2]<.12:violations['root_height_m']=float(state.q[2])
    accepted=failure is None and completed==length and not violations
    # A better failure rank is only a search heuristic, never acceptance.
    # Within the same failed control interval, prefer a later first physical
    # collision, not a lower effort obtained by letting the body fall earlier.
    fraction=0.
    if failure and failure.get('first_ground_violation') and set(failure.get('violations',{}))=={'nonfoot_ground_violation_m'}:
        fraction=failure['first_ground_violation']['micro']/substeps
    rank=[0 if accepted else 1,length-completed-fraction,0 if accepted else max(0,80-trailing),cost/max(1,completed)]
    return dict(accepted=accepted,completed_model_steps=completed,required_model_steps=length,
        rank=rank,failure=failure,terminal=end,terminal_violations=violations,
        trailing_model_window_s=trailing*.01,longest_model_window_s=longest*.01,
        cost=float(cost),substeps=substeps,wall_seconds=time.monotonic()-start,
        trajectory=rows if save else None,native_execution=False,valid_expert_labels=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--query',type=Path,required=True)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--sphere-audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--population',type=int,default=24)
    p.add_argument('--generations',type=int,default=4)
    p.add_argument('--max-seconds',type=float,default=900.)
    p.add_argument('--seed',type=int,default=882110)
    a=p.parse_args()
    if a.output.exists() or not 8<=a.population<=64 or not 1<=a.generations<=8 or not 60<=a.max_seconds<=1800:
        raise ValueError('Fresh output and bounded search required')
    read=lambda x:json.loads(Path(x).read_text());sha=lambda x:hashlib.sha256(Path(x).read_bytes()).hexdigest()
    query=read(a.query)
    if query['case']!=3 or query['tick']!=300 or query['native_execution'] or query['initial_nonfoot_gap_m']<0:
        raise ValueError('Verified early ground-start query required')
    for f,h in query['source_sha256'].items():
        if sha(f)!=h:raise ValueError('Changed query provenance '+f)
    manifest=read(a.bundle/'manifest.json');controller=build_controller(manifest['configuration']);model=controller.qp.model
    q=np.asarray(query['initial_q']);v=np.asarray(query['initial_v'])
    import pinocchio as pin
    initial=PublicState(q,v,pin.Quaternion(q[3:7]).matrix(),q[:3].copy(),q[model.q_indices].copy(),
                        v[model.v_indices].copy(),np.asarray(query['initial_contacts'],bool),0.)
    initial.previous_command_target=np.asarray(query['actual_previous_command'])
    if initial.contacts.any():raise ValueError('This reference search is for the verified flight query')
    geometry_path=ROOT/'configs/go2_piper_authored_ground_geometry_v1.json'
    geometry=AuthoredGroundClearance(geometry_path,model,margin_m=0.)
    protocol_path=ROOT/'configs/evaluation_protocol_v2.json';protocol=read(protocol_path)['recovery']
    native_path=Path(manifest['baseline'])/'native_command_inputs.json'
    default=np.asarray(read(native_path)['default_joint_position_rad'])
    spheres=read(a.sphere_audit)['spheres']
    sources=[Path(__file__),a.query,a.bundle/'manifest.json',a.sphere_audit,geometry_path,protocol_path,native_path,model.path]
    sources += [Path(m.__file__) for n,m in list(sys.modules.items()) if n.startswith('scripts.') and getattr(m,'__file__',None)]
    hashes={str(f.resolve()):sha(f) for f in sources}
    limits=controller.qp.interface.limits[:12]
    seed=np.stack([.5*(initial.previous_command_target[:12]+default[:12]),default[:12],default[:12]])
    lower=np.maximum(limits[:,0],seed-.8);upper=np.minimum(limits[:,1],seed+.8)
    mean=np.clip(seed,lower,upper);std=np.full((3,12),.22);rng=np.random.default_rng(a.seed)
    began=time.monotonic();deadline=began+a.max_seconds;search_end=began+.65*a.max_seconds
    best=None;best_knots=None;ledger=[];candidates=[]
    a.output.parent.mkdir(parents=True,exist_ok=True)
    for generation in range(a.generations):
        population=np.clip(rng.normal(mean,std,(a.population,3,12)),lower,upper)
        population[0]=mean if best_knots is None else best_knots
        ranked=[]
        for i,knots in enumerate(population):
            if time.monotonic()>=search_end:break
            result=evaluate(controller,initial,knots,spheres,geometry,protocol,substeps=16,deadline=search_end)
            ledger.append(dict(generation=generation,index=i,knots_rad=knots.copy(),result=result))
            ranked.append((result['rank'],knots.copy(),result));candidates.append((result['rank'],knots.copy()))
            if best is None or result['rank']<best['rank']:
                best=result;best_knots=knots.copy()
                print(json.dumps(plain(dict(generation=generation,index=i,best=best))),flush=True)
        a.output.with_suffix('.partial.json').write_text(json.dumps(plain(ledger),allow_nan=False)+'\n')
        if not ranked:break
        ranked.sort(key=lambda x:x[0]);elite=np.stack([x[1] for x in ranked[:max(2,len(ranked)//4)]])
        mean=elite.mean(0);std=np.maximum(elite.std(0),.035)
    verified=[];seen=set()
    for rank,knots in sorted(candidates,key=lambda x:x[0]):
        identity=knots.tobytes()
        if identity in seen:continue
        seen.add(identity)
        if len(verified)>=2 or time.monotonic()>=deadline:break
        fine=evaluate(controller,initial,knots,spheres,geometry,protocol,substeps=160,deadline=deadline,save=True)
        verified.append(dict(knots_rad=knots,fine=fine))
        print(json.dumps(plain(dict(fine={k:v for k,v in fine.items() if k!='trajectory'}))),flush=True)
    for f,h in hashes.items():
        if sha(f)!=h:raise ValueError('Changed search source '+f)
    output=dict(scope=__doc__,case=3,tick=300,seed=a.seed,
        configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        search_space='3 bounded leg-position knots; PCHIP; arm measured-pose braking; persistent attitude feedback',
        knot_times_s=[.08,.20,.50],bounds_lower_rad=lower,bounds_upper_rad=upper,
        ledger=ledger,fine_candidates=verified,candidate_count=len(ledger),
        any_model_feasible=any(x['fine']['accepted'] for x in verified),wall_seconds=time.monotonic()-began,
        source_sha256=hashes,native_execution=False,valid_expert_labels=False,
        limitations=['Finite structured search cannot establish physical impossibility.',
            'Model has no head contact response or self-collision certification.',
            'No task success, robust domain, MPC deployment or student learning is claimed.'])
    a.output.write_text(json.dumps(plain(output),indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
