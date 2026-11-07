"""Joint leg/arm trajectory search at an observed post-release airborne state.

Three coupled 18-joint target knots use the original PD/rate/effort interface.
The original model evaluator checks all steps to the real episode end and a
trailing stable window. This is offline candidate generation, not a deployed
MPC, a physical feasibility proof, or successful correction data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import pinocchio as pin
from scipy.interpolate import PchipInterpolator

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.parallel_landing_solver_v1 import build_controller
from scripts.loaded_pinocchio_model_v1 import PublicState
from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance
from scripts.search_early_landing_trajectory_v1 import evaluate
from scripts.audit_joint_sequence_proposal_v1 import plain


class WholeBodyEarlyReference:
    """Leg absolute knots and arm offsets about the measured-pose brake."""
    def __init__(self,model,interface,initial,knots):
        knots=np.asarray(knots,float)
        if knots.shape!=(3,18) or not np.isfinite(knots).all():
            raise ValueError('Three finite full-body knots required')
        self.model=model;self.interface=interface;self.initial=initial
        legs=np.clip(knots[:,:12],interface.limits[:12,0],interface.limits[:12,1])
        self.legs=PchipInterpolator([0.,.08,.20,.50],np.vstack([initial.previous_command_target[:12],legs]),axis=0)
        self.arms=PchipInterpolator([0.,.08,.20,.50],np.vstack([np.zeros(6),knots[:,12:]]),axis=0)

    def action(self,offset,state,previous):
        step=offset-80
        if type(step) is not int or not 0<=step<520:
            raise ValueError('Complete original early episode clock required')
        t=.01*step;u=float(np.clip(t/.04,0.,1.));blend=u*u*(3.-2.*u)
        target=self.initial.previous_command_target.copy()
        target[:12]=self.legs(min(t,.50))
        target[12:]=(1.-blend)*target[12:]+blend*self.initial.joint_position[12:]+self.arms(min(t,.50))
        target=np.clip(target,self.interface.limits[:,0],self.interface.limits[:,1])
        target[12:]=np.clip(target[12:],previous[12:]-.01*self.interface.arm_rate,
                           previous[12:]+.01*self.interface.arm_rate)
        return target,'continuous_whole_body_landing_reference'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['query','bundle','sphere-audit','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--population',type=int,default=64)
    parser.add_argument('--generations',type=int,default=8)
    parser.add_argument('--max-seconds',type=float,default=900.)
    parser.add_argument('--seed',type=int,default=882111)
    a=parser.parse_args()
    if a.output.exists() or not 8<=a.population<=64 or not 1<=a.generations<=8 or not 60<=a.max_seconds<=1800:
        raise ValueError('Fresh output and finite search required')
    read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    query=read(a.query);tick=query['tick']
    if query['case']!=3 or tick not in (280,300) or query['native_execution'] or query['initial_nonfoot_gap_m']<0:
        raise ValueError('Previously verified post-release airborne query required')
    for f,h in query['source_sha256'].items():
        if sha(f)!=h:raise ValueError('Changed original query source '+f)
    manifest=read(a.bundle/'manifest.json');controller=build_controller(manifest['configuration'])
    model=controller.qp.model;interface=controller.qp.interface
    q=np.asarray(query['initial_q']);v=np.asarray(query['initial_v'])
    initial=PublicState(q,v,pin.Quaternion(q[3:7]).matrix(),q[:3].copy(),q[model.q_indices].copy(),
        v[model.v_indices].copy(),np.asarray(query['initial_contacts'],bool),0.)
    initial.previous_command_target=np.asarray(query['actual_previous_command'])
    if initial.contacts.any():raise ValueError('Query must remain airborne')
    geometry_path=ROOT/'configs/go2_piper_authored_ground_geometry_v1.json'
    geometry=AuthoredGroundClearance(geometry_path,model,margin_m=0.)
    protocol_path=ROOT/'configs/evaluation_protocol_v2.json';protocol=read(protocol_path)['recovery']
    native_path=Path(manifest['baseline'])/'native_command_inputs.json'
    default=np.asarray(read(native_path)['default_joint_position_rad']);spheres=read(a.sphere_audit)['spheres']
    sources=[Path(__file__),a.query,a.bundle/'manifest.json',a.sphere_audit,geometry_path,protocol_path,native_path,model.path]
    sources += [Path(m.__file__) for n,m in list(sys.modules.items()) if n.startswith('scripts.') and getattr(m,'__file__',None)]
    hashes={str(f.resolve()):sha(f) for f in sources}
    seed=np.zeros((3,18));seed[:,:12]=np.stack([.5*(initial.previous_command_target[:12]+default[:12]),default[:12],default[:12]])
    lower=np.zeros((3,18));upper=lower.copy()
    lower[:,:12]=np.maximum(interface.limits[:12,0],seed[:,:12]-.8)
    upper[:,:12]=np.minimum(interface.limits[:12,1],seed[:,:12]+.8)
    lower[:,12:]=np.maximum(-.6,interface.limits[12:,0]-initial.joint_position[12:])
    upper[:,12:]=np.minimum(.6,interface.limits[12:,1]-initial.joint_position[12:])
    mean=np.clip(seed,lower,upper);std=np.tile(np.r_[np.full(12,.22),np.full(6,.12)],(3,1))
    rng=np.random.default_rng(a.seed);started=time.monotonic();deadline=started+a.max_seconds
    search_deadline=started+.65*a.max_seconds;best=None;best_knots=None;ledger=[];candidates=[]
    def run(knots,substeps,limit,save=False):
        return evaluate(controller,initial,knots,spheres,geometry,protocol,substeps=substeps,
            deadline=limit,save=save,policy_factory=WholeBodyEarlyReference,control_tick=tick)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    for generation in range(a.generations):
        population=np.clip(rng.normal(mean,std,(a.population,3,18)),lower,upper)
        population[0]=mean if best_knots is None else best_knots
        # Preserve a fixed measured-arm control in each population; compare
        # it honestly instead of assuming more action dimensions must help.
        population[1]=population[0];population[1,:,12:]=0.
        ranked=[]
        for i,knots in enumerate(population):
            if time.monotonic()>=search_deadline:break
            result=run(knots,16,search_deadline)
            row=dict(generation=generation,index=i,knots_rad=knots.copy(),result=result)
            ledger.append(row);ranked.append((result['rank'],knots.copy()));candidates.append((result['rank'],knots.copy()))
            if best is None or result['rank']<best['rank']:
                best=result;best_knots=knots.copy();print(json.dumps(plain(dict(generation=generation,index=i,best=best))),flush=True)
        a.output.with_suffix('.partial.json').write_text(json.dumps(plain(ledger),allow_nan=False)+'\n')
        if not ranked:break
        ranked.sort(key=lambda x:x[0]);elite=np.stack([x[1] for x in ranked[:max(2,len(ranked)//4)]])
        mean=elite.mean(0);std=np.maximum(elite.std(0),np.r_[np.full(12,.035),np.full(6,.02)][None,:])
    fine=[];seen=set()
    for rank,knots in sorted(candidates,key=lambda x:x[0]):
        if knots.tobytes() in seen:continue
        seen.add(knots.tobytes())
        if len(fine)>=2 or time.monotonic()>=deadline:break
        result=run(knots,160,deadline,save=True);fine.append(dict(knots_rad=knots,fine=result))
        print(json.dumps(plain(dict(fine={k:v for k,v in result.items() if k!='trajectory'}))),flush=True)
    for f,h in hashes.items():
        if sha(f)!=h:raise ValueError('Changed search source '+f)
    output=dict(scope=__doc__,case=3,tick=tick,required_model_steps=800-tick,
        configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        lower=lower,upper=upper,ledger=ledger,fine_candidates=fine,candidate_count=len(ledger),
        any_model_feasible=any(x['fine']['accepted'] for x in fine),wall_seconds=time.monotonic()-started,
        source_sha256=hashes,native_execution=False,valid_expert_labels=False,
        limitations=['Finite 54-parameter search does not prove unreachability.',
            'Initial state is an actual ground-start observation; no new native execution is performed.',
            'Full feedback continuation is checked; model has no nonfoot contact response or self-collision certification.'])
    a.output.write_text(json.dumps(plain(output),indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
