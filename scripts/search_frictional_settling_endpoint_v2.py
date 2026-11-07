"""Refine a settling endpoint with signed component constraints, then validate its tail.

QP slacks only propose a direction. Every line-search candidate is re-simulated
with unchanged geometry, motor, contact and terminal acceptance constraints.
No slack or reduced violation is accepted as a valid teacher trajectory.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import scripts.search_frictional_recovery_cem_v1 as core
from scripts.frictional_candidate_pool_v1 import FrictionalCandidatePoolV1
from scripts.dense_active_set_qp_v1 import DenseActiveSetQP
from scripts.terminal_settling_constraints_v2 import assess, constraints, rank
from scripts.extend_frictional_recovery_v1 import replay
from scripts.extend_recovery_to_episode_v1 import kinematic_window


def expand(values):
    knots=np.vstack([np.zeros((2,18)),np.asarray(values).reshape(3,18)])
    times=np.array([0.,12.,25.,55.,79.])
    return np.column_stack([np.interp(np.arange(80),times,knots[:,j]) for j in range(18)])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['bundle','sphere-audit','start-result','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--iterations',type=int,default=4)
    a=p.parse_args()
    if a.output.exists() or not 1<=a.iterations<=8:raise ValueError('Bounded new restoration required')
    sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    start=json.loads(a.start_result.read_text());hashes=dict(start['source_sha256'])
    for f,h in hashes.items():
        if sha(f)!=h:raise ValueError('Changed search input '+f)
    if start['horizon_steps']!=80 or not start['final']['solved']:
        raise ValueError('Require completed 80-step predicted trajectory')
    if str((a.bundle/'manifest.json').resolve()) not in hashes or str(a.sphere_audit.resolve()) not in hashes:
        raise ValueError('Different query binding')
    core.init_worker(str(a.bundle),str(a.sphere_audit),start['case'])
    c,state,previous,_,spheres,_=core.CONTEXT;state.previous_command_target=previous.copy()
    cfg=json.loads((a.bundle/'manifest.json').read_text())['configuration']
    seed=np.asarray(start['final']['commands']);current=np.zeros(54);records=[];started=time.monotonic()
    if seed.shape!=(80,18):raise ValueError('Invalid starting commands')
    files=[Path(__file__),a.start_result,c.qp.model.path]
    files += [Path(v.__file__) for k,v in list(sys.modules.items()) if k.startswith('scripts.') and getattr(v,'__file__',None)]
    for f in files:hashes[str(f.resolve())]=sha(f)
    with FrictionalCandidatePoolV1(cfg,spheres,workers=4,timeout_s=300.) as pool:
        def evaluate(points,save=False):
            values=pool.batch([dict(lane=i,state=state,seed=seed,correction=expand(x),save_states=True) for i,x in enumerate(points)])
            for value in values:
                value['settling']=assess(c,state,value,spheres)
                if not save:value['states']=None;value['commands']=None
            return values
        baseline=evaluate([current],True)[0];best=baseline
        if abs(best['maximum_scaled_violation']-start['final']['maximum_scaled_violation'])>1e-9:
            raise ValueError('Starting dynamics not reproduced')
        trust=.08
        for iteration in range(a.iterations if rank(best)[0] else 0):
            residual=constraints(best);epsilon=.001
            directions=np.eye(54)*epsilon
            values=evaluate([current+d for d in directions]+[current-d for d in directions])
            jac=np.zeros((len(residual),54));one_sided=0;unavailable=[]
            for j in range(54):
                pos,neg=values[j],values[j+54]
                if pos['solved'] and neg['solved'] and pos['settling']['solved'] and neg['settling']['solved']:
                    jac[:,j]=(constraints(pos)-constraints(neg))/(2*epsilon)
                elif pos['solved'] and pos['settling']['solved']:
                    jac[:,j]=(constraints(pos)-residual)/epsilon;one_sided+=1
                elif neg['solved'] and neg['settling']['solved']:
                    jac[:,j]=(residual-constraints(neg))/epsilon;one_sided+=1
                else:unavailable.append(j)
            n=54;m=len(residual)
            A=np.vstack([np.c_[jac,-np.eye(m)],np.c_[np.zeros((m,n)),np.eye(m)],np.c_[np.eye(n),np.zeros((n,m))]])
            low=np.r_[np.full(m,-np.inf),np.zeros(m),np.full(n,-trust)]
            high=np.r_[-residual,np.full(m,np.inf),np.full(n,trust)]
            for j in unavailable:low[2*m+j]=high[2*m+j]=0.
            solver=DenseActiveSetQP();solver.setup(P=np.diag(np.r_[np.ones(n),np.full(m,1e4)]),
                q=np.zeros(n+m),A=A,l=low,u=high,eps_abs=1e-7,eps_rel=1e-7)
            answer=solver.solve()
            record=dict(iteration=iteration,trust_rad=trust,one_sided_columns=one_sided,unavailable_columns=unavailable,
                qp_status=answer.info.status,before_max_violation=rank(best)[1])
            if answer.info.status_val!=1:
                record['accepted_direction']=False;records.append(record);break
            delta=answer.x[:n];scales=[1.,.5,.25,.125]
            candidates=[current+s*delta for s in scales]
            tested=evaluate(candidates);winner=min(range(len(tested)),key=lambda i:rank(tested[i]))
            record.update(proposal_slack_max=float(np.max(answer.x[n:])),
                direction_norm=float(np.linalg.norm(delta)),line_search=[dict(scale=s,accepted=not rank(r)[0],
                    solved=r['solved'],maximum_scaled_violation=rank(r)[1],error=r.get('error')) for s,r in zip(scales,tested)])
            improved=rank(tested[winner])<rank(best);record['accepted_direction']=improved
            if improved:
                current=candidates[winner];best=tested[winner];trust=min(.15,trust*1.25)
            else:trust*=.5
            record['after_max_violation']=rank(best)[1];records.append(record)
            print(json.dumps(record),flush=True)
            partial=dict(records=records,parameters=current.tolist(),source_sha256=hashes,native_execution=False,valid_teacher=False)
            a.output.with_suffix('.partial.json').write_text(json.dumps(partial,indent=2)+'\n')
            if not rank(best)[0]:break
        final=evaluate([current],True)[0]
    if rank(final)!=rank(best):raise ValueError('Independent terminal replay differs')
    tail=dict(attempted=False,reason='Settling targets not met')
    if not rank(final)[0]:
        from scripts.extend_frictional_recovery_v1 import set_state
        endpoint=final['states'][-1];command=np.asarray(final['commands'][-1])
        terminal=set_state(state,c.qp.model,endpoint['q'],endpoint['v'],endpoint['contacts'],command)
        try:
            held=replay(c,terminal,command,np.repeat(command[None],160,axis=0),spheres,core.CONTEXT[-1])
            protocol=json.loads((ROOT/'configs/evaluation_protocol_v2.json').read_text())['recovery']
            tail=dict(attempted=True,path_constraints_passed=True,model_window=kinematic_window(c.qp.model,held,protocol,.01))
        except ValueError as error:tail=dict(attempted=True,path_constraints_passed=False,error=repr(error))

    for f,h in hashes.items():
        if sha(f)!=h:raise ValueError('Restoration input changed '+f)
    def plain(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,dict):return {k:plain(x) for k,x in v.items()}
        if isinstance(v,list):return [plain(x) for x in v]
        return v
    out=dict(scope=__doc__,case=start['case'],horizon_steps=80,iterations=a.iterations,records=records,
        final=plain(final),settling_targets_met=not rank(final)[0],held_tail_validation=tail,parameters=current.tolist(),source_sha256=hashes,wall_seconds=time.monotonic()-started,
        native_execution=False,valid_teacher=False,
        limitation='QP slacks are only search directions; full nonlinear verification and native execution remain necessary')
    a.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
