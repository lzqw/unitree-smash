"""Repair a saved model-only recursive-feasibility failure without resetting it.

Keeps the exact failed q/v, contact state and executed command history. Uses
the unchanged nonlinear checker; local QP slacks never certify a trajectory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.frictional_candidate_pool_v1 import FrictionalCandidatePoolV1
from scripts.parallel_landing_solver_v1 import build_controller
from scripts.extend_frictional_recovery_v1 import set_state
from scripts.restore_frictional_terminal_v1 import violation_vector,expand
from scripts.dense_active_set_qp_v1 import DenseActiveSetQP
from scripts.search_frictional_recovery_cem_v1 import key
from scripts.audit_joint_sequence_proposal_v1 import plain


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['bundle','sphere-audit','failed-result','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--iterations',type=int,default=4)
    a=p.parse_args()
    if a.output.exists() or not 1<=a.iterations<=8:raise ValueError('Fresh bounded repair required')
    sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    failed=json.loads(a.failed_result.read_text());hashes=dict(failed['source_sha256'])
    if failed['failed_query'] is None or failed['native_execution']:
        raise ValueError('Require explicitly model-only failed query')
    for name,digest in hashes.items():
        if sha(name)!=digest:raise ValueError('Changed query source '+name)
    if str((a.bundle/'manifest.json').resolve()) not in hashes or str(a.sphere_audit.resolve()) not in hashes:
        raise ValueError('Different query binding')
    manifest=json.loads((a.bundle/'manifest.json').read_text());cfg=manifest['configuration']
    c=build_controller(cfg);spheres=json.loads(a.sphere_audit.read_text())['spheres'];case=failed['case']
    if c.settings.terminal_support_margin_m!=.01:raise ValueError('Constraint vector assumes original 1 cm margin')
    with np.load(a.bundle/manifest['bundle_name'],allow_pickle=False) as z:
        template=c.qp.model.state_from_public(z[f'case_{case:03d}_query_public'],np.asarray(cfg['posture_target']))
    f=failed['failed_query'];state=set_state(template,c.qp.model,f['q'],f['v'],f['contacts'],f['previous_command'])
    seed=np.asarray(f['warm_commands']);current=np.zeros(54);records=[];began=time.monotonic()
    if seed.shape!=(80,18):raise ValueError('Require complete failed horizon')
    for file in [Path(__file__),a.failed_result,c.qp.model.path]+[Path(v.__file__) for k,v in list(sys.modules.items()) if k.startswith('scripts.') and getattr(v,'__file__',None)]:
        hashes[str(file.resolve())]=sha(file)
    with FrictionalCandidatePoolV1(cfg,spheres,workers=4,timeout_s=300.) as pool:
        def evaluate(points,save=False,shift=0):
            return pool.batch([dict(lane=i+shift,state=state,seed=seed,correction=expand(x),save_states=save) for i,x in enumerate(points)])
        best=evaluate([current],True)[0]
        expected=failed['records'][-1]['decision']['records'][0]['result']
        if abs(best['maximum_scaled_violation']-expected['maximum_scaled_violation'])>1e-9:
            raise ValueError('Failed warm sequence did not reproduce')
        trust=.03
        for iteration in range(a.iterations if not best['accepted'] else 0):
            residual=violation_vector(best);epsilon=.001;directions=np.eye(54)*epsilon
            values=evaluate([current+d for d in directions]+[current-d for d in directions])
            jac=np.zeros((len(residual),54));unavailable=[]
            for j in range(54):
                pos,neg=values[j],values[j+54]
                if pos['solved'] and neg['solved']:jac[:,j]=(violation_vector(pos)-violation_vector(neg))/(2*epsilon)
                elif pos['solved']:jac[:,j]=(violation_vector(pos)-residual)/epsilon
                elif neg['solved']:jac[:,j]=(residual-violation_vector(neg))/epsilon
                else:unavailable.append(j)
            n=54;m=len(residual)
            A=np.vstack([np.c_[jac,-np.eye(m)],np.c_[np.zeros((m,n)),np.eye(m)],np.c_[np.eye(n),np.zeros((n,m))]])
            low=np.r_[np.full(m,-np.inf),np.zeros(m),np.full(n,-trust)]
            high=np.r_[-residual,np.full(m,np.inf),np.full(n,trust)]
            for j in unavailable:low[2*m+j]=high[2*m+j]=0.
            solver=DenseActiveSetQP();solver.setup(P=np.diag(np.r_[np.ones(n),np.full(m,1e4)]),q=np.zeros(n+m),A=A,l=low,u=high,eps_abs=1e-7,eps_rel=1e-7)
            answer=solver.solve();record=dict(iteration=iteration,qp_status=answer.info.status,before_max_violation=best['maximum_scaled_violation'])
            if answer.info.status_val!=1:records.append(record);break
            delta=answer.x[:n];scales=[1.,.5,.25,.125];points=[current+s*delta for s in scales]
            tested=evaluate(points);winner=min(range(len(tested)),key=lambda i:key(tested[i]))
            improved=key(tested[winner])<key(best)
            record.update(proposal_slack_max=float(np.max(answer.x[n:])),unavailable_columns=unavailable,
                line_search=[dict(scale=s,accepted=r['accepted'],maximum_scaled_violation=r['maximum_scaled_violation']) for s,r in zip(scales,tested)],improved=improved)
            if improved:current=points[winner];best=tested[winner];trust=min(.05,trust*1.25)
            else:trust*=.5
            record['after_max_violation']=best['maximum_scaled_violation'];records.append(record)
            print(json.dumps(record),flush=True)
            if best['accepted']:break
        final=evaluate([current],True,shift=1)[0]
    if key(final)!=key(best):raise ValueError('Independent replay differs')
    for name,digest in hashes.items():
        if sha(name)!=digest:raise ValueError('Input changed '+name)
    output=dict(scope=__doc__,case=case,offset=failed['predicted_executed_steps'],horizon_steps=80,
        maximum_iterations=a.iterations,completed_iterations=len(records),records=records,
        final=final,failed_query=f,parameters=current,source_sha256=hashes,
        wall_seconds=time.monotonic()-began,native_execution=False,valid_teacher=False)
    a.output.write_text(json.dumps(plain(output),indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
