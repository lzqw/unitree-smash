"""Search front-foot height coordination from a measured failed family state.

Only reference knots change. Original measured state, full remaining episode,
actuator limits, clearance, friction and recovery checks remain unchanged.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np
import pinocchio as pin

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.check_constrained_recovery_suffix_v1 import source_context,state_from_row
from scripts.recovery_family_queries_v1 import transfer_reference
from scripts.search_preserved_early_support_v1 import initialize
from scripts.search_landing_support_tail_v3 import _WORKER
from scripts.current_state_recovery_planner_v3 import predict_plan
from scripts.audit_joint_sequence_proposal_v1 import plain


def foot_bottoms(model,q,spheres):
    """Sphere bottom geometry and its position derivative, not rolling velocity."""
    pin.computeJointJacobians(model.model,model.data,q)
    pin.updateFramePlacements(model.model,model.data)
    points=[];jacobians=[]
    for sphere in spheres:
        fid=model.body_frames[sphere['body']];pose=model.data.oMf[fid]
        offset=pose.rotation@np.asarray(sphere['center_body_m'])
        points.append(pose.translation+offset-np.array([0.,0.,sphere['radius_m']]))
        jac=pin.getFrameJacobian(model.model,model.data,fid,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        # The lowest geometric point's world vertical offset is constant. It
        # must not be differentiated as a material point rotating with the foot.
        jacobians.append((jac[:3]-pin.skew(offset)@jac[3:])[:,model.v_indices[:12]])
    return np.asarray(points),np.asarray(jacobians).reshape(12,12)


def reference_knots(model,interface,state,original,spheres,level,forward,down):
    knots=np.asarray(original,float).copy();errors=[]
    yaw=np.arctan2(state.root_rotation_world[1,0],state.root_rotation_world[0,0])
    direction=np.array([np.cos(yaw),np.sin(yaw),0.])
    if level==forward==down==0:return knots,[0.,0.,0.]
    for k in range(3):
        q=state.q.copy();q[model.q_indices[:12]]=knots[k,:12]
        q[model.q_indices[12:]]=np.clip(state.joint_position[12:]+knots[k,12:],
            interface.limits[12:,0],interface.limits[12:,1])
        goal,_=foot_bottoms(model,q,spheres)
        goal[:2,2]-=level*(goal[:2,2]-goal[:2,2].min())+down
        goal[:2]+=forward*direction
        for _ in range(60):
            points,jac=foot_bottoms(model,q,spheres);error=(goal-points).reshape(12)
            if np.max(abs(error))<1e-5:break
            dq=jac.T@np.linalg.solve(jac@jac.T+.003**2*np.eye(12),error)
            q[model.q_indices[:12]]=np.clip(q[model.q_indices[:12]]+np.clip(dq,-.12,.12),
                interface.limits[:12,0],interface.limits[:12,1])
        points,_=foot_bottoms(model,q,spheres)
        errors.append(float(np.max(np.linalg.norm(goal-points,axis=1))))
        knots[k,:12]=q[model.q_indices[:12]]
    return knots,errors


def evaluate(job):
    return predict_plan(job['plan'],job['query'],280,0,job['geometry'],job['substeps'],job['deadline'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['plan','search','failed-local','output']:parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Fresh output required')
    read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    failed=read(args.failed_local);setup=read(args.plan);search=read(args.search);source=read(setup['source'])
    if failed['accepted'] or failed['control_tick']!=280 or not failed['query_is_native']:
        raise ValueError('Require original failed measured-state query')
    _,_,_,geometry,context=source_context(source['configuration']['source'])
    initialize(*context);context,_=transfer_reference(context,state_from_row(search['query']))
    context,_=transfer_reference(context,state_from_row(failed['query']));initialize(*context)
    w=_WORKER;model=w['controller'].qp.model;interface=w['controller'].qp.interface
    state=state_from_row(failed['query']);warm=copy.deepcopy(failed['records'][0]['row']['plan'])
    if warm['corrections']:raise ValueError('Unmodified original reference required')
    # Independently check the geometric derivative used by the IK candidates.
    points,jac=foot_bottoms(model,state.q,w['spheres']);eps=1e-6;numeric=np.empty_like(jac)
    for i,vi in enumerate(model.v_indices[:12]):
        tangent=np.zeros(24);tangent[vi]=eps
        plus,_=foot_bottoms(model,pin.integrate(model.model,state.q,tangent),w['spheres'])
        minus,_=foot_bottoms(model,pin.integrate(model.model,state.q,-tangent),w['spheres'])
        numeric[:,i]=((plus-minus)/(2*eps)).reshape(12)
    jac_error=float(np.max(abs(jac-numeric)))
    if jac_error>1e-7:raise ValueError('Foot geometry derivative failed')
    pins=dict(failed['source_sha256'])
    for p in [args.plan,args.search,args.failed_local,Path(__file__).resolve()]:pins[str(p.resolve())]=sha(p)
    for p,h in pins.items():
        if sha(p)!=h:raise ValueError('Changed input '+p)
    began=time.monotonic();deadline=began+900.;proposals=[];rejected=[]
    for level in [0.,.5,1.]:
        for forward in [0.,.04,.08]:
            for down in [0.,.03]:
                knobs=dict(level=level,forward=forward,down=down)
                knots,errors=reference_knots(model,interface,state,warm['original_knots_rad'],w['spheres'],**knobs)
                record=dict(parameters=knobs,ik_max_errors_m=errors)
                if max(errors)>.005:rejected.append(record);continue
                plan=copy.deepcopy(warm);plan['original_knots_rad']=knots
                proposals.append(dict(record=record,plan=plan))
    args.output.mkdir(parents=True)
    def write(name,data):
        p=args.output/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(plain(data),indent=2,allow_nan=False)+'\n');t.replace(p)
    write('status.json',dict(state='SEARCHING',case=failed['case'],proposals=len(proposals),ik_rejected=len(rejected)))
    with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=context) as pool:
        def batch(plans,resolution):
            jobs={pool.submit(evaluate,dict(plan=p,query=failed['query'],geometry=geometry,
                substeps=resolution,deadline=deadline)):i for i,p in enumerate(plans)}
            result=[None]*len(plans)
            for future in as_completed(jobs):result[jobs[future]]=future.result()
            return result
        baseline=batch([warm],160)[0]
        old=failed['records'][0]['row']['result']
        for key in ['accepted','completed_model_steps','failure','rank','terminal']:
            if baseline['result'][key]!=old[key]:raise ValueError('Changed baseline '+key)
        coarse=batch([r['plan'] for r in proposals],32)
        order=sorted(range(len(coarse)),key=lambda i:coarse[i]['result']['rank'])[:8]
        fine=batch([coarse[i]['plan'] for i in order],160)
        best=min([baseline,*fine],key=lambda r:r['result']['rank']);verified=None
        if best['result']['accepted']:
            replay=batch([best['plan']],160)[0]
            if replay['result']['accepted'] and all(np.array_equal(a[k],b[k]) for a,b in
                    zip(best['result']['trajectory'],replay['result']['trajectory']) for k in ['q','v','target','contacts']):
                if len(replay['result']['trajectory'])==520:verified=replay
    for p,h in pins.items():
        if sha(p)!=h:raise ValueError('Input changed during search '+p)
    write('result.json',dict(scope=__doc__,case=failed['case'],query=failed['query'],
        proposals=proposals,ik_rejected=rejected,coarse=coarse,fine=fine,best=best,verified=verified,
        geometric_jacobian_error=jac_error,baseline_reproduced=True,source_sha256=pins,
        wall_seconds=time.monotonic()-began,physical_constraints_changed=False,native_execution=False,
        valid_expert_labels=False,physical_impossibility_claim=False))
    write('status.json',dict(state='COMPLETED',accepted=verified is not None,
        best_completed=best['result']['completed_model_steps'],case=failed['case']))
    print(json.dumps(dict(accepted=verified is not None,completed=best['result']['completed_model_steps'],
        failure=best['result']['failure'],jacobian_error=jac_error,seconds=time.monotonic()-began)))


if __name__=='__main__':main()
