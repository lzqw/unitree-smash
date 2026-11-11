"""CPU-only replay of the first persistent-fallback states.

Replays the existing QP/sequence controller from saved native states and
records object-level geometry failures, solver status/residuals and contact
assumptions. No Isaac/native rollout and no command is executed.
"""
import copy, csv, json, hashlib, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pinocchio as pin

ROOT=Path(__file__).resolve().parents[1]; MAIN=Path('/home/autodrive/LZQW/unitree_smash_contact_recovery')
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(MAIN))
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.loaded_pinocchio_model_v1 import PublicState
from scripts.pure_control_state_driven_long_mpc_v1_qp_continuity_repair import PureControlStateDrivenLongMPCV1QPContinuityRepair
from scripts.audit_sequence_repair_run import finite

RUN=ROOT/'reports/pure_control_state_driven_long_mpc_qp_continuity_repair_all128_corrected_20261110_03/native'
BASE=MAIN/'reports/pure_control_state_driven_long_mpc_qp_continuity_all128_20261109_01/native'
OUT=ROOT/'reports/fallback_conflicts_cpu_20261111'

def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def make_state(c, row, data, index):
    q=data['q'][index].copy();v=data['v'][index].copy()
    s=PublicState(q,v,pin.Quaternion(q[3:7]).matrix(),q[:3].copy(),
                  q[c.model.q_indices].copy(),v[c.model.v_indices].copy(),
                  np.array(row['phase']['ground_contacts']),0.)
    s.previous_command_target=data['previous'][index].copy()
    s.measured_force_world=data['ground_force'][index].copy()
    return s

def labels(c):
    out=[]
    for i,shape in enumerate(c.geometry.shapes):
        if 'constraint' in shape:
            out.append((i,shape['constraint'],shape['body']))
        else:
            out.append((i,'ground',shape['body']))
    return out

def wrap(c, capture):
    old_pred=c._predict_request
    def pred(state, requested):
        before=copy.deepcopy(state)
        try: result=old_pred(state, requested)
        except Exception as exc:
            capture['prediction_exceptions'].append(dict(error=type(exc).__name__+': '+str(exc), requested=np.asarray(requested).tolist()))
            raise
        gaps,_=c.geometry.evaluate(result['state'].q)
        bad=np.flatnonzero(gaps < -1e-6)
        capture['predictions'].append(dict(requested=np.asarray(requested).tolist(),
            previous=before.previous_command_target.tolist(),
            contacts=before.contacts.tolist(),
            gaps=[float(x) for x in gaps],
            min_gap_m=float(gaps.min()),
            negative=[dict(index=int(i),kind=labels(c)[i][1],body=labels(c)[i][2],gap_m=float(gaps[i])) for i in bad],
            diagnostics=dict(result.get('diagnostics',{})),
            events=result.get('events',[])))
        return result
    c._predict_request=pred
    old_solve=c.solve
    def solve(state, reference, **kw):
        p=old_solve(state,reference,**kw)
        capture['solves'].append({k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in p.items() if k not in ('reference','contacts','acceleration','torque','force_world','joint_position_target','normalized_action','predicted_joint_position','predicted_joint_velocity')})
        return p
    c.solve=solve
    old_worker=c._make_repair_worker
    def worker():
        w=old_worker(); wrap(w,capture); return w
    c._make_repair_worker=worker

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    metrics=read(RUN/'metrics.json'); fixed=sorted(r['training_index'] for r in read(BASE/'metrics.json')['complete5_case_outcomes'] if r['air']); assert len(fixed)==74
    rows=[json.loads(x) for x in (RUN/'pure_control_diagnostics.jsonl').open()]
    with np.load(RUN/'pure_control_commands.npz') as z: data={k:z[k].copy() for k in z.files}
    assert all((r['tick'],r['lane'])==(data['tick'][i],data['lane'][i]) for i,r in enumerate(rows))
    config=read(ROOT/'reports/pure_control_state_driven_long_mpc_qp_continuity_repair_native_preparation_20261110_01/controller.json')
    nominal=np.asarray(read(RUN/'native_command_inputs.json')['default_joint_position_rad'])
    first_by_lane={}
    for i,r in enumerate(rows):
        lane=int(r['lane'])
        if lane in fixed and r['diagnostic'].get('replanned') and not r['diagnostic'].get('solver_accepted') and lane not in first_by_lane:
            first_by_lane[lane]=i
    all_cases=[]; all_predictions=[]; all_solves=[]; exceptions=[]
    for n,lane in enumerate(fixed):
        index=first_by_lane[lane]; row=rows[index]
        c=PureControlStateDrivenLongMPCV1QPContinuityRepair(PositionPDInterface(**config['interface']),nominal)
        cap={'lane':lane,'tick':int(row['tick']),'predictions':[],'solves':[],'prediction_exceptions':[]}
        wrap(c,cap)
        state=make_state(c,row,data,index)
        try:
            target,diag=c.command(state,row['phase']['phase'],float(data['time_s'][index]))
            cap['result']=dict(target=target.tolist(),diagnostic=diag)
        except Exception as exc:
            cap['exception']=type(exc).__name__+': '+str(exc)
        all_cases.append(cap); all_predictions += [dict(lane=lane,tick=cap['tick'],node=j,**p) for j,p in enumerate(cap['predictions'])]
        all_solves += [dict(lane=lane,tick=cap['tick'],**p) for p in cap['solves']]
        exceptions += [dict(lane=lane,tick=cap['tick'],**p) for p in cap['prediction_exceptions']]
        print(lane,cap['tick'],len(cap['predictions']),len(cap['solves']),cap.get('exception',''),flush=True)
    with (OUT/'geometry_conflicts.csv').open('w',newline='') as f:
        fields=['lane','tick','node','min_gap_m','negative','contacts','diagnostics'];w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
        for p in all_predictions:w.writerow({k:(json.dumps(p[k]) if isinstance(p[k],(list,dict)) else p.get(k)) for k in fields})
    with (OUT/'qp_solve_records.jsonl').open('w') as f:
        for x in all_solves:f.write(json.dumps(finite(x))+'\n')
    counts=Counter(); objects=Counter(); nodes=Counter(); statuses=Counter();
    for p in all_predictions:
        if p['negative']:
            for x in p['negative']:objects[(x['kind'],x['body'])]+=1;nodes[(x['kind'],x['body'],p['node'])]+=1
    for p in all_solves:
        statuses[p.get('solver_status','missing')]+=1
        for key in ['residuals','violations']:
            for k,v in (p.get(key) or {}).items():
                if isinstance(v,(int,float)) and v>0:counts[(key,k)]+=1
    summary=dict(status='CPU_DIAGNOSTIC_COMPLETE',native_evaluations=0,fixed74=fixed,
        first_persistent_fallback_states=len(first_by_lane),prediction_nodes=len(all_predictions),solve_records=len(all_solves),
        prediction_exceptions=exceptions,geometry_objects=[dict(kind=k[0],body=k[1],count=v) for k,v in objects.most_common()],
        geometry_node_counts=[dict(kind=k[0],body=k[1],node=k[2],count=v) for k,v in nodes.most_common()],
        solver_status=dict(statuses),positive_residual_fields=[dict(scope=k[0],field=k[1],count=v) for k,v in counts.most_common()],
        checker_contract='Existing geometry check is diagnostic only; original collision, actuator, friction and contact rules unchanged.',
        contact_model='State.contacts is measured foot support; inactive feet have force constrained exactly zero in the QP.',
        source_sha256={str(p):sha(p) for p in [Path(__file__),ROOT/'scripts/whole_body_support_qp_v1.py',ROOT/'scripts/pure_control_interval_qp_v1.py',ROOT/'scripts/recovery_surface_constraints_v1.py',ROOT/'scripts/predict_self_contact_spheres_v2.py']})
    (OUT/'SUMMARY.json').write_text(json.dumps(finite(summary),indent=2)+'\n')
    (OUT/'all_cases.json').write_text(json.dumps(finite(all_cases),indent=2)+'\n')
    print(json.dumps(finite(summary),indent=2))

if __name__=='__main__':main()
