"""Compare dispatched one-step responses and active-plan predictions to native data.

One-step diagnostics start from measured state and use the decoded final target.
Plan errors use the plan active for the preceding physical interval, including
at replanning boundaries. Neither comparison changes physics or task scoring.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pinocchio as pin

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.check_constrained_recovery_suffix_v1 import source_context
from scripts.search_preserved_early_support_v1 import initialize
from scripts.search_landing_support_tail_v3 import _WORKER
from scripts.check_early_support_dispatch_v1 import state_from_arrays
from scripts.hip_mount_path_constraint_v1 import HipMountPathConstraintV1
from scripts import predict_frictional_spheres_v1 as predictor


def state_error(model,pq,pv,aq,av):
    pq,pv,aq,av=map(np.asarray,[pq,pv,aq,av]);dq=pin.difference(model.model,aq,pq)
    pr=pin.Quaternion(pq[3:7]).matrix();ar=pin.Quaternion(aq[3:7]).matrix()
    return dict(root_position_m=float(np.linalg.norm(pq[:3]-aq[:3])),
        root_attitude_rad=float(np.linalg.norm(dq[3:6])),
        joint_position_max_rad=float(np.max(abs(dq[model.v_indices]))),
        root_linear_velocity_world_m_s=float(np.linalg.norm(pr@pv[:3]-ar@av[:3])),
        root_angular_velocity_world_rad_s=float(np.linalg.norm(pr@pv[3:6]-ar@av[3:6])),
        joint_velocity_max_rad_s=float(np.max(abs(pv[model.v_indices]-av[model.v_indices]))))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['run','bundle','model-source','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--ticks',type=int,nargs='+',default=[280,300,320,340,350,351,360,375,400,450,500,600,700,750])
    a=p.parse_args();sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if a.output.exists():raise ValueError('Preserve earlier response evidence')
    job=json.loads((a.run/'job.json').read_text())
    if job['status']!='SUCCEEDED' or job['exit_code']!=0:raise ValueError('Completed native run required')
    manifest=json.loads((a.bundle/'manifest.json').read_text());meta=json.loads((a.run/'sequence_native_result.json').read_text())
    _,_,_,audit,context=source_context(a.model_source);initialize(*context);w=_WORKER
    if context[0]!=manifest['configuration']:raise ValueError('Prediction/execution configuration mismatch')
    controller=w['controller'];model=controller.qp.model
    if sha(a.bundle/manifest['bundle_name'])!=meta['bundle_sha256']:raise ValueError('Wrong executed bundle')
    with np.load(a.run/'sequence_native_trace.npz',allow_pickle=False) as z:trace={k:z[k].copy() for k in z.files}
    with np.load(a.bundle/manifest['bundle_name'],allow_pickle=False) as z:
        nominal={k:z['case_003_predicted_'+k].copy() for k in ['q','v','contacts']}
    records=meta['records'];ticks=[r['control_tick'] for r in records];index={t:i for i,t in enumerate(ticks)}
    if len(index)!=len(records) or ticks!=list(range(meta['start_tick'],meta['start_tick']+len(records))):
        raise ValueError('Continuous original episode trace required')
    if not np.array_equal(trace['previous_target_rad'][1:],trace['final_target_rad'][:-1]):
        raise ValueError('Actual command history is discontinuous')
    limits=controller.qp.interface.limits
    decoded=limits[:,0]+.5*(trace['final_action']+1.)*(limits[:,1]-limits[:,0])
    decode_error=float(np.max(abs(decoded-trace['final_target_rad'])))
    if decode_error>1e-6:raise ValueError('Target differs from decoded dispatched action')
    planning=a.run/'current_state_planning.json';calls=[]
    if planning.exists():calls=json.loads(planning.read_text())['calls']
    pins={**manifest['source_sha256']}
    files=[Path(__file__),a.run/'job.json',a.run/'sequence_native_result.json',a.run/'sequence_native_trace.npz',
        a.bundle/'manifest.json',a.bundle/manifest['bundle_name'],a.model_source]
    if planning.exists():files.append(planning)
    files += [Path(m.__file__) for n,m in list(sys.modules.items()) if n.startswith('scripts.') and getattr(m,'__file__',None)]
    for path in files:pins[str(path.resolve())]=sha(path)
    for path,h in pins.items():
        if sha(path)!=h:raise ValueError('Changed input '+path)
    rows=[];original=predictor.frictional_velocity_step
    for tick in sorted(set(a.ticks)):
        if tick not in index or tick+1 not in index:continue
        i=index[tick];j=index[tick+1]
        state=state_from_arrays(model,trace['q'][i],trace['v'][i],trace['contacts'][i],trace['foot_force_world_n'][i])
        state.previous_command_target=trace['previous_target_rad'][i].copy();target=trace['final_target_rad'][i].copy()
        observed=state.contacts.copy();last_normal=None
        def observe(*args,**kwargs):
            nonlocal observed,last_normal
            answer=original(*args,**kwargs)
            last_normal=answer['impulse'].reshape(4,3)[:,2]/args[8]
            observed=np.where(observed,abs(last_normal)>.5,abs(last_normal)>1.)
            return answer
        geometry=HipMountPathConstraintV1(w['geometry'],audit,margin_rad=.02)
        row=dict(control_tick=tick,next_tick=tick+1,native_contacts_before=state.contacts.tolist(),
            native_contacts_after=trace['contacts'][j].tolist(),native_next_force_world_n=trace['foot_force_world_n'][j].tolist(),
            actual_command_used=True)
        predictor.frictional_velocity_step=observe
        try:
            pred=predictor.predict_frictional_spheres(model,controller.qp.interface,state,target,w['spheres'],
                substeps=160,geometry=geometry)
            next_state=pred['state'];row.update(prediction_rejected=False,
                errors=state_error(model,next_state.q,next_state.v,trace['q'][j],trace['v'][j]),
                predicted_impulse_contacts=next_state.contacts.tolist(),
                predicted_force_hysteresis_contacts=observed.tolist(),predicted_last_normal_force_n=last_normal.tolist(),
                raw_contact_disagreements=int(np.sum(next_state.contacts!=trace['contacts'][j])),
                aligned_contact_disagreements=int(np.sum(observed!=trace['contacts'][j])),
                diagnostics=pred['diagnostics'],first_ground_violation=pred['first_ground_violation'])
        except (ValueError,np.linalg.LinAlgError) as error:
            row.update(prediction_rejected=True,reason=repr(error),hip_constraint=geometry.last_violation)
        finally:predictor.frictional_velocity_step=original
        rows.append(row)
    plan_rows=[]
    for tick in ticks[1:]:
        i=index[tick];eligible=[c for c in calls if c['accepted'] and c['control_tick']<tick]
        if eligible:
            active=eligible[-1];origin=active['control_tick'];pr=active['prediction']['trajectory'][tick-origin-1]
            pq,pv,pc=pr['q'],pr['v'],pr['contacts'];label='accepted_current_state_plan'
        else:
            origin=meta['start_tick'];definition=manifest.get('replanned_suffix_definition')
            # At the query instant itself use the preceding prefix prediction;
            # the new plan has not yet produced a physical transition.
            if definition and tick==definition['control_tick']:continue
            if definition and tick>definition['control_tick']:origin=definition['control_tick']
            offset=tick-meta['start_tick'];pq,pv,pc=[nominal[k][offset] for k in ['q','v','contacts']]
            label='offline_suffix_prediction' if definition and tick>definition['control_tick'] else 'offline_prefix_prediction'
        plan_rows.append(dict(control_tick=tick,plan_tick=origin,prediction_steps=tick-origin,kind=label,
            errors=state_error(model,pq,pv,trace['q'][i],trace['v'][i]),
            contact_flags_equal=bool(np.array_equal(pc,trace['contacts'][i]))))
    good=[r for r in rows if not r['prediction_rejected']]
    summary=dict(sampled_native_transitions=len(rows),computed_transitions=len(good),
        rejected_transitions=[r['control_tick'] for r in rows if r['prediction_rejected']],
        errors={k:dict(median=float(np.median([r['errors'][k] for r in good])),maximum=max(r['errors'][k] for r in good))
            for k in good[0]['errors']} if good else {})
    for path,h in pins.items():
        if sha(path)!=h:raise ValueError('Input changed during response audit '+path)
    result=dict(scope=__doc__,summary=summary,one_step=rows,active_plan_errors=plan_rows,source_sha256=pins,
        maximum_action_decode_error_rad=decode_error,
        native_rerun=False,teacher_promoted=False,valid_expert_labels=False,
        limitations=['Native 1/0.5 N normal-force hysteresis differs from positive model impulse contact.',
            'Aligned labels do not reproduce contact compliance, self-collision response or the full native sensor.',
            'Constraint rejection and response error are separate; error correlation alone does not establish cause.',
            'One-step errors use actual commands; long-horizon plan errors also include feedback-induced command changes.'])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
