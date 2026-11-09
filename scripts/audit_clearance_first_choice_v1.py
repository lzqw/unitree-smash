"""Re-predict first changed commands with a longer declared nominal tail.

Starts from the common MODEL state before the first divergence, not a native
reset. Only the first command differs; later commands use the stored planned
reference. This diagnoses a candidate horizon, not a new closed-loop result.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.pure_control_clearance_reference_v25 import PureControlClearanceReferenceV25, surface_margin_penalty
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.audit_actual_handoff_model_v1 import state_from_record
from scripts.event_recovery_manager_v1 import authored_clearance_state
from scripts.predict_self_contact_spheres_v2 import predict_self_contact_spheres
from scripts.audit_settled_stance_model_v2 import ForceRecorder
from scripts.retry_budgeted_settling_feedback_v1 import sha, write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['baseline','candidate','configuration','run','output']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();read=lambda p:json.loads(Path(p).read_text())
    if a.output.exists():raise ValueError('Fresh audit required')
    inp=read(a.candidate/'inputs.json');pins=dict(inp['source_sha256'])
    for f,h in pins.items():
        if sha(f)!=h:raise ValueError('Changed source '+f)
    cfg=read(a.configuration);nominal=read(a.run/'native_command_inputs.json')['default_joint_position_rad']
    if pins.get(str(a.configuration.resolve()))!=sha(a.configuration):raise ValueError('Keep execution interface')
    results=[];deadline=time.monotonic()+300
    for key in inp['states']:
        name='state_'+key.replace(':','_')+'.json'
        files=[a.baseline/name,a.candidate/name,a.baseline/'candidate_snapshots'/name]
        old,new,plan=[read(f) for f in files]
        if old['initial_actual_state']!=new['initial_actual_state']:raise ValueError('Different actual origin')
        pairs=list(zip(old['trajectory'],new['trajectory']))
        changed=[i for i,(x,y) in enumerate(pairs) if np.max(abs(np.asarray(x['target'])-y['target']))>1e-8]
        if not changed:raise ValueError('No changed command')
        index=changed[0]
        for x,y in pairs[:index]:
            if any(not np.array_equal(x[k],y[k]) for k in ['q','v','target','ground_force']):
                raise ValueError('States diverged before the command')
        origin=(old['initial_actual_state'] if index==0 else
                dict(old['trajectory'][index-1],previous=old['trajectory'][index-1]['target']))
        case=dict(state_key=key,first_changed_tick=old['trajectory'][index]['tick'],
                  state_origin='common predicted state after identical model prefix',rollouts=[])
        for label,trace in [('baseline_first',old),('clearance_first',new)]:
            c=PureControlClearanceReferenceV25(PositionPDInterface(**cfg['interface']),nominal,cfg['model_reference_library'])
            state=state_from_record(c.model,origin);rows=[];failure=None;cost=0.
            for offset in range(min(40,len(plan['fine']['trajectory'])-index)):
                if time.monotonic()>deadline:raise TimeoutError('Bounded diagnostic expired')
                target=np.asarray(trace['trajectory'][index]['target'] if offset==0 else plan['fine']['trajectory'][index+offset]['target'])
                recorder=ForceRecorder(c.model)
                try:
                    pred=predict_self_contact_spheres(c.model,c.interface,state,target,c.geometry.spheres,
                        substeps=160,geometry=c.path_geometry,collision_response=recorder)
                except (ValueError,np.linalg.LinAlgError) as error:
                    failure=dict(offset=offset,error=repr(error));break
                bad={k:v for k,v in pred['diagnostics'].items() if
                     (k=='nonfoot_ground_violation_m' and v>1e-6) or
                     (k in ['joint_position_violation_rad','joint_velocity_violation_rad_s','motor_torque_violation_nm'] and v>3e-4)}
                if bad:
                    failure=dict(offset=offset,violations=bad,collision=pred['first_ground_violation']);break
                state=pred['state'];state.previous_command_target=target.copy()
                state.measured_force_world=recorder.last_force.copy()
                state.contacts=np.where(state.contacts,recorder.last_force[:,2]>.5,recorder.last_force[:,2]>1.)
                surfaces=authored_clearance_state(c.geometry,state);penalty,_=surface_margin_penalty(surfaces)
                desired=c.library.state(dict(query=plan['initial_actual_state'],trajectory=plan['fine']['trajectory']),index+offset)
                cost+=c._tracking_cost(state,desired)
                head=min(surfaces[n]['gap_m'] for n in ['Head_lower','Head_upper'])
                row=dict(offset=offset,end_time_s=(offset+1)*.01,head_gap_m=head,surface_cost=penalty,cumulative_objective=cost)
                if offset==0:
                    expected=trace['trajectory'][index]
                    row['first_prediction_q_error']=float(np.max(abs(state.q-np.asarray(expected['q']))))
                    row['first_prediction_v_error']=float(np.max(abs(state.v-np.asarray(expected['v']))))
                    if max(row['first_prediction_q_error'],row['first_prediction_v_error'])>1e-8:
                        raise ValueError('Cannot reproduce first predicted transition')
                rows.append(row)
            case['rollouts'].append(dict(label=label,steps=len(rows),failure=failure,trajectory=rows))
        results.append(case);pins.update({str(f.resolve()):sha(f) for f in files})
    for f,h in pins.items():
        if sha(f)!=h:raise ValueError('Changed diagnostic source '+f)
    pins[str(Path(__file__).resolve())]=sha(__file__)
    write(a.output,dict(results=results,source_sha256=pins,native_execution=False,native_success=False,
        new_controller=False,continuation='One differing command then unchanged original planned tail, fine model',
        limitation='Future feedback decisions are not simulated; a longer nominal-tail prediction is not a verified recovery policy.'))
    print(json.dumps([dict(state_key=r['state_key'],rollouts=[dict(label=x['label'],steps=x['steps'],failure=x['failure']) for x in r['rollouts']]) for r in results]))


if __name__=='__main__':main()
