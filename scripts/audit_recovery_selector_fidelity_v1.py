"""Check the first feedback departure from a fine-planned recovery prefix.

This isolates identical candidate requests at identical model states. It does
not certify the prefix natively or modify any running controller/search.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.pure_control_stance_foot_reference_v15 import PureControlStanceFootReferenceV15
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.audit_actual_handoff_model_v1 import state_from_record
from scripts.predict_self_contact_spheres_v2 import predict_self_contact_spheres
from scripts.audit_joint_sequence_proposal_v1 import plain


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['configuration','run','plan','feedback','output']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();read=lambda f:json.loads(f.read_text())
    if a.output.exists():raise ValueError('Fresh audit required')
    plan=read(a.plan);feedback=read(a.feedback);cfg=read(a.configuration)
    if plan['state_key']!=feedback['state_key']:raise ValueError('Different actual initial state')
    nominal=read(a.run/'native_command_inputs.json')['default_joint_position_rad']
    c=PureControlStanceFootReferenceV15(PositionPDInterface(**cfg['interface']),nominal,cfg['model_reference_library'])
    original=plan['initial_actual_state'];rows=plan['fine']['trajectory'];changed=None
    for k,row in enumerate(feedback['trajectory']):
        if np.max(abs(np.asarray(row['target'])-rows[k]['target']))>1e-8:
            changed=k;break
    if changed is None or changed==0:raise ValueError('Require a reproducible matching prefix before departure')
    before=rows[changed-1]
    actual_before=feedback['trajectory'][changed-1]
    for key in ['q','v','target','ground_force']:
        if np.max(abs(np.asarray(before[key])-actual_before[key]))>1e-8:
            raise ValueError('Prior model states already diverged: '+key)
    record=dict(original,q=before['q'],v=before['v'],previous=before['target'],
        ground_force=before['ground_force'],phase=before['phase'])
    state=state_from_record(c.model,record)
    c.chosen=dict(query=dict(q=original['q'],v=original['v'],
        contacts=original['phase']['ground_contacts'],target=original['previous']),
        trajectory=rows,source_key=hashlib.sha256(a.plan.read_bytes()).hexdigest())
    planned=np.asarray(rows[changed]['target'])
    selected=np.asarray(feedback['trajectory'][changed]['target'])
    candidates={'planned':planned,'selected_feedback':selected,'half_selected':.5*(planned+selected)}
    results=[]
    for name,target in candidates.items():
        for substeps in [16,160]:
            predicted=state;cost=0.;stages=[];failure=None
            for offset in range(min(2,len(rows)-changed)):
                command=target if offset==0 else np.asarray(rows[changed+offset]['target'])
                try:
                    pred=predict_self_contact_spheres(c.model,c.interface,predicted,command,
                        c.geometry.spheres,geometry=c.path_geometry,substeps=substeps)
                except (ValueError,np.linalg.LinAlgError) as error:
                    failure=dict(offset=offset,error=repr(error));break
                bad={n:v for n,v in pred['diagnostics'].items() if
                    (n=='nonfoot_ground_violation_m' and v>1e-6) or
                    (n in ['joint_position_violation_rad','joint_velocity_violation_rad_s','motor_torque_violation_nm'] and v>3e-4)}
                if bad:failure=dict(offset=offset,violations=bad);break
                predicted=pred['state'];predicted.previous_command_target=command.copy()
                desired=c.library.state(c.chosen,changed+offset)
                value=c._tracking_cost(predicted,desired);cost+=value
                stages.append(dict(offset=offset,tracking_cost=value,
                    maximum_velocity_error=float(np.max(abs(predicted.v-desired.v))),
                    maximum_configuration_error=float(np.max(abs(predicted.q-desired.q)))))
            results.append(dict(name=name,substeps=substeps,accepted=failure is None,
                cost=cost if failure is None else None,failure=failure,stages=stages))
    files=[Path(__file__),a.configuration,a.run/'native_command_inputs.json',a.plan,a.feedback,Path(cfg['model_reference_library'])]
    files += [Path(m.__file__) for n,m in list(sys.modules.items()) if n.startswith('scripts.') and getattr(m,'__file__',None)]
    result=dict(state_key=plan['state_key'],first_departure_step=changed,
        first_departure_tick=original['tick']+changed,
        maximum_command_change_rad=float(np.max(abs(planned-selected))),
        selected_source=feedback['trajectory'][changed]['source'],prior_states_equal=True,
        candidates=results,native_success=False,valid_expert_labels=False,
        source_sha256={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(plain(result),indent=2,allow_nan=False)+'\n')
    print(json.dumps(plain({k:v for k,v in result.items() if k!='source_sha256'})))


if __name__=='__main__':main()
