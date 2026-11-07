"""Replay an existing recovery plan under the frictional sphere model.

Stops at the first original hardware/geometry constraint violation. Passing
only verifies this additional predictive model, never native recovery.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.parallel_landing_solver_v1 import build_controller
from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance
from scripts.predict_frictional_spheres_v1 import predict_frictional_spheres
from scripts.extend_recovery_to_episode_v1 import kinematic_window


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['bundle','sphere-audit','geometry','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--case',type=int,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve evidence')
    sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    m=json.loads((a.bundle/'manifest.json').read_text());sphere=json.loads(a.sphere_audit.read_text())
    archive=a.bundle/m['bundle_name'];key=f'case_{a.case:03d}';row=next(r for r in m['rows'] if r['case']==a.case)
    if sha(archive)!=m['bundle_sha256'] or sha(a.geometry)!=m['ground_geometry_sha256']:
        raise ValueError('Changed recovery plan or geometry')
    for f,digest in sphere['source_sha256'].items():
        if sha(f)!=digest:raise ValueError('Changed sphere asset evidence '+f)
    c=build_controller(m['configuration']);model=c.qp.model;interface=c.qp.interface
    geometry=AuthoredGroundClearance(a.geometry,model,margin_m=0.)
    with np.load(archive) as z:
        commands=z[key+'_commands_rad'].copy();expected_q=z[key+'_predicted_q'].copy()
        state=model.state_from_public(z[key+'_query_public'],np.asarray(m['configuration']['posture_target']))
        prior=interface.limits[:,0]+.5*(z[key+'_previous_action']+1)*np.diff(interface.limits).ravel()
    state.previous_command_target=prior.copy();states=[];records=[];failure=None
    for i,command in enumerate(commands):
        if np.max(abs(command[12:]-prior[12:]))>interface.arm_rate*.01+1e-6:
            raise ValueError('Original plan violates arm command rate')
        try:
            result=predict_frictional_spheres(model,interface,state,command,sphere['spheres'],geometry=geometry)
            following=result['state'];following.previous_command_target=command.copy()
            diagnostic=result['diagnostics']
            violates=diagnostic['nonfoot_ground_violation_m']>1e-6 or any(diagnostic[k]>3e-4 for k in
                ['joint_position_violation_rad','joint_velocity_violation_rad_s','motor_torque_violation_nm'])
            record=dict(tick=row['query_tick']+i,diagnostics=diagnostic,events=result['events'],
                first_ground_violation=result['first_ground_violation'],
                maximum_joint_difference_from_old_plan_rad=float(np.max(abs(following.joint_position-expected_q[i+1,model.q_indices]))),
                root_position_difference_from_old_plan_m=float(np.linalg.norm(following.q[:3]-expected_q[i+1,:3])),
                constraint_violation=bool(violates))
            records.append(record);states.append(copy.deepcopy(following));state=following;prior=command.copy()
            if violates:failure=dict(tick=record['tick'],reason='Predicted original constraint violation');break
        except (ValueError,np.linalg.LinAlgError) as error:
            failure=dict(tick=row['query_tick']+i,reason=repr(error));break
    protocol_path=ROOT/'configs/evaluation_protocol_v2.json'
    protocol=json.loads(protocol_path.read_text())['recovery']
    window=kinematic_window(model,states,protocol,.01) if states else None
    files=[Path(__file__),a.bundle/'manifest.json',archive,a.sphere_audit,a.geometry,protocol_path,
        ROOT/'scripts/predict_frictional_spheres_v1.py',ROOT/'scripts/frictional_impulse_step_v1.py']
    result=dict(scope=__doc__,case=a.case,start_tick=row['query_tick'],requested_steps=len(commands),
        predicted_steps=len(records),failure=failure,whole_sequence_valid_in_new_model=failure is None and len(records)==len(commands),
        kinematic_window=window,native_execution=False,valid_expert_labels=False,records=records,
        source_sha256={str(f.resolve()):sha(f) for f in files})
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['records','source_sha256']}))


if __name__=='__main__':main()
