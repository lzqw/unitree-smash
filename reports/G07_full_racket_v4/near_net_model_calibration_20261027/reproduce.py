"""Recompute native-contact FK/velocity checks; no physics or policy rollout."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from quadsmash.control.full_racket_geometry_v4 import FullRacketGeometryV4


def main(output):
    assert not output.exists()
    run=ROOT/'reports/G07_full_racket_v4/body_strike_grid16_fixed_train128_01'
    source=run/'contacts.json'
    events=json.loads(source.read_text())
    model=FullRacketGeometryV4()
    checks=[]
    for event in events:
        poses=np.asarray(event['body_link_poses_local'])
        rotations=Rotation.from_quat(poses[:,3:][:,[1,2,3,0]]).as_matrix()
        q=np.asarray(event['joint_position_rad'])[12:]
        qd=np.asarray(event['joint_velocity_rad_s'])[12:]
        kwargs=dict(base_position=poses[0,:3],base_rpy=Rotation.from_matrix(rotations[0]).as_euler('xyz'))
        point,rotation=model.forward(q,**kwargs)
        hit=np.asarray(event['point_m'])
        local=rotations[-1].T@(hit-poses[-1,:3])
        columns=[]
        for axis in np.eye(6):
            plus,rplus=model.forward(q+axis*1e-6,**kwargs)
            minus,rminus=model.forward(q-axis*1e-6,**kwargs)
            columns.append((plus+rplus@local-minus-rminus@local)/2e-6)
        jacobian=np.stack(columns,axis=1)
        body=np.asarray(event['root_com_linear_velocity_mps'])+np.cross(
            event['root_angular_velocity_rad_s'],hit-np.asarray(event['root_com_position_m']))
        predicted=body+jacobian@qd
        flange=np.asarray(event['flange_linear_velocity_mps'])+np.cross(
            event['flange_angular_velocity_rad_s'],hit-np.asarray(event['flange_position_m']))
        checks.append(dict(env_id=event['env_id'],event_index=event['event_index'],
            FK_center_error_m=float(np.linalg.norm(point-poses[-1,:3])),
            FK_rotation_max_error=float(abs(rotation-rotations[-1]).max()),
            joint_and_body_Jacobian_velocity_error_mps=float(np.linalg.norm(predicted-event['point_velocity_mps'])),
            flange_rigid_velocity_error_mps=float(np.linalg.norm(flange-event['point_velocity_mps']))))
    maximum={key:max(item[key] for item in checks) for key in
        ['FK_center_error_m','FK_rotation_max_error','joint_and_body_Jacobian_velocity_error_mps','flange_rigid_velocity_error_mps']}
    result=dict(checked_at=datetime.now(timezone.utc).isoformat(),actual_native_contacts=len(events),
        max_errors=maximum,source_contacts_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        scope='Previously recorded real native contacts, independent URDF FK/Jacobian and rigid velocity formula only.',
        new_native_execution=False,formal_cases_read=False,formal_acceptance=False)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(maximum),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
