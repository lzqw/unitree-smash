"""Reconstruct actual and commanded faces at saved native contact instants."""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from quadsmash.control.full_racket_kinematics_v4 import FullRacketKinematicsV4

ROOT = Path.cwd()
SOURCE = ROOT / 'reports/G07_full_racket_v4/normal38_native_tick400_snapshot'
OUT = Path(__file__).parent
assert not (OUT / 'REPORT.json').exists()
events = [row for row in json.loads((SOURCE / 'contacts.partial.json').read_text()) if row['episode_index'] == 0]
state = torch.load(SOURCE / 'partial_search_state.pt', map_location='cpu', weights_only=False)
warm_path = ROOT / 'reports/G07_full_racket_v4/attack_recovered_normal38_ready03_1415/policy.pt'
warm = torch.load(warm_path, map_location='cpu', weights_only=False)
reference_normal = np.array(warm['seed']['outward_contact_normal_world'])
kin = FullRacketKinematicsV4().double()
offset = torch.tensor([.0256, 0., 0.], dtype=torch.float64)
rows = []
for event in events:
    cid = event['env_id']//2
    params = state['parameters'][cid].numpy()
    q = np.array([event['joint_position_rad'][-6:], event['commanded_joint_position_rad'][-6:]])
    root = np.array(event['body_link_poses_local'][0])
    rotation = Rotation.from_quat(root[[4, 5, 6, 3]]).as_matrix()
    with torch.inference_mode():
        point, normal, *_ = kin(torch.tensor(q), offset)
    points = point.numpy() @ rotation.T + root[:3]
    normals = normal.numpy() @ rotation.T
    measured = np.array(event['face_normal'])
    reconstruction_error = np.linalg.norm(normals[0]-measured)
    assert reconstruction_error < 1e-4, reconstruction_error
    angle = float(params[37]);cosine, sine = np.cos(angle), np.sin(angle)
    desired = np.array([cosine*reference_normal[0]+sine*reference_normal[2], reference_normal[1],
                        -sine*reference_normal[0]+cosine*reference_normal[2]])
    degrees = lambda a, b: float(np.degrees(np.arccos(np.clip(a@b/(np.linalg.norm(a)*np.linalg.norm(b)), -1, 1))))
    pitch = lambda n: float(np.degrees(np.arctan2(n[2], n[0])))
    rows.append(dict(env_id=event['env_id'], candidate_id=cid, source_training_index=1 if event['env_id']%2==0 else 65,
        string_contact=event['string_contact'], frame_contact=event['frame_contact'], time_s=event['time_s'],
        normal_downward_bias_rad=angle, velocity_gain=float(params[7]), arm_lead_s=float(params[14]),
        actual_normal_world=measured.tolist(), commanded_normal_at_actual_root_world=normals[1].tolist(),
        desired_normal_world=desired.tolist(), actual_face_pitch_deg=pitch(measured),
        commanded_face_pitch_deg=pitch(normals[1]), desired_face_pitch_deg=pitch(desired),
        actual_to_commanded_normal_error_deg=degrees(measured, normals[1]),
        actual_to_desired_normal_error_deg=degrees(measured, desired),
        maximum_arm_joint_tracking_error_rad=float(abs(q[0]-q[1]).max()),
        actual_to_commanded_point_displacement_m=float(np.linalg.norm(points[1]-points[0])),
        native_normal_reconstruction_error=reconstruction_error))
keys=['actual_to_commanded_normal_error_deg', 'actual_to_desired_normal_error_deg',
      'maximum_arm_joint_tracking_error_rad', 'actual_to_commanded_point_displacement_m',
      'actual_face_pitch_deg', 'commanded_face_pitch_deg', 'desired_face_pitch_deg']
groups={name:[row for row in rows if name=='all' or row['string_contact']] for name in ['all', 'strings']}
summary={name:{key:dict(min=float(min(r[key] for r in group)), median=float(np.median([r[key] for r in group])),
                        max=float(max(r[key] for r in group))) for key in keys} for name,group in groups.items()}
paths=[SOURCE/'contacts.partial.json', SOURCE/'partial_search_state.pt', warm_path, Path(__file__),
       ROOT/'quadsmash/control/full_racket_kinematics_v4.py']
report=dict(created_at=datetime.now(timezone.utc).isoformat(),
    scope='Kinematic reconstruction at85actual native contact instants from the400tick snapshot; includes frame events. Commanded face uses actual body pose with commanded joints and is not a counterfactual robot rollout, achieved face or outgoing velocity.',
    contacts=len(rows), string_contacts=len(groups['strings']),
    maximum_native_normal_reconstruction_error=max(row['native_normal_reconstruction_error'] for row in rows),
    statistics=summary, native_contact_sources=True, new_controller_executed=False,
    interpretation='A requested world face target, its joint command and the achieved contact face are distinct. Tracking/dynamic effects must be measured; the commanded tilt cannot be reported as achieved ball steepness.',
    formal_acceptance=False, sha256={str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
(OUT/'REPORT.json').write_text(json.dumps(report, indent=2)+'\n')
(OUT/'ROWS.json').write_text(json.dumps(rows, indent=2)+'\n')
with (OUT/'ROWS.csv').open('w') as stream:
    writer=csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n');writer.writeheader();writer.writerows(rows)
print(json.dumps({key:value for key,value in report.items() if key!='sha256'}, indent=2))
