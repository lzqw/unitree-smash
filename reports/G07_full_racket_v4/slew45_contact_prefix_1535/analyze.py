"""Check applied slew and measured faces in a frozen native contact prefix."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from quadsmash.control.full_racket_kinematics_v4 import FullRacketKinematicsV4

OUT=Path(__file__).parent;SOURCE=OUT/'native_prefix'
assert not (OUT/'REPORT.json').exists()
progress=json.loads((SOURCE/'progress.json').read_text())
state=torch.load(SOURCE/'partial_search_state.pt',map_location='cpu',weights_only=False)
start=torch.load(SOURCE/'generation_start_000.pt',map_location='cpu',weights_only=False)
ticks=progress['control_ticks'];assert ticks==state['control_ticks']==400
assert progress['generation']==state['generation']==0
with np.load(SOURCE/'arm_targets.partial.npz') as data:trace={k:data[k].copy() for k in data.files}
assert all(np.isfinite(v).all() for v in trace.values())
raw=trace['targets_before_slew_rad'];limited=trace['targets_after_slew_rad']
assert raw.shape==limited.shape==(ticks,128,6)
valid=trace['valid_first_episode'];dt=np.diff(trace['clock_s'],axis=0)
mask=valid[1:]&valid[:-1]&(dt>0)
before_rate=abs(np.diff(raw,axis=0)/np.where(dt>0,dt,1)[:,:,None])[mask]
after_rate=abs(np.diff(limited,axis=0)/np.where(dt>0,dt,1)[:,:,None])[mask]
assert after_rate.max()<=4.5+1e-4
for name in ['all_first_episodes.partial.npz','partial_public_and_poses.npz',
             'near_contact_all.partial.npz','control_window.partial.npz']:
    with np.load(SOURCE/name) as data:
        assert all(np.isfinite(data[k]).all() for k in data.files)
        if 'actions' in data.files:assert abs(data['actions']).max()<=1
events=[e for e in json.loads((SOURCE/'contacts.partial.json').read_text()) if e['episode_index']==0]
rows=[r for r in json.loads((SOURCE/'episodes.partial.json').read_text()) if r['episode_index']==0]
assert len(rows)==int(state['first_finished'].sum())==progress['first_episodes_complete']
kin=FullRacketKinematicsV4().double()
normal_reference=np.array(start['controller_template']['seed']['outward_contact_normal_world'])
contact_offset=torch.tensor([.0256,0.,.1],dtype=torch.float64)
checked=[]
for event in events:
    env=event['env_id'];candidate=env//2
    target=np.array(event['commanded_joint_position_rad'][-6:])
    possible=np.where(valid[:,env])[0]
    errors=abs(limited[possible,env]-target).max(1)
    assert errors.min()<1e-5
    matched=possible[errors<1e-5]
    tick=int(matched[np.argmin(abs(trace['clock_s'][matched,env]-event['time_s']))])
    q=np.array([event['joint_position_rad'][-6:],target,raw[tick,env]])
    with torch.inference_mode():_,normals,*_=kin(torch.tensor(q,dtype=torch.float64),contact_offset)
    root=np.array(event['body_link_poses_local'][0])
    normals=normals.numpy()@Rotation.from_quat(root[[4,5,6,3]]).as_matrix().T
    actual=np.array(event['face_normal']);error=float(np.linalg.norm(normals[0]-actual));assert error<1e-4
    angle=float(state['parameters'][candidate,37]);c,s=np.cos(angle),np.sin(angle)
    desired=np.array([c*normal_reference[0]+s*normal_reference[2],normal_reference[1],
                      -s*normal_reference[0]+c*normal_reference[2]])
    degrees=lambda a,b:float(np.degrees(np.arccos(np.clip(a@b/(np.linalg.norm(a)*np.linalg.norm(b)),-1,1))))
    checked.append(dict(env_id=env,candidate_id=candidate,string_contact=event['string_contact'],
        time_s=event['time_s'],matched_control_tick=tick,
        limited_target_to_native_command_error_rad=float(abs(limited[tick,env]-target).max()),
        actual_normal_reconstruction_error=error,
        actual_to_after_slew_command_face_error_deg=degrees(actual,normals[1]),
        actual_to_before_slew_command_face_error_deg=degrees(actual,normals[2]),
        actual_to_desired_face_error_deg=degrees(actual,desired),
        actual_face_pitch_deg=float(np.degrees(np.arctan2(actual[2],actual[0])))))
strings=[e for e in checked if e['string_contact']]
assert strings
columns=['actual_to_after_slew_command_face_error_deg','actual_to_before_slew_command_face_error_deg',
         'actual_to_desired_face_error_deg','actual_face_pitch_deg']
string_ids={e['env_id'] for e in strings};closed_ids={r['env_id'] for r in rows}
closed={k:sum(bool(r[k]) for r in rows) for k in ['airborne_string_hit','recovery_success',
    'crossed_net','landed_legal_court','landed_target','complete_smash','robot_net_fault','racket_invasion_fault']}
closed['numerical_failures']=sum(r['numerical_failures'] for r in rows)
actual_velocity=abs(trace['observed_velocity_rad_s'])[valid]
report=dict(verified_at=datetime.now(timezone.utc).isoformat(),
    scope='Actual CEM22generation0first400ticks only.36closed92pending. Every contact command is matched to the native pre/post-slew target log; actual goal-face error is kept separate from closeness to the limited command. Not a complete generation or formal score.',
    generation=0,control_ticks=ticks,closed_first_episodes=len(rows),pending_first_episodes=128-len(rows),closed_counts=closed,
    contact_events=len(events),string_contact_events=len(strings),pending_string_contact_episodes=len(string_ids-closed_ids),
    software_target_rate_rad_s=4.5,logged_after_slew_command_rate_max_rad_s=float(after_rate.max()),
    logged_before_slew_command_rate_max_rad_s=float(before_rate.max()),
    maximum_target_adjustment_rad=float(abs(limited-raw)[valid].max()),
    measured100Hz_arm_velocity_peak_rad_s=float(actual_velocity.max()),
    measured_samples_above5_plus1e_4_rad_s=int((actual_velocity>5.0001).sum()),
    velocity_scope='Valid first-episode100Hz samples, not continuous physics-substep maxima; physical observations retained unmodified.',
    all79_contact_target_records_match_native_commands=True,
    maximum_contact_command_match_error_rad=max(e['limited_target_to_native_command_error_rad'] for e in checked),
    maximum_native_normal_reconstruction_error=max(e['actual_normal_reconstruction_error'] for e in checked),
    string_face_statistics={key:dict(min=min(e[key] for e in strings),
        median=float(np.median([e[key] for e in strings])),max=max(e[key] for e in strings)) for key in columns},
    comparison_caution='Candidate timing/normal grids differ from CEM21. Aggregate medians alone cannot isolate the slew change, and smaller command-face error is not proof of better outgoing balls or recovery.',
    native_execution=True,formal_acceptance=False,
    sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [*sorted(SOURCE.iterdir()),Path(__file__),ROOT/'quadsmash/control/full_racket_kinematics_v4.py']})
(OUT/'ROWS.json').write_text(json.dumps(checked,indent=2)+'\n')
(OUT/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='sha256'},indent=2))
