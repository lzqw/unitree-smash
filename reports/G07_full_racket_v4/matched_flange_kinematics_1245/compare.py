from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
from scipy.spatial.transform import Rotation
root=Path(__file__).resolve().parents[3];base=root/'reports/G07_full_racket_v4'
old_path=root/'configs/assembly_v2.json';new_path=root/'configs/full_racket_v4_r3.json'
old=json.loads(old_path.read_text())['racket'];new=json.loads(new_path.read_text());mount=json.loads(Path(__file__).with_name('LEGACY_MOUNT.json').read_text())
old_offset=np.array(old['blade_center_relative_to_link6_m']);new_offset=np.array(new['string_center_in_flange_m'])
np.testing.assert_allclose(old_offset,mount['local_position0_m'],atol=1e-7,rtol=0)
records=[];sources=[old_path,new_path,Path(__file__),Path(__file__).with_name('LEGACY_MOUNT.json')]
for name,env in [('ball_near_intercept_blend_cem16',35),('ball_attack_recovery_cem17',53)]:
    run=base/name;gp=run/'geometry.json';cp=run/'contacts_000.json';geometry=json.loads(gp.read_text());event=next(c for c in json.loads(cp.read_text()) if c['env_id']==env and c['first_racket_contact'])
    poses=np.array(event['body_link_poses_local']);idx=geometry['body_names'].index('link6');tool=geometry['body_names'].index('Racket')
    flange=np.array(event['flange_position_m']);omega=np.array(event['flange_angular_velocity_rad_s']);velocity=np.array(event['flange_linear_velocity_mps'])
    rot=Rotation.from_quat(poses[idx,[4,5,6,3]]).as_matrix();old_lever=rot@old_offset;new_lever=rot@new_offset
    old_center=flange+old_lever;new_center=flange+new_lever
    new_error=float(np.linalg.norm(new_center-poses[tool,:3]));normal=rot@np.array(new['face_normal_in_flange']);normal_error=float(np.linalg.norm(normal-event['face_normal']))
    hit_velocity=velocity+np.cross(omega,np.array(event['point_m'])-flange);hit_error=float(np.linalg.norm(hit_velocity-event['point_velocity_mps']))
    assert new_error<2e-5 and normal_error<2e-6 and hit_error<1e-4
    old_velocity=velocity+np.cross(omega,old_lever);new_velocity=velocity+np.cross(omega,new_lever)
    records.append(dict(run=name,env_id=env,time_s=event['time_s'],actual_flange_position_m=flange.tolist(),actual_flange_velocity_mps=velocity.tolist(),actual_flange_angular_velocity_rad_s=omega.tolist(),counterfactual_shortboard_center_m=old_center.tolist(),reconstructed_full_racket_center_m=new_center.tolist(),same_flange_state_center_height_gain_m=float(new_center[2]-old_center[2]),counterfactual_shortboard_center_velocity_mps=old_velocity.tolist(),full_racket_center_velocity_mps=new_velocity.tolist(),counterfactual_shortboard_center_speed_mps=float(np.linalg.norm(old_velocity)),full_racket_center_speed_mps=float(np.linalg.norm(new_velocity)),same_flange_state_forward_center_velocity_gain_mps=float(new_velocity[0]-old_velocity[0]),counterfactual_old_blade_normal_world=(rot@np.array([1.,0.,0.])).tolist(),actual_full_racket_normal_world=normal.tolist(),actual_strings_contact_point_m=event['point_m'],actual_strings_contact_point_velocity_mps=event['point_velocity_mps'],new_center_native_position_error_m=new_error,new_normal_native_error=normal_error,actual_hit_velocity_identity_error_mps=hit_error))
    sources.extend([gp,cp])
report=dict(created_at=datetime.now(timezone.utc).isoformat(),scope='Matched observed flange pose/linear/angular velocity only. The old shortboard is reconstructed kinematically from its verified legacy USD mount; no old-body dynamics, legal strike, global maximum reach or recovery performance is inferred. Old and new face normals differ at these poses.',old_shortboard_center_in_flange_m=old_offset.tolist(),full_racket_string_center_in_flange_m=new_offset.tolist(),old_payload_mass_kg=old['mass_kg'],full_racket_and_fixture_mass_kg=sum(new['assumptions'][k] for k in ['unstrung_mass_kg','strings_mass_kg','added_grip_mass_kg','clamp_mass_kg']),records=records,old_shortboard_native_rollout_performed=False,formal_acceptance=False,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
Path(__file__).with_name('REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps([dict(env_id=r['env_id'],center_height_gain_m=r['same_flange_state_center_height_gain_m'],old_center_speed=r['counterfactual_shortboard_center_speed_mps'],new_center_speed=r['full_racket_center_speed_mps'],new_center_error_m=r['new_center_native_position_error_m'],hit_velocity_error_mps=r['actual_hit_velocity_identity_error_mps']) for r in records]))
