"""Corroborate a real complete native TRAIN episode and retained failures."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from quadsmash.evaluation_v4 import independent_episode
from quadsmash.rl.episode_trace_v4 import unpack_public


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = HERE/'FIRST_COMPLETE_NATIVE_EPISODE_CHECK.json'
    assert not output.exists()
    snapshot = ROOT/'reports/G07_full_racket_v4/video_snapshot_placement075_first_complete_20261026'
    manifest = read(snapshot/'cases.json')
    assert manifest['source_job_complete'] is False and manifest['includes_failure']
    assert [(case['env_id'],case['episode_index']) for case in manifest['cases']] == [(208,4),(23,2),(8,3)]
    cases = manifest['cases']; case = cases[0]; row, event = case['row'], case['contact']
    protocol_path = ROOT/'configs/g07_full_racket_eval_v4_r3.json'
    protocol = read(protocol_path); protocol['_root'] = str(ROOT)
    outcomes = [independent_episode(value['row'],protocol) for value in cases]
    assert [value['complete'] for value in outcomes] == [True,False,False]
    directory = snapshot/case['run']
    observed_path = directory/'observed_episodes.json'
    assert sha(observed_path) == manifest['observed_raw_publication_sha256']
    observed = read(observed_path)
    actual = [independent_episode(value,protocol) for value in observed]
    counts = dict(closed_episodes=len(observed),air=sum(value['airborne_contact']for value in actual),
        conditional_recovery=sum(value['conditional_recovery']for value in actual),
        complete=sum(value['complete']for value in actual),numerical_failures=sum(value['numerical_failures']for value in observed))
    assert counts['complete'] >= 1 and counts['numerical_failures'] == 0
    speed = float(np.linalg.norm(row['outgoing_velocity_mps']))
    assert speed >= 8. and row['outgoing_velocity_mps'][0] > 0 and row['outgoing_velocity_mps'][2] <= -1.
    assert row['ground_init'] and row['active_rally_launch_count'] == row['contact_count'] == 1
    assert event['string_contact'] and not event['frame_contact'] and not event['fixture_contact']
    assert event['legal_initial_contact'] and not any(event['fault_flags'].values())
    names = read(directory/'geometry.json')['body_names']
    poses = np.asarray(event['body_link_poses_local'])
    feet = {name:float(poses[names.index(name),2]) for name in ['FL_foot','FR_foot','RL_foot','RR_foot']}
    assert min(feet.values()) > .035
    racket = poses[names.index('Racket')]
    point_local = Rotation.from_quat(racket[3:][[1,2,3,0]]).inv().apply(np.asarray(event['point_m'])-racket[:3])
    semiaxes = np.array([.0915,.1165])
    rho_squared = float(np.sum((point_local[1:]/semiaxes)**2))
    assert rho_squared <= 1.
    measured_velocity = np.asarray(event['point_velocity_mps'])
    recomputed_velocity = np.asarray(event['flange_linear_velocity_mps']) + np.cross(
        event['flange_angular_velocity_rad_s'],np.asarray(event['point_m'])-event['flange_position_m'])
    velocity_error = float(np.linalg.norm(measured_velocity-recomputed_velocity))
    # Report the fixed-joint/native float32 reconstruction residual, rather than
    # converting an earlier capture's extra diagnostic tolerance into acceptance.
    precision = np.finfo(np.float32).eps * 8 * max(1.,np.linalg.norm(measured_velocity),np.linalg.norm(recomputed_velocity))
    assert abs(velocity_error-row['hit_velocity_error_mps']) <= precision
    data_path = directory/'lane0208_episode000004.npz'
    with np.load(data_path) as data:
        public = unpack_public(data['packed_public']); all_poses = data['link_poses']
        assert public.shape == (800,794)
        initial_root_error = float(np.max(abs(all_poses[0,0,:3]-protocol['initialization']['root_position_m'])))
        assert initial_root_error <= 2e-5
        assert np.max(abs(all_poses[0,0,3:]-[1.,0.,0.,0.])) <= 1e-5
        assert abs(public[0,775]) < 1e-6 and np.max(abs(np.diff(public[:,775])-.01)) < 2e-6
        mask = public[:,775] >= 7.2
        q = all_poses[mask,0,3:][:,[1,2,3,0]]
        angles = Rotation.from_quat(q).as_euler('xyz',degrees=True)
        # These are100Hz public root-LINK velocities and pose corroboration;
        # native62.5us COM/force/recovery tracking remains authoritative.
        corroboration = dict(samples=int(mask.sum()),start_time_s=float(public[mask,775][0]),
            maximum_abs_roll_pitch_degrees=float(np.max(abs(angles[:,:2]))),
            maximum_root_link_horizontal_speed_mps=float(np.max(np.linalg.norm(public[mask,:2],axis=1))),
            maximum_root_angular_speed_rad_s=float(np.max(np.linalg.norm(public[mask,3:6],axis=1))),
            maximum_abs_joint_speed_rad_s=float(np.max(abs(public[mask,27:45]))),
            minimum_supporting_feet=int(np.min(np.sum(public[mask,57:61]>.5,axis=1))))
        assert corroboration['maximum_abs_roll_pitch_degrees'] <= 15.
        assert corroboration['maximum_root_link_horizontal_speed_mps'] <= .3
        assert corroboration['maximum_root_angular_speed_rad_s'] <= 1.
        assert corroboration['maximum_abs_joint_speed_rad_s'] <= 1.
        assert corroboration['minimum_supporting_feet'] > 0
    sources = [Path(__file__),ROOT/'reports/G07_full_racket_v4/centered_resume_training_capture_20261026/prepare.py',snapshot/'cases.json',observed_path,
        directory/'observed_trace_manifest.json',data_path,data_path.with_suffix('.json'),protocol_path,
        ROOT/'quadsmash/tasks/stand/observation_contract.py',ROOT/'quadsmash/evaluation_v4.py']
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(),
        scope='First complete episode of this075 strict8-to40 TRAIN continuation, not the first historical project success, a completed job, a frozen-policy score or formal acceptance.',
        actual_case_id=row['case_id'],env_id=row['env_id'],episode_index=row['episode_index'],TRAIN=row['source_training_index'],
        all800_actual100Hz_ground_start_samples=True,initial_root_position_max_error_m=initial_root_error,
        actor_iterations_encountered=case['actor_iterations_encountered'],actual_independent_outcome=outcomes[0],
        actual_contact_time_s=row['contact_time_s'],actual_contact_point_m=row['contact_point_m'],
        actual_contact_point_local_racket_m=point_local.tolist(),actual_contact_point_strings_rho_squared=rho_squared,
        actual_feet_heights_at_contact_m=feet,actual_outgoing_velocity_mps=row['outgoing_velocity_mps'],
        actual_outgoing_speed_mps=speed,actual_speed_threshold_margin_mps=speed-8.,
        actual_net_crossing_point_m=row['crossing_point_m'],actual_net_margin_m=row['net_margin_m'],
        actual_first_landing_m=row['first_landing_m'],actual_first_landing_time_s=row['first_landing_time_s'],
        actual_target_far_x_margin_m=4.-row['first_landing_m'][0],actual_recovery_s=row['jump']['recovery_s'],
        actual_body_ground_maximum_force_n=row['jump']['maximum_body_ground_force_n'],
        actual_point_velocity_mps=measured_velocity.tolist(),actual_point_speed_mps=float(np.linalg.norm(measured_velocity)),
        flange_plus_rotation_point_velocity_error_mps=velocity_error,
        recorded_native_point_velocity_consistency_error_mps=row['hit_velocity_error_mps'],
        diagnostic_residual_agreement_bound_from_float32_mps=float(precision),
        below_prior_capture_extra_diagnostic_bound_2e5=bool(velocity_error <= 2e-5),
        no_original_acceptance_tolerance_changed=True,
        last_080s_public_pose_and_velocity_corroboration=corroboration,
        all_observed_closed_changing_policy_counts=counts,retained_failure_keys=[[23,2],[8,3]],
        limitations=['Target-edge margin remains about1cm; this episode does not establish robust performance.',
            'The actual racket-point speed is reported separately from outgoing ball speed; an8m/s outgoing ball is not an8m/s racket swing.',
            'No restoring of in-contact PhysX state, no generated motion, no episode splice, and no new frozen score.',
            'The native recovery tracker checks substeps;100Hz root-link corroboration is additional evidence, not a substitute for it.'],
        source_sha256={str(path.relative_to(ROOT)):sha(path) for path in sources},
        native_source_job_still_running=True,frozen_policy_performance=False,formal_cases_read=False,formal_acceptance=False,goal_complete=False)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ['actual_case_id','actual_outgoing_speed_mps','actual_target_far_x_margin_m',
        'actual_recovery_s','all_observed_closed_changing_policy_counts','last_080s_public_pose_and_velocity_corroboration']},indent=2))


if __name__ == '__main__':
    main()
