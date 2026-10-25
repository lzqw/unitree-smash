"""Audit the second saved complete TRAIN motion and decode its actual video."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image
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
    snapshot = ROOT/'reports/G07_full_racket_v4/video_snapshot_centered_second_complete_20261026'
    video = ROOT/'reports/G07_full_racket_v4/video/Current_training_centered_second_complete_and_failure_20261026.mp4'
    output = HERE/'SECOND_COMPLETE_NATIVE_AND_VIDEO_CHECK.json'
    samples_dir = HERE/'decoded_samples'
    assert not output.exists() and not samples_dir.exists()
    manifest, metadata = read(snapshot/'cases.json'), read(video.with_suffix('.json'))
    assert not manifest['source_job_complete'] and manifest['includes_failure']
    cases = manifest['cases']
    assert [(case['env_id'], case['episode_index']) for case in cases] == [(153, 5), (8, 3)]
    protocol_path = ROOT/'configs/g07_full_racket_eval_v4_r3.json'
    protocol = read(protocol_path)
    protocol['_root'] = str(ROOT)
    outcomes = [independent_episode(case['row'], protocol) for case in cases]
    assert [outcome['complete'] for outcome in outcomes] == [True, False]
    case = cases[0]
    row, contact = case['row'], case['contact']
    assert row['source_training_index'] == 25 and row['ground_init']
    assert row['contact_count'] == row['active_rally_launch_count'] == 1
    assert contact['string_contact'] and contact['legal_initial_contact']
    assert not contact['frame_contact'] and not contact['fixture_contact']
    assert not any(contact['fault_flags'].values())
    directory = snapshot/case['run']
    observed_path = directory/'observed_episodes.json'
    assert sha(observed_path) == manifest['observed_raw_publication_sha256']
    observed = read(observed_path)
    matching = [value for value in observed if (value['env_id'], value['episode_index']) == (153, 5)]
    assert len(matching) == 1 and matching[0] == row
    counts = dict(closed_episodes=len(observed), complete=sum(independent_episode(value, protocol)['complete'] for value in observed),
                  numerical_failures=sum(value['numerical_failures'] for value in observed))
    assert counts['complete'] >= 2 and counts['numerical_failures'] == 0
    names = read(directory/'geometry.json')['body_names']
    poses = np.asarray(contact['body_link_poses_local'])
    racket = poses[names.index('Racket')]
    local = Rotation.from_quat(racket[3:][[1, 2, 3, 0]]).inv().apply(np.asarray(contact['point_m'])-racket[:3])
    rho_squared = float(np.sum((local[1:]/[.0915, .1165])**2))
    assert rho_squared <= 1.
    feet = {name: float(poses[names.index(name), 2]) for name in ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']}
    assert min(feet.values()) > .035
    point_velocity = np.asarray(contact['point_velocity_mps'])
    recomputed = np.asarray(contact['flange_linear_velocity_mps']) + np.cross(
        contact['flange_angular_velocity_rad_s'], np.asarray(contact['point_m'])-contact['flange_position_m'])
    velocity_error = float(np.linalg.norm(point_velocity-recomputed))
    assert velocity_error <= 2e-5
    data_path = directory/'lane0153_episode000005.npz'
    trace_metadata = read(data_path.with_suffix('.json'))
    assert sha(data_path) == trace_metadata['data_sha256']
    with np.load(data_path) as data:
        public = unpack_public(data['packed_public'])
        assert public.shape == (800, 794) and np.isfinite(public).all()
        assert hashlib.sha256(public.tobytes()).hexdigest() == trace_metadata['full_public794_sha256']
        assert abs(public[0, 775]) < 1e-6 and np.max(abs(np.diff(public[:, 775])-.01)) < 2e-6
        initial_error = float(np.max(abs(data['link_poses'][0, 0, :3]-protocol['initialization']['root_position_m'])))
        assert initial_error <= 2e-5
        mask = public[:, 775] >= 7.2
        angles = Rotation.from_quat(data['link_poses'][mask, 0, 3:][:, [1, 2, 3, 0]]).as_euler('xyz', degrees=True)
        recovery = dict(samples=int(mask.sum()), maximum_abs_roll_pitch_deg=float(np.max(abs(angles[:, :2]))),
                        maximum_root_link_horizontal_speed_mps=float(np.max(np.linalg.norm(public[mask, :2], axis=1))),
                        maximum_root_angular_speed_rad_s=float(np.max(np.linalg.norm(public[mask, 3:6], axis=1))),
                        maximum_abs_joint_speed_rad_s=float(np.max(abs(public[mask, 27:45]))),
                        minimum_supporting_feet=int(np.min(np.sum(public[mask, 57:61] > .5, axis=1))))
        assert recovery['maximum_abs_roll_pitch_deg'] <= 15.
        assert recovery['maximum_root_link_horizontal_speed_mps'] <= .3
        assert recovery['maximum_root_angular_speed_rad_s'] <= 1.
        assert recovery['maximum_abs_joint_speed_rad_s'] <= 1.
        assert recovery['minimum_supporting_feet'] > 0
    assert sha(video) == metadata['sha256']
    assert metadata['native_recorded_motion'] and not metadata['interpolated_or_generated_motion']
    assert not metadata['renderer_uses_gpu'] and 'llvmpipe' in metadata['renderer'].lower()
    selected = {25}
    for segment in metadata['segments']:
        if segment['name'] in ['whole_episode', 'exact_contact', 'first_landing', 'native_terminal']:
            selected.add(segment['start_frame']+segment['frames']//2)
    selected = sorted(selected)
    samples_dir.mkdir()
    saved, decoded = [], 0
    with imageio.get_reader(video) as reader:
        assert abs(reader.get_meta_data()['fps']-50.) < 1e-9
        for index, frame in enumerate(reader):
            assert frame.shape == (720, 1280, 3) and frame.dtype == np.uint8
            if index in selected:
                path = samples_dir/f'frame_{index:06d}.jpg'
                Image.fromarray(frame).save(path, quality=94)
                saved.append(dict(frame_index=index, video_time_s=index/50., path=str(path.relative_to(ROOT)), sha256=sha(path)))
            decoded += 1
    assert decoded == metadata['frames'] and len(saved) == len(selected)
    speed = float(np.linalg.norm(row['outgoing_velocity_mps']))
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), scope='Second saved complete changing-policy TRAIN episode of this continuation plus an actual body-ground failure; not frozen-policy performance.',
                  actual_case_id=row['case_id'], actual_independent_outcome=outcomes[0], all800_actual100Hz_ground_start_samples=True,
                  actor_iterations_encountered=case['actor_iterations_encountered'], initial_root_position_max_error_m=initial_error,
                  actual_contact_time_s=row['contact_time_s'], actual_contact_point_m=row['contact_point_m'], actual_contact_strings_rho_squared=rho_squared,
                  actual_feet_heights_at_contact_m=feet, actual_outgoing_velocity_mps=row['outgoing_velocity_mps'], actual_outgoing_speed_mps=speed,
                  actual_speed_margin_mps=speed-8., actual_first_landing_m=row['first_landing_m'], actual_target_far_x_margin_m=4.-row['first_landing_m'][0],
                  actual_crossing_point_m=row['crossing_point_m'], conservative_net_margin_m=row['net_margin_m'], actual_shuttle_net_contact=row['shuttle_net_contact'],
                  actual_recovery_s=row['jump']['recovery_s'], actual_point_speed_mps=float(np.linalg.norm(point_velocity)),
                  flange_plus_rotation_point_velocity_error_mps=velocity_error, last080s_public_corroboration=recovery,
                  observed_changing_policy_counts=counts, retained_failure_key=[8, 3], video=str(video.relative_to(ROOT)), sha256=sha(video),
                  bytes=video.stat().st_size, decoded_frames=decoded, all_frames_decoded=True, duration_s=decoded/50., samples=saved,
                  event_previews=metadata['previews'], human_visual_review_is_separate=True,
                  limitations=['Video selections occur after outcomes; one displayed success is not an estimated success rate.',
                               'TRAIN25 has a small speed margin. Recorded outgoing ball speed differs from racket-point speed.',
                               'The conservative whole-sphere-bottom net margin is not a native collision or fault flag. Negative margin alone does not invalidate a legal return; actual original rule/collision records govern.',
                               'The native substep recovery tracker is authoritative;100Hz root-link corroboration is additional evidence.',
                               'Source training remains incomplete at snapshot time. Policies, normalization and exploration change during the recorded episode.'],
                  source_sha256={str(path.relative_to(ROOT)): sha(path) for path in [Path(__file__), snapshot/'cases.json', observed_path, data_path, data_path.with_suffix('.json'), protocol_path, video.with_suffix('.json')]},
                  formal_acceptance=False, goal_complete=False)
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: result[key] for key in ['actual_case_id', 'actual_outgoing_speed_mps', 'actual_target_far_x_margin_m', 'actual_recovery_s', 'actual_point_speed_mps', 'decoded_frames', 'duration_s', 'sha256']}))


if __name__ == '__main__':
    main()
