"""Fourth actual TRAIN trace with the original finite-sphere net-passage rule.

The preserved prior auditor's extra positive late bottom-margin diagnostic
rejected this case. Its source, raw data and failed check remain unchanged.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
REPORTS = ROOT/'reports/G07_full_racket_v4'
SNAPSHOT = REPORTS/'video_snapshot_placement075_fourth_complete_20261026'
AUDITOR = REPORTS/'placement075_additional_complete_training_20261026/audit.py'
sys.path.insert(0, str(ROOT))
from quadsmash.evaluation_v4 import independent_episode
from quadsmash.rl.episode_trace_v4 import unpack_public
from quadsmash.physics.court_v4 import net_band_segments, top_height

def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def same(a,b):return json.dumps(a,sort_keys=True)==json.dumps(b,sort_keys=True)

def audit_case(case, protocol, indexed, recorded):
    row, event = case['row'], case['contact']
    key = row['env_id'], row['episode_index']
    directory = SNAPSHOT/case['run']
    path = directory/f'lane{key[0]:04d}_episode{key[1]:06d}.npz'
    meta = read(path.with_suffix('.json'))
    entry = next(e for e in recorded['saved_episodes'] if (e['env_id'], e['episode_index']) == key)
    assert same(row, indexed[key]) and same(row, meta['raw_episode'])
    assert sha(path) == meta['data_sha256'] == entry['data_sha256']
    assert meta['native_execution'] and meta['full_public_reconstruction_bitwise_verified']
    assert meta['actual_contact_events'] == [event]
    outcome = independent_episode(row, protocol)
    assert outcome['complete'] and outcome['airborne_contact'] and outcome['conditional_recovery']
    assert not outcome['fault'] and row['numerical_failures'] == 0
    assert row['ground_init'] and row['active_rally_launch_count'] == row['contact_count'] == 1
    assert event['string_contact'] and not event['frame_contact'] and not event['fixture_contact']
    assert event['legal_initial_contact'] and not any(event['fault_flags'].values())
    outgoing = np.asarray(row['outgoing_velocity_mps'])
    speed = float(np.linalg.norm(outgoing))
    assert speed >= 8. and outgoing[0] > 0 and outgoing[2] <= -1.
    landing = np.asarray(row['first_landing_m'])
    assert 2. <= landing[0] <= 4. and abs(landing[1]) <= 1.
    assert row['contact_time_s'] < row['crossing_time_s'] < row['first_landing_time_s'] < 8.
    # Original legal crossing uses trailing-sphere center above actual strip top;
    # a later bottom-minus-top diagnostic is not a required acceptance margin.
    court = read(ROOT/'configs/court_net_v4.json')
    from quadsmash.physics.court_v4 import top_height
    assert row['crossed_net'] and row['crossing_point_m'][2] >= float(top_height(row['crossing_point_m'][1], court))
    assert row['jump']['recovery_s'] >= .8
    assert not row['jump']['body_landing'] and row['jump']['maximum_body_ground_force_n'] == 0.
    names = read(directory/'geometry.json')['body_names']
    poses = np.asarray(event['body_link_poses_local'])
    feet = {n: float(poses[names.index(n), 2]) for n in ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']}
    assert min(feet.values()) > .035
    racket = poses[names.index('Racket')]
    local_point = Rotation.from_quat(racket[3:][[1, 2, 3, 0]]).inv().apply(np.asarray(event['point_m'])-racket[:3])
    rho_squared = float(np.sum((local_point[1:]/[.0915, .1165])**2))
    assert rho_squared <= 1.
    velocity = np.asarray(event['point_velocity_mps'])
    reconstructed = np.asarray(event['flange_linear_velocity_mps'])+np.cross(
        event['flange_angular_velocity_rad_s'], np.asarray(event['point_m'])-event['flange_position_m'])
    residual = float(np.linalg.norm(velocity-reconstructed))
    precision = float(np.finfo(np.float32).eps*8*max(1., np.linalg.norm(velocity), np.linalg.norm(reconstructed)))
    assert abs(residual-row['hit_velocity_error_mps']) <= precision
    with np.load(path) as data:
        public = unpack_public(data['packed_public'])
        all_poses = data['link_poses']
        assert public.shape == (800, 794) and all_poses.shape == (800, len(names), 7)
        assert np.isfinite(public).all() and np.isfinite(all_poses).all()
        assert hashlib.sha256(public.tobytes()).hexdigest() == meta['full_public794_sha256'] == entry['full_public794_sha256']
        assert all(data[n].shape == (800, 18) and np.isfinite(data[n]).all()
            for n in ['residual_actions', 'base_actions', 'native_actions'])
        assert np.max(abs(data['native_actions'])) <= 1.000001
        assert np.array_equal(data['episode_clock_s'], public[:, 775])
        assert abs(public[0, 775]) < 1e-6 and np.max(abs(np.diff(public[:, 775])-.01)) < 2e-6
        assert abs(float(public[-1, 775])+.01-row['terminal_native_state']['time_s']) <= 2e-5
        root_error = float(np.max(abs(all_poses[0, 0, :3]-protocol['initialization']['root_position_m'])))
        assert root_error <= 2e-5 and np.max(abs(all_poses[0, 0, 3:]-[1., 0., 0., 0.])) <= 1e-5
        iterations = np.unique(data['learner_iteration']).tolist()
        assert iterations == case['actor_iterations_encountered'] == meta['actor_iterations_encountered'] == entry['actor_iterations_encountered']
        inverse = Rotation.from_quat(all_poses[:, 0, 3:][:, [1, 2, 3, 0]]).inv()
        tool = names.index('Racket')
        tool_error = float(np.max(np.linalg.norm(inverse.apply(all_poses[:, tool, :3]-all_poses[:, 0, :3])-public[:, 45:48], axis=1)))
        normals = Rotation.from_quat(all_poses[:, tool, 3:][:, [1, 2, 3, 0]]).apply(np.tile([1., 0., 0.], (800, 1)))
        normal_error = float(np.max(np.linalg.norm(inverse.apply(normals)-public[:, 48:51], axis=1)))
        present = public[:, 67] > .5
        ball_error = float(np.max(np.linalg.norm(inverse.apply(data['ball_position_m']-all_poses[:, 0, :3])[present]-public[present, 61:64], axis=1)))
        assert tool_error <= 2e-4 and ball_error <= 2e-4 and normal_error <= 5e-5
        last = public[:, 775] >= 7.2
        angles = Rotation.from_quat(all_poses[last, 0, 3:][:, [1, 2, 3, 0]]).as_euler('xyz', degrees=True)
        corroboration = dict(samples=int(last.sum()), start_time_s=float(public[last, 775][0]),
            maximum_abs_roll_pitch_degrees=float(np.max(abs(angles[:, :2]))),
            maximum_root_link_horizontal_speed_mps=float(np.max(np.linalg.norm(public[last, :2], axis=1))),
            maximum_root_angular_speed_rad_s=float(np.max(np.linalg.norm(public[last, 3:6], axis=1))),
            maximum_abs_joint_speed_rad_s=float(np.max(abs(public[last, 27:45]))),
            minimum_supporting_feet=int(np.min(np.sum(public[last, 57:61] > .5, axis=1))))
        assert corroboration['maximum_abs_roll_pitch_degrees'] <= 15.
        assert corroboration['maximum_root_link_horizontal_speed_mps'] <= .3
        assert corroboration['maximum_root_angular_speed_rad_s'] <= 1.
        assert corroboration['maximum_abs_joint_speed_rad_s'] <= 1.
        assert corroboration['minimum_supporting_feet'] > 0
    return dict(actual_case_id=row['case_id'], env_id=key[0], episode_index=key[1], TRAIN=row['source_training_index'],
        actual_independent_outcome=outcome, actual100Hz_samples=800, initial_root_position_max_error_m=root_error,
        actor_iterations_encountered=iterations, actual_contact_time_s=row['contact_time_s'],
        actual_contact_point_m=row['contact_point_m'], actual_contact_point_local_racket_m=local_point.tolist(),
        actual_contact_point_strings_rho_squared=rho_squared, actual_feet_heights_at_contact_m=feet,
        actual_outgoing_velocity_mps=outgoing.tolist(), actual_outgoing_speed_mps=speed,
        actual_speed_threshold_margin_mps=speed-8., actual_crossing_time_s=row['crossing_time_s'],
        actual_net_margin_m=row['net_margin_m'], actual_first_landing_m=landing.tolist(),
        actual_first_landing_time_s=row['first_landing_time_s'], actual_target_far_x_margin_m=4.-landing[0],
        actual_recovery_s=row['jump']['recovery_s'], actual_point_speed_mps=float(np.linalg.norm(velocity)),
        flange_plus_rotation_point_velocity_error_mps=residual,
        recorded_native_point_velocity_consistency_error_mps=row['hit_velocity_error_mps'],
        diagnostic_residual_agreement_bound_from_float32_mps=precision,
        below_prior_capture_extra_diagnostic_bound_2e_minus5=bool(residual <= 2e-5),
        no_original_acceptance_tolerance_changed=True, last_080s_public_pose_and_velocity_corroboration=corroboration,
        tool_public_pose_max_error_m=tool_error, normal_public_pose_max_error=normal_error,
        ball_public_pose_max_error_m=ball_error, original_data_sha256=sha(path),
        full_public794_sha256=meta['full_public794_sha256']), [path, path.with_suffix('.json')]


def segment_box_distance(a, b, low, high):
    direction = b-a
    breaks = [0., 1.]
    for axis in range(3):
        if direction[axis] != 0:
            breaks.extend(float(t) for t in [(low[axis]-a[axis])/direction[axis], (high[axis]-a[axis])/direction[axis]] if 0 < t < 1)
    breaks = sorted(set(breaks)); candidates = list(breaks)
    for left,right in zip(breaks[:-1],breaks[1:]):
        middle = a+(left+right)*.5*direction
        below,above = middle<low,middle>high
        coefficients = np.where(below,low-a,np.where(above,a-high,0.))
        slope = np.where(below,-direction,np.where(above,direction,0.))
        if np.dot(slope,slope) > 0:
            candidates.append(float(np.clip(-np.dot(coefficients,slope)/np.dot(slope,slope),left,right)))
    distance = lambda t: float(np.linalg.norm(np.maximum(low-(a+t*direction),0.)+np.maximum((a+t*direction)-high,0.)))
    t = min(candidates,key=distance)
    return distance(t),t


def sampled_net_geometry(case):
    row = case['row'];court=read(ROOT/'configs/court_net_v4.json')
    data_path=SNAPSHOT/case['run']/'lane1466_episode000006.npz'
    with np.load(data_path)as data:
        points=data['ball_position_m'].astype(np.float64);clock=data['episode_clock_s'].astype(np.float64)
    # Use every actual100Hz chord spanning the near-net0.2m slab; no synthetic
    # motion is rendered and no continuous-substep collision proof is claimed.
    ids=[i for i in range(len(points)-1) if clock[i]>row['release_time_s'] and clock[i+1]<row['first_landing_time_s']
         and min(points[i,0],points[i+1,0])<=court['net_plane_x_m']+.1 and max(points[i,0],points[i+1,0])>=court['net_plane_x_m']-.1]
    assert ids
    minimum=float('inf');witness=None
    for index in ids:
        for segment,(vertices,_)in enumerate(net_band_segments(court)):
            box=np.asarray(vertices,dtype=float);box[:,0]+=court['net_plane_x_m']
            distance,fraction=segment_box_distance(points[index],points[index+1],box.min(0),box.max(0))
            if distance<minimum:minimum=distance;witness=dict(sample_interval=index,segment=segment,fraction=fraction)
    height=float(top_height(row['crossing_point_m'][1],court))
    assert row['crossing_point_m'][2]>=height and not row['shuttle_net_contact']
    assert minimum>.025
    return dict(actual100Hz_chords_checked=len(ids),actual_net_primitive_boxes=64,
        minimum_actual_sample_chord_center_distance_to_net_boxes_m=minimum,
        sample_chord_sphere_clearance_m=minimum-.025,closest_chord_witness=witness,
        original_net_thickness_m=court['net_equivalent_thickness_m'],
        original_trailing_sphere_center_x_m=row['crossing_point_m'][0],
        original_trailing_sphere_center_above_actual_strip_top_m=row['crossing_point_m'][2]-height,
        original_late_bottom_margin_m=row['net_margin_m'],original_native_shuttle_net_contact=row['shuttle_net_contact'],
        native_substep_net_contact_and_passage_remain_authoritative=True,
        scope='Exact segment/AABB distance for chords between actual100Hz ball positions, additional geometric corroboration only. It is not a replay or a bound on unrecorded substep curvature.')

def main():
    output = HERE/'FOURTH_COMPLETE_NATIVE_EPISODE_CHECK.json'
    assert not output.exists()
    manifest = read(SNAPSHOT/'cases.json')
    assert not manifest['source_job_complete'] and manifest['includes_failure']
    assert [(c['env_id'], c['episode_index']) for c in manifest['cases']] == [(1466, 6), (23, 2), (8, 3)]
    directory = SNAPSHOT/manifest['cases'][0]['run']
    raw_path, trace_path = directory/'observed_episodes.json', directory/'observed_trace_manifest.json'
    assert sha(raw_path) == manifest['observed_raw_publication_sha256']
    assert sha(trace_path) == manifest['observed_trace_manifest_sha256']
    rows, recorded = read(raw_path), read(trace_path)
    indexed = {(r['env_id'], r['episode_index']): r for r in rows}
    assert len(indexed) == len(rows)
    protocol_path = ROOT/'configs/g07_full_racket_eval_v4_r3.json'
    protocol = read(protocol_path)
    protocol['_root'] = str(ROOT)
    checked, paths = audit_case(manifest['cases'][0], protocol, indexed, recorded)
    checked['actual_net_geometry_corroboration'] = sampled_net_geometry(manifest['cases'][0])
    checked['prior_extra_positive_late_bottom_margin_gate_passed'] = False
    assert checked['TRAIN'] == 58 and checked['actor_iterations_encountered'] == list(range(19, 26))
    outcomes = [independent_episode(r, protocol) for r in rows]
    counts = dict(closed_episodes=len(rows), air=sum(o['airborne_contact'] for o in outcomes),
        conditional_recovery=sum(o['conditional_recovery'] for o in outcomes),
        complete=sum(o['complete'] for o in outcomes), numerical_failures=sum(r['numerical_failures'] for r in rows))
    assert counts['complete'] >= 4 and counts['numerical_failures'] == 0
    assert all(not independent_episode(c['row'], protocol)['complete'] for c in manifest['cases'][1:])
    dependencies = [Path(__file__), AUDITOR, SNAPSHOT/'cases.json', raw_path, trace_path,
        protocol_path, directory/'geometry.json', ROOT/'quadsmash/evaluation_v4.py',
        ROOT/'configs/court_net_v4.json', ROOT/'quadsmash/physics/court_v4.py', ROOT/'quadsmash/physics/rally_rules_v4.py',
        HERE/'INITIAL_REUSED_AUDITOR.py', HERE/'INITIAL_EXTRA_DIAGNOSTIC_REJECTION.json',
        ROOT/'quadsmash/tasks/stand/observation_contract.py', *paths]
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), actual_fourth_complete_episode=checked,
        all_observed_closed_changing_policy_counts=counts, retained_failure_keys=[[23, 2], [8, 3]],
        scope='Actual fourth complete ground-start TRAIN episode of075 strict continuation, using original legal net passage and a separately versioned100Hz audit. Preserved prior extra diagnostic rejection retained. Actor19..25, not a frozen-policy result.',
        source_job_complete_at_capture=False,
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in dependencies},
        formal_cases_read=False, new_native_execution=False, frozen_policy_performance=False,
        formal_acceptance=False, goal_complete=False)
    for path in paths:
        destination = HERE/path.name
        assert not destination.exists()
        shutil.copy2(path, destination)
        assert sha(destination) == sha(path)
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: checked[key] for key in ['TRAIN', 'env_id', 'episode_index',
        'actual_outgoing_speed_mps', 'actual_first_landing_m', 'actual_target_far_x_margin_m',
        'actual_recovery_s', 'actual_point_speed_mps']}, indent=2))


if __name__ == '__main__':
    main()
