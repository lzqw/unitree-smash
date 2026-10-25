"""Reconcile existing model-comparison evidence without rewriting historical results."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
REPORTS = ROOT / 'reports/G07_full_racket_v4'
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from quadsmash.evaluation_v4 import independent_episode


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    original = REPORTS / 'MODEL_RULES.json'
    matched_path = REPORTS / 'matched_flange_kinematics_1245/REPORT.json'
    matched = read(matched_path)
    # Verify the original inputs, without rerunning an old writer over its report.
    for name, expected in matched['source_sha256'].items():
        assert sha(Path(name)) == expected, name
    mount = read(matched_path.with_name('LEGACY_MOUNT.json'))
    assert sha(Path(mount['asset'])) == mount['asset_sha256']
    model = read(original)
    assert sha(ROOT / model['asset']) == model['asset_sha256']
    protocol = read(ROOT / model['formal_protocol'])
    protocol['_root'] = str(ROOT)
    geometry = []
    for record in matched['records']:
        old = np.asarray(record['counterfactual_shortboard_center_m'])
        new = np.asarray(record['reconstructed_full_racket_center_m'])
        assert abs(new[2] - old[2] - record['same_flange_state_center_height_gain_m']) < 1e-12
        omega = np.asarray(record['actual_flange_angular_velocity_rad_s'])
        delta = np.asarray(record['full_racket_center_velocity_mps']) - record['counterfactual_shortboard_center_velocity_mps']
        assert np.max(abs(delta - np.cross(omega, new - old))) < 1e-12
        geometry.append(dict(source_run=record['run'], env_id=record['env_id'], time_s=record['time_s'],
            same_flange_state_center_height_gain_m=float(new[2] - old[2]),
            old_counterfactual_center_speed_mps=record['counterfactual_shortboard_center_speed_mps'],
            new_reconstructed_center_speed_mps=record['full_racket_center_speed_mps'],
            forward_center_velocity_change_mps=record['same_flange_state_forward_center_velocity_gain_mps'],
            old_and_new_normals_are_different=True,
            velocity_scope='Post-integration observed flange state. Kinematic reconstruction only; no independent pre-impact or old-body dynamic capability claim.'))
    run = REPORTS / 'contact_lead_gain_normal_candidate43_cem28'
    native_successes = []
    sources = [original, matched_path, matched_path.with_name('LEGACY_MOUNT.json'),
        REPORTS / 'CEM32_CONTACT_VELOCITY_SCOPE_20261025.json',
        ROOT / 'configs/assembly_v2.json', ROOT / 'configs/full_racket_v4_r3.json',
        ROOT / 'configs/court_net_v4.json', ROOT / model['formal_protocol'], Path(__file__)]
    for generation in range(4):
        path = run / f'episodes_{generation:03}.json'
        rows = read(path)
        sources.append(path)
        for row in rows:
            if not row['complete_smash']:
                continue
            poses_path = run / f'all_first_episodes_{generation:03}.npz'
            sources.append(poses_path)
            with np.load(poses_path) as poses:
                lane = row['env_id']
                assert poses['valid_first_episode'][0, lane]
                assert abs(poses['episode_clock_s'][0, lane]) < 1e-6
                assert np.max(abs(poses['link_poses'][0, lane, 0, :3] - protocol['initialization']['root_position_m'])) < 1e-5
            checked = independent_episode(dict(row, ground_init=True,
                case_id=f'TRAIN{row["source_training_index"]}_G{generation}_C{row["candidate_id"]}'), protocol)
            assert checked['complete'] and not checked['fault'] and row['numerical_failures'] == 0
            native_successes.append(dict(generation=generation, candidate_id=row['candidate_id'],
                source_training_index=row['source_training_index'], env_id=row['env_id'],
                actual_contact_point_m=row['contact_point_m'], outgoing_velocity_mps=row['outgoing_velocity_mps'],
                outgoing_ball_speed_mps=float(np.linalg.norm(row['outgoing_velocity_mps'])),
                first_landing_m=row['first_landing_m'], recovery_s=row['jump']['recovery_s'],
                independently_recomputed_complete=True, initial_ground_pose_and_zero_clock_verified=True))
    assert len(native_successes) == 2 and {r['source_training_index'] for r in native_successes} == {65}
    old_formal_path = ROOT / 'reports/G07/causal_formal400_v2/metrics.json'
    new_formal_path = REPORTS / 'formal_aggregate_cem31_candidate5_1005.json'
    current_path = REPORTS / 'residual_ppo_fixed_train128_1250/metrics.json'
    baseline_path = REPORTS / 'train128_lateral_c12_1025/metrics.json'
    sources += [old_formal_path, new_formal_path, current_path, baseline_path]
    old_formal, new_formal = read(old_formal_path), read(new_formal_path)
    assert old_formal['sample_count'] == new_formal['sample_count'] == 400
    assert not old_formal['thresholds_met'] and not new_formal['formal_acceptance']
    current = read(current_path)
    assert current['complete'] == 0 and current['airborne_contacts'] == 18 and current['conditional_recovery_count'] == 9
    report = dict(created_at=datetime.now(timezone.utc).isoformat(),
        scope='Current evidence supplement for the requested short-board/full-racket comparison. Historical files stay unchanged; no new native rollout or formal acceptance is claimed.',
        original_model_record=str(original.relative_to(ROOT)), asset_sha256=model['asset_sha256'],
        source_rules=model['source_rules'],
        corrections=dict(old_tool_mass_kg=matched['old_payload_mass_kg'], full_tool_fixture_mass_kg=matched['full_racket_and_fixture_mass_kg'],
            old_flange_center_offset_m=matched['old_shortboard_center_in_flange_m'], new_flange_string_center_offset_m=matched['full_racket_string_center_in_flange_m'],
            full_racket_length_m=.675, added_tool_fixture_mass_kg=matched['full_racket_and_fixture_mass_kg'] - matched['old_payload_mass_kg'],
            full_tool_center_of_mass_and_inertia=model['mass_properties'],
            note='The675mm total length is not the flange lever. Clamp/grip overlap, mass, COM, inertia and face/shaft directions are explicit in the pinned model.'),
        matched_observed_flange_kinematics=geometry,
        reach_and_speed_limits='The approximately251mm height changes refer to two observed poses, not global reachable maxima. Post-contact racket-point velocities are not pre-impact calibration. Outgoing ball speed is a distinct quantity.',
        observed_full_model_complete_training_episodes=native_successes,
        complete_example_limits='Two separately executed parameter/generation episodes of the SAME TRAIN65 case, not two distinct cases or a fixed-policy distribution pass.',
        legal_position=dict(net_plane_x_m=read(ROOT / 'configs/court_net_v4.json')['net_plane_x_m'],
            new_model_contact_point_checked_on_own_side=True,
            old_and_new_face_normals_differ=True,
            old_task_not_rule_equivalent='Legacy2m ground-up obstacle/long preparation and short-board contact definitions differ from the suspended6.1m net, realistic court incoming states and actual string/robot legality checks.'),
        historical_legacy_formal400=old_formal,
        new_full_model_previous_policy_formal400=dict(policy_sha256=new_formal['policy_sha256'],
            worst_panel_rates=new_formal['worst_panel_rates'], numerical_failures=new_formal['numerical_failures'],
            formal_acceptance=new_formal['formal_acceptance']),
        current_PPO_iter8_fixed_TRAIN128=current, earlier_full_model_fixed_TRAIN128=read(baseline_path),
        recovery_comparison_limits='The legacy formal400 and new TRAIN/formal suites use different scenes, policies and incoming states. Preserve separate rates; do not infer a causal recovery effect from their difference.',
        historical_model_record_status_correction=dict(original_formal400_completed_flag_was_at_creation=False,
            full_model_formal400_has_since_run=True, full_model_formal400_has_passed=False,
            legal_air_attack_and_recovery_observed=True, single_training_case_complete_observed=True,
            broad_full_task_success_unproven=True),
        active_learning='512-env native PPO continues independently. This document does not freeze or interrupt it.',
        formal_acceptance=False, goal_complete=False,
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in sorted(set(sources))})
    path = HERE / 'REPORT.json'
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(report, indent=2) + '\n')
    tmp.replace(path)
    print(json.dumps(dict(report=str(path.relative_to(ROOT)), matched_poses=len(geometry),
        independently_rechecked_complete_examples=len(native_successes), distinct_complete_TRAIN_cases=1,
        formal_acceptance=False)))


if __name__ == '__main__':
    main()
