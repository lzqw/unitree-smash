"""Audit the declared teacher comparison without inheriting historical scores."""
import json
from pathlib import Path

import numpy as np

from scripts.ground_extent_diagnostic_v1 import compare_completed, validate_bounds
from scripts.run_landing_validation_sequence_v1 import sha, verify_completed
from scripts.run_recovery_curriculum_followup_v1 import score_run
from scripts.standing_contact_telemetry_v1 import path_identity


def contact_event_audit(source, count):
    source = Path(source)
    meta = json.loads((source/'contact_telemetry_metadata.json').read_text())
    events = json.loads((source/'first_nonfoot_contact_details.json').read_text())
    by_name = {s['name']: s for s in meta['sensors']}
    for sensor in meta['sensors']:
        expected = [(i, name) for i in range(count) for name in sensor['body_names']]
        actual = [path_identity(p) for p in sensor['sensor_paths']]
        if (actual != expected or sensor['sensor_paths'] != sensor['body_view_paths']
                or not all(p == ['/World/Ground/geometry/mesh'] for p in sensor['filter_paths'])):
            raise ValueError('Native contact identities or ground filters differ')
    seen, suspicious, errors = [], [], []
    for event in events:
        seen.extend(event['first_event_env_ids'])
        for group in event['sensors']:
            sensor = by_name[group['name']]
            matrix = np.asarray(group['force_matrix_world_n'], dtype=float).reshape(-1, 1, 3)
            for patch in group['native_patch_contacts']:
                if path_identity(sensor['sensor_paths'][patch['sensor_row']]) != (patch['env_id'], patch['body']):
                    raise ValueError('Patch and force row identities differ')
                forces = np.asarray(patch['normal_force_n'], dtype=float)
                points = np.asarray(patch['point_world_m'], dtype=float)
                normals = np.asarray(patch['normal_world'], dtype=float)
                separation = np.asarray(patch['separation_m'], dtype=float)
                actual = matrix[patch['sensor_row'], patch['filter_column']]
                total = (forces[:, None]*normals).sum(0)
                if not all(np.isfinite(v).all() for v in [forces, points, normals, separation, total]):
                    raise ValueError('Nonfinite native contact event')
                if not np.allclose(total, actual, atol=1e-3, rtol=2e-6):
                    raise ValueError('Raw patch and original force matrix disagree')
                errors.append(float(abs(total-actual).max()))
                bad = (forces > 1.) & (points[:, 2] > .05) & (normals[:, 2] < -.5) & (separation < -.05)
                if bad.any():
                    suspicious.append(dict(physics_frame=event['physics_frame'], env_id=patch['env_id'],
                        body=patch['body'], force_world_n=total.tolist(),
                        points_world_m=points[bad].tolist(), normals_world=normals[bad].tolist(),
                        separations_m=separation[bad].tolist()))
    if len(seen) != len(set(seen)):
        raise ValueError('Duplicated first-episode contact events')
    return dict(first_event_lanes=sorted(seen), suspicious_elevated_downward_patches=suspicious,
        maximum_raw_sum_error_n=max(errors, default=0.),
        limitation='Only recorded first-event patches; absence does not certify all collision responses',
        evidence_sha256={name:sha(source/name) for name in
            ['contact_telemetry_metadata.json', 'first_nonfoot_contact_details.json']})


def standing_diagnostic_gate(source, baseline):
    """Choose whether to spend the next job on task-level extent comparison.

    This is an experiment-selection check, not a new task acceptance condition.
    """
    source, baseline = Path(source), Path(baseline)
    measured = compare_completed(source, baseline)
    candidate, original = [contact_event_audit(p, 12) for p in [source, baseline]]
    scans = []
    for p in [source, baseline]:
        with np.load(p/'native_contact_telemetry.npz') as z:
            force = z['nonfoot_ground_force_world_n']
        scans.append(dict(source=str(p), minimum_vertical_force_n=float(force[..., 2].min()),
            samples_below_minus_one_n=int((force[..., 2] < -1.).sum())))
    proceed = bool(original['suspicious_elevated_downward_patches']
        and not candidate['suspicious_elevated_downward_patches']
        and scans[0]['samples_below_minus_one_n'] == 0
        and measured['numerical_failures'] == 0)
    return dict(proceed_to_teacher_diagnostic=proceed, standing_counts=measured['counts'],
        original_standing_counts=measured['ground_extent_comparison']['baseline_counts'],
        candidate_contact_events=candidate, original_contact_events=original,
        continuous_vertical_force_scans=scans,
        reason='Compare full teacher task only after the observed contact anomaly disappears in this standing diagnostic',
        physical_cause_proven=False, landing_success_claimed=False,
        formal_acceptance=False, goal_complete=False)


def compare_teacher(source, baseline, protocol):
    source, baseline = Path(source), Path(baseline)
    measured, original = [score_run(p, protocol) for p in [source, baseline]]
    one, two = [json.loads((p/'INPUTS.json').read_text()) for p in [source, baseline]]
    keys = ['policy_sha256','protocol_sha256','training_suite','source_checkpoint_protocol_sha256',
        'origin_precision_receipt_sha256','origin_spacing_m','training_indices',
        'reset_noise_amplitudes','reset_noise_seed','controller_trial','landing_trial',
        'support_trial','support_config','support_disabled','record_all_first_episode_actions',
        'residual_actor_enabled','reset_noise','formal_cases_read']
    for key in keys:
        if one[key] != two[key]:
            raise ValueError('Teacher comparison changed '+key)
    for name in ['initial_states.json','native_command_inputs.json','origin_layout.json']:
        if json.loads((source/name).read_text()) != json.loads((baseline/name).read_text()):
            raise ValueError('Actual teacher comparison differs in '+name)
    ground = json.loads((source/'ground_geometry.json').read_text())
    validate_bounds(ground['actual_world_lower_m'],ground['actual_world_upper_m'],ground['requested_dimensions_m'])
    if not ground['actual_collision_enabled'] or ground['requested_dimensions_m'] != [64.,64.,.1]:
        raise ValueError('Unexpected ground geometry')
    if not json.loads((source/'native_admission.json').read_text())['passed']:
        raise ValueError('Actual model/actuator admission failed')
    events = contact_event_audit(source,128)
    rows = json.loads((source/'episodes.json').read_text())
    expected = sorted(r['env_id'] for r in rows if r['jump']['maximum_body_ground_force_n'] > 1.)
    if events['first_event_lanes'] != expected:
        raise ValueError('Recorded first events disagree with actual episode maxima')
    with np.load(source/'all_first_episodes.npz') as z:
        poses, valid = z['link_poses'], z['valid_first_episode']
    extent = np.max(np.abs(poses[..., :2])[valid],axis=(0,1))
    if np.any(extent+1. >= 32.):
        raise ValueError('Recorded link origins approach the changed remote boundary')
    return dict(scope='Original TRAIN128 teacher, same task thresholds, explicit numerical ground-extent diagnostic',
        original4096=original, candidate64=measured, ground_geometry=ground,
        actual_initial_states_and_control_inputs_equal=True,
        maximum_recorded_link_origin_abs_xy_m=extent.tolist(),
        boundary_check_scope='Recorded link origins plus 1 m margin, not a swept-collider proof',
        candidate_contact_events=events,
        original_scores_relabelled=False, formal_acceptance=False, goal_complete=False,
        evidence_sha256={name:sha(source/name) for name in ['ground_geometry.json',
            'native_admission.json','initial_states.json','native_command_inputs.json',
            'contact_telemetry_summary.json','all_first_episodes.npz']})
