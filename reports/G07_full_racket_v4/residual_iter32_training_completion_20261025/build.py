"""Independently check completed512-env learning; frozen assessment stays separate."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
REPORTS = ROOT / 'reports/G07_full_racket_v4'
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not (HERE / 'REPORT.json').exists(), 'Never overwrite a preserved completion report'
    spec = importlib.util.spec_from_file_location('completed512_checks', REPORTS / 'residual_resize512_queue01/run.py')
    checks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checks)
    core, run, evaluation = checks.core, checks.TRAIN, checks.EVALUATION
    job = core.verify_job(run, checks.STEPS)
    trained = checks.verify_training()
    from quadsmash.evaluation_v4 import independent_episode
    protocol_path = ROOT / 'configs/g07_full_racket_eval_v4_r3.json'
    protocol = core.read(protocol_path)
    protocol['_root'] = str(ROOT)
    rows = core.read(run / 'episodes.json')
    assert len({(r['env_id'], r['episode_index']) for r in rows}) == len(rows)
    checked = [independent_episode(row, protocol) for row in rows]
    counts = dict(completed_episodes=len(rows), air=sum(c['airborne_contact'] for c in checked),
        conditional_recovery_count=sum(c['conditional_recovery'] for c in checked),
        complete=sum(c['complete'] for c in checked), numerical_failures=sum(r['numerical_failures'] for r in rows))
    assert counts == trained['changing_policy_counts']
    assert all(r['ground_init'] and r['source_training_index'] == r['env_id'] % 128 for r in rows)
    complete = [r for r, c in zip(rows, checked) if c['complete']]
    assert len(complete) == 1
    row = complete[0]
    assert row['case_id'] == 'TRAIN19_LANE275_EP1'
    milestone = REPORTS / 'residual_n512_first_complete_20261025'
    assert json.dumps(row, sort_keys=True) == json.dumps(core.read(milestone / 'complete_episode.json'), sort_keys=True)
    contacts_path = run / 'contacts.json'
    events = [c for c in core.read(contacts_path) if c['env_id'] == 275 and c['episode_index'] == 1]
    assert len(events) == row['contact_count'] == 1
    event = events[0]
    assert event['first_racket_contact'] and event['string_contact'] and event['legal_initial_contact']
    assert not event['frame_contact'] and not event['fixture_contact'] and not any(event['fault_flags'].values())
    assert abs(event['time_s'] - row['contact_time_s']) <= 1e-7
    assert np.max(abs(np.asarray(event['point_m']) - row['contact_point_m'])) <= 1e-7
    assert np.asarray(event['body_link_poses_local']).shape == (28, 7)
    assert np.isfinite(event['body_link_poses_local']).all()
    expected = np.asarray(event['flange_linear_velocity_mps']) + np.cross(
        event['flange_angular_velocity_rad_s'], np.asarray(event['point_m']) - event['flange_position_m'])
    velocity_error = float(np.linalg.norm(expected - event['point_velocity_mps']))
    assert velocity_error <= 1e-5
    confirmation_path = milestone / 'FINAL_CONTACT_CONFIRMATION.json'
    assert not confirmation_path.exists()
    core.atomic(confirmation_path, dict(created_at=datetime.now(timezone.utc).isoformat(),
        native_case_id=row['case_id'], preserved_episode_sha256=sha(milestone / 'complete_episode.json'),
        completed_training_episodes_sha256=sha(run / 'episodes.json'),
        completed_training_contacts_sha256=sha(contacts_path), exact_contact_record_matches=True,
        rigid_point_velocity_identity_error_mps=velocity_error, actual_native_contact=event,
        velocity_scope='Recorded post-PhysX contact integration state, not independently measured incident impact velocity.',
        original_snapshot_pending_flag_retained_as_history=True,
        full_training_motion_trajectory_available=False, frozen_policy_performance=False, formal_acceptance=False))
    ground = core.read(evaluation / 'initial_states.json')
    inputs = core.read(evaluation / 'INPUTS.json')
    source_ground = core.read(run / 'initial_ground_states.json')
    assert len(ground) == 128 and [g['env_id'] for g in ground] == list(range(128))
    assert inputs['training_indices'] == list(range(128))
    assert inputs['policy_sha256'] == trained['policy_sha256']
    assert all(value == 0 for value in inputs['reset_noise_amplitudes'].values())
    for item in ground:
        lane = item['env_id']
        assert np.max(abs(np.asarray(item['root_state_local']) - source_ground['root_state_local'][lane])) <= 1e-5
        assert np.max(abs(np.asarray(item['joint_position_rad']) - source_ground['joint_position_rad'][lane])) <= 1e-5
        assert np.max(abs(np.asarray(item['joint_velocity_rad_s']))) <= 1e-5
    policy = torch.load(run / 'policy.pt', map_location='cpu', weights_only=False)
    history = core.read(run / 'metrics.json')['history']
    report = dict(created_at=datetime.now(timezone.utc).isoformat(),
        scope='Completed native512-env changing-policy TRAIN learning, plus source/ground verification of a newly started fixed assessment. No frozen-policy performance yet.',
        training=trained, independently_recomputed_changing_policy_counts=counts,
        update_history=history,
        actual_gpu_hours=job['charged_gpu_hours'], actual_current_job_control_env_steps=checks.STEPS,
        cumulative_learner_control_env_steps=1703936, training_source_commit=job['git_sha'],
        terminal_training_job_status=job['status'], canonical_restart=True,
        model_optimizer_tensors_finite=True, actor_export=policy['residual_training_metadata'],
        confirmed_complete_training_episode=row['case_id'],
        complete_contact_confirmation=str(confirmation_path.relative_to(ROOT)),
        fixed_TRAIN128_launch=dict(output=str(evaluation.relative_to(ROOT)),
            policy_sha256=inputs['policy_sha256'], source_inputs_sha256=sha(evaluation / 'INPUTS.json'),
            actual_initial_ground_states_sha256=sha(evaluation / 'initial_states.json'),
            all128_original_initial_states_verified=True, reset_noise='none', performance_pending=True),
        protocol_sha256=sha(protocol_path), asset_sha256=protocol['asset']['sha256'],
        sources={str((run / name).relative_to(ROOT)): sha(run / name) for name in
            ['job.json', 'completed.json', 'metrics.json', 'INPUTS.json', 'checkpoint.pt', 'policy.pt',
             'episodes.json', 'contacts.json', 'initial_ground_states.json', 'source_snapshot.zip', 'source_hashes.json']},
        limitations=['3171 closed training episodes experience changing actors/normalizers and stochastic exploration; they are not one frozen policy sample.',
            'The single complete training episode spans actor14..20; final actor32 is not assigned that success.',
            'Only contact and terminal native states exist for that training episode, not its complete motion trace.',
            '128 distinct TRAIN cases were repeated four times in the512-env batch.',
            'Original formal A/B each200 thresholds and full deliverables remain required.'],
        formal_cases_read=False, formal_acceptance=False, goal_complete=False)
    core.atomic(HERE / 'REPORT.json', report)
    print(json.dumps(dict(training_policy_sha256=trained['policy_sha256'], counts=counts,
        actual_gpu_hours=job['charged_gpu_hours'], contact_record_matched=True,
        fixed_TRAIN128_initial_ground_states_verified=True, formal_acceptance=False)), flush=True)


if __name__ == '__main__':
    main()
