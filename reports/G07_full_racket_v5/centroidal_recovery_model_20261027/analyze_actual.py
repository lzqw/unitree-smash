"""Validate known-model public-state reconstruction against real native traces."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
MAIN = HERE.parents[2]
ROOT = MAIN.parent / 'unitree_smash_contact_recovery'
sys.path.insert(0, str(ROOT))
from quadsmash.rl.episode_trace_v4 import unpack_public
from known_model import KnownModel, rotation, apply


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def longest_sampled_run(mask, clock):
    best = current = 0
    first_qualifying = None
    for active, time in zip(mask, clock):
        current = current+1 if active else 0
        best = max(best, current)
        if current >= 80 and first_qualifying is None:
            first_qualifying = float(time)
    return dict(longest_sampled_stable_duration_s=best*.01,
        first_sampled_08s_window_end_clock_s=first_qualifying,
        sampled_dt_s=.01, native_first_qualifying_clock_not_recorded=True)


def check(model, public, defaults, actual_poses, stride):
    state = model.public_world_state(public, defaults)
    calculated = state['body_position_world'][::stride]
    assert calculated.shape == actual_poses[:, :, :3].shape
    position_error = np.linalg.norm(calculated-actual_poses[:, :, :3], axis=-1)
    actual_rotation = rotation(actual_poses[:, :, 3:])
    model_rotation = state['root_rotation_world'][::stride, None] @ state['rotation'][::stride]
    difference = model_rotation.transpose(0, 1, 3, 2) @ actual_rotation
    angle_error = Rotation.from_matrix(difference.reshape(-1, 3, 3)).magnitude()
    racket = model.indices['Racket']
    relative_linear = state['linear'][:, racket]-public[:, :3]-np.cross(public[:, 3:6], state['position'][:, racket])
    relative_angular = state['angular'][:, racket]-public[:, 3:6]
    errors = dict(body_position_max_m=float(position_error.max()),
        body_orientation_max_rad=float(angle_error.max()),
        material_frame_origin_public_position_max_m=float(abs(state['position'][:, racket]-public[:, 45:48]).max()),
        tool_relative_linear_velocity_max_m_s=float(abs(relative_linear-public[:, 51:54]).max()),
        tool_relative_angular_velocity_max_rad_s=float(abs(relative_angular-public[:, 54:57]).max()))
    assert errors['body_position_max_m'] < 2e-5
    assert errors['body_orientation_max_rad'] < 5e-5
    assert errors['material_frame_origin_public_position_max_m'] < 2e-5
    assert errors['tool_relative_linear_velocity_max_m_s'] < 1e-3
    assert errors['tool_relative_angular_velocity_max_rad_s'] < 1e-3
    return state, errors


def main():
    model = KnownModel()
    assert all(sha(Path(p)) == h for p, h in model.record['source_sha256'].items())
    measured = HERE.parent / 'measured_swing9_n128_01'
    inputs_path = measured / 'INPUTS.json'
    defaults = np.asarray(json.loads(inputs_path.read_text())['default_joint_position_rad'])
    data_path, pose_path = measured/'public_and_actions_000.npz', measured/'all_first_episodes_000.npz'
    with np.load(data_path) as data:
        public, indices = data['observations'].copy(), data['training_indices'].tolist()
        assert data['valid_first_episode'].all()
    with np.load(pose_path) as data:
        poses = data['link_poses'][:, indices].copy()
        assert np.array_equal(data['episode_clock_s'][:, indices], public[::2, :, 775])
    reference_checks, source_states = [], []
    for lane, index in enumerate(indices):
        state, result = check(model, public[:, lane], defaults, poses[:, lane], 2)
        reference_checks.append(dict(training_index=index, actual_public_rows=800, actual_pose_rows=400, **result))
        source_states.append(state)
    trace = HERE.parent/'residual_contact_recovery_n128_updates15_02/trajectory_traces/lane0069_episode000000.npz'
    trace_metadata = trace.with_suffix('.json')
    metadata = json.loads(trace_metadata.read_text())
    assert metadata['data_sha256'] == sha(trace)
    with np.load(trace) as data:
        recovered = unpack_public(data['packed_public'])
        recovered_poses = data['link_poses'].copy()
    assert hashlib.sha256(recovered.tobytes()).hexdigest() == metadata['full_public794_sha256']
    recovered_state, recovered_errors = check(model, recovered, defaults, recovered_poses, 1)
    failed_index = indices.index(69)
    failed, failed_state = public[:, failed_index], source_states[failed_index]
    original = json.loads((measured/'episodes_000.json').read_text())[69]
    recovered_row = metadata['raw_episode']
    assert original['contact_flight_recovery']['bound'] and recovered_row['contact_flight_recovery']['success']
    assert original['contact_flight_recovery']['body_landing'] and not recovered_row['contact_flight_recovery']['body_landing']
    # The authored racket principal frame must reconstruct its complete
    # compound inertia, including the off-diagonal flange/shaft coupling.
    asset_manifest = ROOT/'reports/G07_full_racket_v4/build_asset_r3_01/asset_manifest.json'
    expected = np.asarray(json.loads(asset_manifest.read_text())['mass_properties']['inertia_in_body_kg_m2'])
    inertia_error = float(abs(model.inertia[model.indices['Racket']]-expected).max())
    assert inertia_error < 1e-8
    diagnostics, saved = [], {}
    feet = [model.indices[name+'_foot'] for name in ['FL', 'FR', 'RL', 'RR']]
    recovery_protocol = json.loads((ROOT/'configs/evaluation_protocol_v2.json').read_text())['recovery']
    for name, obs, state, row in [('failed_source69', failed, failed_state, original),
                                 ('recovered_exploration69', recovered, recovered_state, recovered_row)]:
        time, momentum = obs[:, 775], state['angular_momentum_world']
        euler = Rotation.from_matrix(state['root_rotation_world']).as_euler('xyz')
        root_linear_world = apply(state['root_rotation_world'], obs[:, :3])
        windows = []
        for lower, upper in [(2.60, 2.70), (2.78, 2.98), (3.15, 4.)]:
            use = (time >= lower)&(time < upper)
            values = momentum[use]
            average = values.mean(0)
            windows.append(dict(from_s=lower, to_s=upper, actual_control_rows=int(use.sum()),
                mean_world_angular_momentum_kg_m2_s=average.tolist(),
                maximum_deviation_from_mean_kg_m2_s=float(np.linalg.norm(values-average, axis=1).max()),
                root_angular_speed_max_rad_s=float(np.linalg.norm(obs[use, 3:6], axis=1).max()),
                root_tilt_max_deg=float(np.rad2deg(abs(euler[use, :2])).max())))
        sample_times = [2.60, 2.70, 2.78, 2.98, 3.06, 3.15, 3.30, 3.50, 4.]
        samples = []
        for clock in sample_times:
            k = int(np.argmin(abs(time-clock)))
            samples.append(dict(clock_s=float(time[k]), world_angular_momentum_kg_m2_s=momentum[k].tolist(),
                root_rpy_deg=np.rad2deg(euler[k]).tolist(), root_angular_velocity_body_rad_s=obs[k, 3:6].tolist(),
                observed_foot_support=obs[k, 57:61].tolist(), center_world_m=state['center_world'][k].tolist(),
                feet_world_m=state['body_position_world'][k, feet].tolist(),
                feet_linear_velocity_world_m_s=state['body_velocity_world'][k, feet].tolist()))
        # M(q) maps root angular and18 joint velocities to momentum. Root
        # translational velocity cancels exactly about the total COM.
        chosen = [int(np.argmin(abs(time-t))) for t in sample_times]
        q = defaults[None]+obs[chosen, 9:27]
        matrix = model.angular_momentum_matrix(q)
        measured_velocity = np.concatenate((obs[chosen, 3:6], obs[chosen, 27:45]), axis=1)
        composed = apply(matrix, measured_velocity)
        matrix_error = float(abs(composed-state['angular_momentum'][chosen]).max())
        assert matrix_error < 1e-10
        locked = matrix[:, :, :3]
        symmetry_error = float(abs(locked-locked.transpose(0, 2, 1)).max())
        assert symmetry_error < 1e-10 and (np.linalg.eigvalsh(locked)>0).all()
        stable = ((time >= row['contact_flight_recovery']['contact_flight_landing_s']) & (obs[:, 57:61]>.5).any(1)
            & (abs(euler[:, :2]).max(1) <= np.deg2rad(recovery_protocol['maximum_abs_roll_pitch_deg']))
            & (np.linalg.norm(root_linear_world[:, :2], axis=1) <= recovery_protocol['maximum_horizontal_root_speed_m_s'])
            & (np.linalg.norm(obs[:, 3:6], axis=1) <= recovery_protocol['maximum_root_angular_speed_rad_s'])
            & (abs(obs[:, 27:45]).max(1) <= recovery_protocol['maximum_abs_joint_velocity_rad_s']))
        sampled = longest_sampled_run(stable, time)
        sampled['sampled_kinematic_only_not_native_rescore'] = True
        sampled['episode_body_ground_fault'] = row['contact_flight_recovery']['body_landing']
        diagnostics.append(dict(name=name, windows=windows, actual_contact_flight=row['contact_flight_recovery'],
            samples=samples, momentum_matrix_identity_max_error=matrix_error,
            locked_inertia_symmetry_max_error=symmetry_error,
            minimum_locked_inertia_eigenvalue=float(np.linalg.eigvalsh(locked).min()),
            sampled_stability_diagnostic=sampled))
        saved[name+'_H_world']=momentum
        saved[name+'_root_rpy_rad']=euler
        saved[name+'_root_angular_body_rad_s']=obs[:,3:6]
        saved[name+'_foot_contact']=obs[:,57:61]
        saved[name+'_COM_world_m']=state['center_world']
        saved[name+'_sampled_matrix']=matrix
    saved['clock_s']=recovered[:,775]
    with (HERE/'DIAGNOSTIC_ARRAYS.npz').open('xb') as stream:
        np.savez_compressed(stream, **saved)
    sources = [HERE/'MODEL.json', HERE/'known_model.py', Path(__file__), inputs_path, data_path, pose_path,
        measured/'episodes_000.json', trace, trace_metadata, asset_manifest,
        ROOT/'quadsmash/tasks/stand/contact_flight_recovery_v5.py',
        ROOT/'quadsmash/tasks/stand/jump_metrics.py', ROOT/'quadsmash/tasks/stand/observation_contract.py',
        ROOT/'configs/evaluation_protocol_v2.json']
    record = dict(checked_at=datetime.now(timezone.utc).isoformat(), actual_public_rows=4800+800,
        actual_native_pose_rows=6*400+800, model_mass_kg=model.mass.sum(), rigid_bodies=28, joints=27,
        source_reference_checks=reference_checks, recovered_trace_check=recovered_errors,
        racket_inertia_reconstruction_error_kg_m2=inertia_error, diagnostics=diagnostics,
        raw_recovery_s_means='Longest continuous eligible stable duration, not the first recovery-completion clock.',
        source_sha256={str(p):sha(p) for p in sources}, diagnostic_arrays_sha256=sha(HERE/'DIAGNOSTIC_ARRAYS.npz'),
        new_native_execution=False, actual_PhysX_inertia_tensor_reads=False,
        new_recovery_feedback_or_closed_loop_outcome_proven=False, formal_acceptance=False, goal_complete=False)
    with (HERE/'ACTUAL_MODEL_CHECK.json').open('x') as stream:
        stream.write(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(actual_public_rows=record['actual_public_rows'],actual_pose_rows=record['actual_native_pose_rows'],
        maximum_body_position_error_m=max([recovered_errors['body_position_max_m'],*[x['body_position_max_m'] for x in reference_checks]]),
        maximum_tool_velocity_error_m_s=max([recovered_errors['tool_relative_linear_velocity_max_m_s'],*[x['tool_relative_linear_velocity_max_m_s'] for x in reference_checks]]),
        recovered_native_longest_stable_duration_s=recovered_row['contact_flight_recovery']['recovery_s'],
        sampled_recovered_stability=diagnostics[1]['sampled_stability_diagnostic'])))


if __name__ == '__main__':
    main()
