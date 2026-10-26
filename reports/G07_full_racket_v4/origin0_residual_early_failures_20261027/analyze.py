"""Replay immutable first training failure traces and compare recorded preparation with the fixed source."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
REPORTS = ROOT / 'reports/G07_full_racket_v4'
TRAIN = REPORTS / 'origin0_residual_fresh_n128_updates15_01'
TRACE = TRAIN / 'trajectory_traces'
COLD = REPORTS / 'gain_body8_best_fixed_train128_01'
POLICY = REPORTS / 'gain_body8_sequential_n128_01/best_candidate.pt'
SELECTED = [85, 102, 39, 117, 5, 54, 0]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from quadsmash.control.full_racket_gain_body_adaptation_v4 import make_rally_controller
from quadsmash.control.full_racket_residual_v4 import ResidualActionMixerV4
from quadsmash.rl.episode_trace_v4 import unpack_public


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tilt(poses):
    quaternion = poses[:, 0, 3:7]
    return np.degrees(np.arccos(np.clip(1 - 2 * (quaternion[:, 1] ** 2 + quaternion[:, 2] ** 2), -1, 1)))


def main():
    assert Path.cwd() == ROOT and not (HERE / 'RESULT.json').exists()
    held = read(REPORTS / 'origin0_residual_after_gain_body_queue01/ADMISSION_DECISION.json')['source_sha256']
    assert all(sha(Path(p)) == s for p, s in held.items())
    manifest = read(TRACE / 'manifest.json')
    entries = {(e['env_id'], e['episode_index']): e for e in manifest['saved_episodes']}
    first_failures = [e['env_id'] for e in manifest['saved_episodes']
                      if 'first_failure_until_declared_limit' in e['selection_reasons']]
    assert first_failures == SELECTED[:6] and manifest['failure_selected'] == 6
    assert manifest['enabled'] and manifest['disabled_error'] is None
    protocol = read(ROOT / 'configs/g07_full_racket_eval_v4_r3_origin0.json')
    checkpoint = torch.load(POLICY, map_location='cpu', weights_only=False)
    assert sha(POLICY) == read(TRAIN / 'INPUTS.json')['contract']['base_policy_sha256']
    default = torch.tensor(manifest['provenance']['default_joint_position_rad'], dtype=torch.float32)
    limits = torch.tensor(manifest['provenance']['joint_limits_rad'], dtype=torch.float32)
    records, input_hashes, public_rows = [], {}, 0
    with np.load(COLD / 'all_first_episodes.npz') as cold, torch.inference_mode():
        for lane in SELECTED:
            entry = entries[lane, 0]
            path = TRACE / entry['data_file']
            metadata_path = path.with_suffix('.json')
            metadata = read(metadata_path)
            assert sha(path) == metadata['data_sha256'] == entry['data_sha256']
            with np.load(path) as z:
                arrays = {k: z[k].copy() for k in z.files}
            public = unpack_public(arrays['packed_public'])
            assert public.shape == (entry['samples'], 794)
            assert hashlib.sha256(public.tobytes()).hexdigest() == entry['full_public794_sha256']
            assert metadata['full_public_reconstruction_bitwise_verified']
            assert all(np.isfinite(v).all() for v in arrays.values())
            assert np.max(abs(arrays['native_actions'])) <= 1.000001
            assert set(np.unique(arrays['learner_iteration'])).issubset(set(range(7)))
            row = metadata['raw_episode']
            assert row['env_id'] == row['source_training_index'] == lane and row['episode_index'] == 0
            assert row['ground_init'] and not row['had_contact'] and row['numerical_failures'] == 0
            assert not metadata['actual_contact_events']
            clock = arrays['episode_clock_s']
            assert clock[0] == 0 and np.max(abs(np.diff(clock) - .01)) < 2e-6
            controller = make_rally_controller(checkpoint, 1, 'cpu')
            mixer = ResidualActionMixerV4(1, 'cpu', [.15] * 18)
            source_error, mixer_error = 0., 0.
            for obs, residual, base, actual in zip(public, arrays['residual_actions'], arrays['base_actions'], arrays['native_actions']):
                observation = torch.from_numpy(obs[None])
                expected = controller.action({'policy': observation}, default, limits).numpy()[0]
                mixed = mixer.action(torch.from_numpy(base[None]), torch.from_numpy(residual[None]), observation, default, limits).numpy()[0]
                source_error = max(source_error, float(np.max(abs(expected - base))))
                mixer_error = max(mixer_error, float(np.max(abs(mixed - actual))))
            assert source_error <= 5e-4 and mixer_error <= 5e-4
            ids = np.arange(0, len(clock), 2)
            cold_ids = np.rint(clock[ids] / .02).astype(int)
            assert cold['valid_first_episode'][cold_ids, lane].all()
            assert np.max(abs(cold['episode_clock_s'][cold_ids, lane] - clock[ids])) < 2e-6
            actual_pose = arrays['link_poses'][ids]
            source_pose = cold['link_poses'][cold_ids, lane]
            terminal = row['terminal_native_state']
            reasons = row['training_terminal_reasons']
            requested = np.clip(arrays['residual_actions'], -1, 1) * .15
            physical_applied_delta = (arrays['native_actions'] - arrays['base_actions']) * .5 * (limits[:, 1] - limits[:, 0]).numpy()
            diagnosis = dict(TRAIN_index=lane, original_episode_index=0, samples=len(clock),
                actual_terminal_time_s=terminal['time_s'], last_recorded100Hz_time_s=float(clock[-1]),
                actual_launch_seen=row['launched'], actual_contact_seen=row['had_contact'],
                actual_training_terminal_reasons=reasons,
                recorded_maximum_body_ground_force_n=row['jump']['maximum_body_ground_force_n'],
                actual_public_rows=len(public), CPU_vs_recorded_GPU_base_action_max_error=source_error,
                CPU_vs_recorded_GPU_mixed_action_max_error=mixer_error,
                requested_leg_residual_std_rad=np.std(requested[:, :12], axis=0).tolist(),
                requested_arm_residual_std_rad=np.std(requested[:, 12:], axis=0).tolist(),
                actual_leg_command_difference_max_rad=float(np.max(abs(physical_applied_delta[:, :12]))),
                actual_arm_command_difference_max_rad=float(np.max(abs(physical_applied_delta[:, 12:]))),
                actual_minimum_recorded_root_z_m=float(arrays['link_poses'][:, 0, 2].min()),
                actual_last_recorded_tilt_deg=float(tilt(arrays['link_poses'])[-1]),
                matched50Hz_samples=len(ids),
                matched50Hz_source_minimum_root_z_m=float(source_pose[:, 0, 2].min()),
                matched50Hz_training_minimum_root_z_m=float(actual_pose[:, 0, 2].min()),
                matched50Hz_source_maximum_tilt_deg=float(tilt(source_pose).max()),
                matched50Hz_training_maximum_tilt_deg=float(tilt(actual_pose).max()),
                actual_actor_iterations=entry['actor_iterations_encountered'],
                selection_reasons=entry['selection_reasons'])
            if lane != 0:
                assert not row['launched'] and terminal['time_s'] < protocol['preparation_seconds']
                assert reasons['body_ground'] or reasons['root_fallen']
            records.append(diagnosis)
            public_rows += len(public)
            for p in [path, metadata_path]:
                input_hashes[str(p)] = sha(p)
            shutil.copy2(metadata_path, HERE / metadata_path.name)
    for p in [Path(__file__), POLICY, COLD / 'all_first_episodes.npz', COLD / 'INPUTS.json',
              TRAIN / 'INPUTS.json', ROOT / 'configs/g07_full_racket_eval_v4_r3_origin0.json']:
        input_hashes[str(p)] = sha(p)
    assert all(sha(Path(p)) == s for p, s in held.items())
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), actual_selected_episode_count=7,
        first6_failure_selection_preserved=True, all6_first_failures_before_shuttle_launch=True,
        extra_predeclared_first_episode_TRAIN0=True, actual_lossless_public_rows=public_rows,
        original_action_tolerance=5e-4, diagnoses=records, input_sha256=input_hashes,
        all519_admitted_inputs_unchanged=True,
        scope='Actual early stochastic TRAIN traces under critic warmup; selected failures are not a rate estimate. Nominal cold comparison shares clocks, not observations or full physical histories.',
        no_causal_noise_or_controller_bug_proven=True, new_native_execution=False,
        formal_cases_read=False, formal_acceptance=False)
    with (HERE / 'RESULT.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(episodes=7, actual_public_rows=public_rows,
        maximum_source_action_error=max(d['CPU_vs_recorded_GPU_base_action_max_error'] for d in records),
        maximum_mixed_action_error=max(d['CPU_vs_recorded_GPU_mixed_action_max_error'] for d in records),
        first6_before_launch=True)), flush=True)


if __name__ == '__main__':
    main()
