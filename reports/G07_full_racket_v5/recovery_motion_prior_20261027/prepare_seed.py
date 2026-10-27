"""Extract an actual recovered TRAIN trace as a bounded shared policy prior."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

MAIN = Path(__file__).resolve().parents[3]
ROOT = MAIN.parent / 'unitree_smash_contact_recovery'
HERE = Path(__file__).resolve().parent
REPORTS = MAIN / 'reports/G07_full_racket_v5'
TRAIN = REPORTS / 'residual_contact_recovery_n128_updates15_02'
TRACE = TRAIN / 'trajectory_traces/lane0069_episode000000.npz'
POLICY = MAIN / 'reports/G07_full_racket_v4/gain_body8_sequential_n128_01/best_candidate.pt'
sys.path.insert(0, str(ROOT))
from quadsmash.control.full_racket_origin0_residual_v4 import make_rally_controller
from quadsmash.control.full_racket_residual_v4 import ResidualActionMixerV4
from quadsmash.evaluation_v4 import independent_episode
from quadsmash.rl.episode_trace_v4 import unpack_public


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    text = json.dumps(value, indent=2) + '\n'
    with path.open('x') as stream:
        stream.write(text)


def main():
    assert read(TRAIN / 'job.json')['status'] == 'SUCCEEDED'
    assert read(TRAIN / 'completed.json')['env_steps'] == 245760
    inputs = read(TRAIN / 'INPUTS.json')
    assert inputs['contract']['base_policy_sha256'] == sha(POLICY)
    metadata = read(TRACE.with_suffix('.json'))
    assert metadata['native_execution'] and metadata['data_sha256'] == sha(TRACE)
    assert metadata['actor_iterations_encountered'] == list(range(7))
    assert metadata['samples'] == 800 and metadata['control_dt_s'] == .01
    protocol_path = ROOT / 'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json'
    protocol = read(protocol_path)
    protocol['_root'] = str(ROOT)
    outcome = independent_episode(metadata['raw_episode'], protocol)
    assert outcome == dict(airborne_contact=True, conditional_recovery=True, complete=False, fault=False)
    with np.load(TRACE) as data:
        public = unpack_public(data['packed_public'])
        residual, actions, bases, clock, poses = [data[key].copy() for key in
            ['residual_actions', 'native_actions', 'base_actions', 'episode_clock_s', 'link_poses']]
    assert hashlib.sha256(public.tobytes()).hexdigest() == metadata['full_public794_sha256']
    assert public.shape == (800, 794) and residual.shape == (800, 18)
    assert np.array_equal(clock, public[:, 775]) and np.isfinite(residual).all() and abs(residual).max() < 1
    assert abs(clock[0]) < 1e-8 and abs(clock[-1] - 7.99) < 1e-6
    commands = read(REPORTS / 'teacher_contact_recovery_cold128_02/native_command_inputs.json')
    default, limits = [torch.tensor(commands[name], dtype=torch.float32) for name in
                       ['default_joint_position_rad', 'joint_limits_rad']]
    base = make_rally_controller(torch.load(POLICY, map_location='cpu', weights_only=False), 1, 'cpu')
    mixer = ResidualActionMixerV4(1, 'cpu', [.15] * 18)
    base_error = action_error = 0.
    planned_event = None
    with torch.inference_mode():
        for values, request, native_base, native_action in zip(public, residual, bases, actions):
            obs = torch.from_numpy(values[None])
            calculated = base.action({'policy': obs}, default, limits)
            actual = mixer.action(calculated, torch.from_numpy(request[None]), obs, default, limits)
            base_error = max(base_error, float(abs(calculated.numpy()[0] - native_base).max()))
            action_error = max(action_error, float(abs(actual.numpy()[0] - native_action).max()))
            if bool(base.reference_controller.planned[0]):
                planned_event = float(base.reference_controller.event_time[0])
    assert action_error <= 5e-4 and base_error <= 5e-4
    assert planned_event is not None and 2.3 <= planned_event <= 3.3
    first_public_hit = float(public[np.flatnonzero(public[:, 773] > .5)[0], 775])
    # Offline causal low-pass filtering of the recorded policy requests.
    # At runtime both arrays are fixed policy constants, never future states.
    smoothed = np.empty_like(residual)
    previous = np.zeros(18, dtype=np.float32)
    for i, row in enumerate(residual):
        previous = .25 * row + .75 * previous
        smoothed[i] = previous
    table_path = HERE / 'recorded_recovery_prior.npz'
    with table_path.open('xb') as stream:
        np.savez_compressed(stream, clock_s=clock, residual_raw=residual, residual_ema025=smoothed)
    with np.load(table_path) as data:
        assert np.array_equal(data['clock_s'], clock) and np.array_equal(data['residual_raw'], residual)
    source_dir = REPORTS / 'teacher_contact_recovery_cold128_02'
    source_row = read(source_dir / 'episodes.json')[69]
    source_outcome = independent_episode(source_row, protocol)
    assert source_outcome['airborne_contact'] and not source_outcome['conditional_recovery']
    # Compare same-case measured states; this does not isolate a causal joint.
    diagnostic = REPORTS / 'measured_swing9_n128_01'
    with np.load(diagnostic / 'public_and_actions_000.npz') as data:
        lane = data['training_indices'].tolist().index(69)
        source_public = data['observations'][:, lane]
    assert np.array_equal(source_public[:, 775], clock)
    windows = []
    for lower, upper in [(2.4, 2.75), (2.75, 3.15), (3.15, 4.)]:
        use = (clock >= lower) & (clock < upper)
        windows.append(dict(from_s=lower, to_s=upper,
            failed_source_root_angular_speed_max_rad_s=float(np.linalg.norm(source_public[use, 3:6], axis=1).max()),
            recovered_trace_root_angular_speed_max_rad_s=float(np.linalg.norm(public[use, 3:6], axis=1).max()),
            failed_source_gravity_xy_norm_max=float(np.linalg.norm(source_public[use, 6:8], axis=1).max()),
            recovered_trace_gravity_xy_norm_max=float(np.linalg.norm(public[use, 6:8], axis=1).max())))
    result = dict(prepared_at=datetime.now(timezone.utc).isoformat(), schema='native-recovery-motion-prior-v1',
        table_file=table_path.name, table_sha256=sha(table_path), source_trace_sha256=sha(TRACE),
        source_policy_sha256=sha(POLICY), source_protocol_sha256=protocol['recovery_association']['source_protocol_sha256'],
        protocol_sha256=sha(protocol_path), asset_sha256=protocol['asset']['sha256'],
        samples=800, control_dt_s=.01, residual_joint_scale_rad=[.15] * 18,
        actual_reference_planned_event_time_s=planned_event,
        actual_reference_first_public_air_hit_phase_s=first_public_hit,
        public_phase773_means='Observed airborne AND contact_seen; latch once for a post-contact recovery controller. Not a persistent hit flag or a legality oracle.',
        actual_reference_outcome=outcome, actual_source_without_exploration_outcome=source_outcome,
        actual_reference_raw_episode=metadata['raw_episode'], actual_source_without_exploration_episode=source_row,
        actual_same_case_state_comparisons=windows,
        maximum_recorded_request=float(abs(residual).max()),
        leg_request_RMS=float(np.sqrt(np.mean(residual[:, :12]**2))),
        arm_request_RMS=float(np.sqrt(np.mean(residual[:, 12:]**2))),
        CPU_base_command_max_error=base_error, CPU_complete_command_max_error=action_error,
        original_action_tolerance=5e-4,
        policy_prior_is_shared_across_cases=True, runtime_case_id_input=False,
        exact_native_reexecution_not_yet_performed=True, smoothed_or_grouped_candidate_performance_unproven=True,
        complete_smash_success_claimed=False, formal_acceptance=False,
        source_sha256={str(path): sha(path) for path in [Path(__file__), TRACE, TRACE.with_suffix('.json'),
            POLICY, TRAIN / 'INPUTS.json', TRAIN / 'job.json', protocol_path,
            diagnostic / 'public_and_actions_000.npz', source_dir / 'episodes.json']})
    save(HERE / 'SEED_RECEIPT.json', result)
    print(json.dumps({key: result[key] for key in ['samples', 'table_sha256', 'actual_reference_planned_event_time_s',
        'actual_reference_first_public_air_hit_phase_s', 'CPU_complete_command_max_error',
        'leg_request_RMS', 'arm_request_RMS', 'actual_same_case_state_comparisons']}))


if __name__ == '__main__':
    main()
