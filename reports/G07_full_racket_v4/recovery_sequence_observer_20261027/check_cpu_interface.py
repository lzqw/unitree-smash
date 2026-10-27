"""Check recorder immutability and serialization on numerical interface fixtures only."""
from datetime import datetime, timezone
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('read_only_recovery_recorder', HERE / 'recorder.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def tensors(env):
    groups = {'env': vars(env), 'rally': vars(env.rally), 'data': vars(env.robot.data)}
    return {group + '.' + name: value.clone() for group, entries in groups.items()
            for name, value in entries.items() if isinstance(value, torch.Tensor)}


def main():
    assert not (HERE / 'CPU_INTERFACE_CHECK.json').exists()
    n = 128
    env = SimpleNamespace(num_envs=n, device='cpu', step_dt=.01,
        episode_index=torch.zeros(n, dtype=torch.long), jump_clock=torch.full((n,), .001),
        episode_length_buf=torch.zeros(n, dtype=torch.long), observed_foot_contact=torch.ones(n, 4, dtype=torch.bool),
        _jump_physics_frame=16)
    for name in ['flight_candidate', 'flight_qualified', 'valid_jump_landed', 'airborne_seen',
                 'landed_once', 'recovery_seen', 'body_ground_seen', 'rewarded_jump', 'launch_done']:
        setattr(env, name, torch.zeros(n, dtype=torch.bool))
    for name in ['recovery_run', 'recovery_time', 'maximum_body_ground_force', 'takeoff_time',
                 'landing_time', 'flight_apex_rise', 'flight_clear_time']:
        setattr(env, name, torch.zeros(n))
    env.rally = SimpleNamespace(contact_time=torch.full((n,), float('nan')))
    for name in ['in_play', 'had_contact', 'legal_initial_contact', 'string_hit',
                 'airborne_string_hit', 'released', 'crossed_net', 'ground_seen', 'landed_target']:
        setattr(env.rally, name, torch.zeros(n, dtype=torch.bool))
    env.robot = SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.zeros(n, 3),
        root_link_lin_vel_w=torch.zeros(n, 3), root_ang_vel_w=torch.zeros(n, 3), joint_vel=torch.zeros(n, 18)))
    roll_pitch = torch.zeros(n, 2)
    rng = torch.random.get_rng_state().clone()
    before = tensors(env)
    with tempfile.TemporaryDirectory(prefix='g07_observer_cpu_interface_') as directory:
        recorder = module.RecoverySequenceRecorder(env, Path(directory) / 'fixture', maximum_samples=3)
        recorder.on_physics(env, roll_pitch)
        after = tensors(env)
        assert set(after) == set(before)
        assert all(torch.equal(torch.nan_to_num(before[k]), torch.nan_to_num(after[k])) for k in before)
        assert all(torch.equal(torch.isnan(before[k]), torch.isnan(after[k])) for k in before)
        rejected_duplicate = False
        try:
            recorder.on_physics(env, roll_pitch)
        except ValueError:
            rejected_duplicate = True
        assert rejected_duplicate and recorder.cursor == 1
        env._jump_physics_frame = 32
        env.jump_clock += .001
        env.episode_index[112] = 1  # Numerical fixture for lane filtering, not a simulated reset.
        recorder.on_physics(env, roll_pitch)
        env.episode_index[69] = 1
        env.episode_records = [dict(env_id=i, fixture_only=True) for i in [69, 112]]
        before_closure = tensors(env)
        recorder.on_episodes(env, torch.tensor([69, 112]))
        assert all(torch.equal(torch.nan_to_num(before_closure[k]), torch.nan_to_num(tensors(env)[k])) for k in before_closure)
        result = recorder.finish()
        assert not result['native_source_confirmed'] and not result['new_physics_or_rewards_supplied']
        assert [row['samples'] for row in result['episodes']] == [2, 1]
        for row in result['episodes']:
            path = Path(directory) / 'fixture' / row['data_file']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row['data_sha256']
            with np.load(path, allow_pickle=False) as arrays:
                assert tuple(arrays['columns']) == module.COLUMNS
                assert (arrays['values'][:, 0] == 0).all()
        exhausted = module.RecoverySequenceRecorder(env, Path(directory) / 'capacity', maximum_samples=1)
        exhausted.on_physics(env, roll_pitch)
        env._jump_physics_frame = 48
        try:
            exhausted.on_physics(env, roll_pitch)
        except ValueError:
            pass
        else:
            raise AssertionError('Missing buffer capacity guard')
    assert torch.equal(rng, torch.random.get_rng_state())
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), environment_tensor_values_unchanged=True,
        CPU_torch_rng_unchanged=True, duplicate_frame_rejected=True, capacity_guard_checked=True,
        first_episode_masking_and_serialization_checked=True,
        temporary_numerical_fixtures_removed=True, native_execution=False,
        scope='Numerical software interface and read-only recorder checks only; no actual physics, recovery trajectory, or performance claim.',
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), HERE / 'recorder.py']})
    with (HERE / 'CPU_INTERFACE_CHECK.json').open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
