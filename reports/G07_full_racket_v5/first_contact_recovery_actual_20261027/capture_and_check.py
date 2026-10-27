"""Freeze actual first-cohort traces and check the corrected recovery event."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
WORK = ROOT.parent / 'unitree_smash_contact_recovery'
RUN = ROOT / 'reports/G07_full_racket_v5/residual_contact_recovery_n128_updates15_02'
OLD = ROOT / 'reports/G07_full_racket_v4/origin0_residual_fresh_n128_updates15_01'
sys.path.insert(0, str(WORK))
from quadsmash.evaluation_v4 import independent_episode
from quadsmash.rl.episode_trace_v4 import unpack_public
from quadsmash.control.full_racket_origin0_residual_v4 import make_rally_controller
from quadsmash.control.full_racket_residual_v4 import ResidualActionMixerV4


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    snapshot = HERE / 'snapshot'
    snapshot.mkdir()
    source_rows = json.loads((RUN / 'episodes.partial.json').read_text())
    first = {row['env_id']: row for row in source_rows if row['episode_index'] == 0}
    assert set(first) == set(range(128))
    old_rows = json.loads((ROOT / 'reports/G07_full_racket_v4/origin0_residual_warmup7_actor8_20261027/FIRST128_EPISODES.json').read_text())
    changed = {}
    for index in range(128):
        row = {key: value for key, value in first[index].items() if key != 'contact_flight_recovery'}
        differing = [key for key in row if json.dumps(row[key], sort_keys=True) != json.dumps(old_rows[index][key], sort_keys=True)]
        if differing:
            changed[index] = differing
    assert changed == {69: ['recovery_success']}, changed
    protocol = json.loads((WORK / 'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json').read_text())
    protocol['_root'] = str(WORK)
    outcomes = [independent_episode(first[index], protocol) for index in range(128)]
    assert sum(row['airborne_contact'] for row in outcomes) == 2
    assert sum(row['conditional_recovery'] for row in outcomes) == 1
    assert not any(row['complete'] for row in outcomes)
    with (snapshot / 'FIRST128_EPISODES.json').open('x') as stream:
        json.dump([first[index] for index in range(128)], stream, indent=2)
        stream.write('\n')
    for name in ['geometry.json', 'geometry.npz', 'initial_ground_states.json', 'INPUTS.json', 'fresh_initial_model_state.pt']:
        shutil.copyfile(RUN / name, snapshot / name)
    shutil.copyfile(RUN / 'ppo/model_6.pt', snapshot / 'warmup7_diagnostic_model.pt')
    original_initial = torch.load(OLD / 'fresh_initial_model_state.pt', map_location='cpu', weights_only=False)
    initial = torch.load(snapshot / 'fresh_initial_model_state.pt', map_location='cpu', weights_only=False)
    diagnostic = torch.load(snapshot / 'warmup7_diagnostic_model.pt', map_location='cpu', weights_only=False)
    assert all(torch.equal(value, initial['model_state_dict'][key]) for key, value in original_initial['model_state_dict'].items())
    assert all(torch.equal(value, diagnostic['model_state_dict'][key]) for key, value in initial['model_state_dict'].items()
               if key.startswith('actor.') or key == 'log_std')
    assert diagnostic['native']['canonical_restart'] is False
    common = sorted({path.name for path in (OLD / 'trajectory_traces').glob('*.npz')} &
                    {path.name for path in (RUN / 'trajectory_traces').glob('*.npz')})
    assert len(common) == 12
    different_traces = []
    for name in common:
        with np.load(OLD / 'trajectory_traces' / name) as a, np.load(RUN / 'trajectory_traces' / name) as b:
            if set(a.files) != set(b.files) or any(not np.array_equal(a[key], b[key]) for key in a.files):
                different_traces.append(name)
    assert not different_traces
    source_policy = ROOT / 'reports/G07_full_racket_v4/gain_body8_sequential_n128_01/best_candidate.pt'
    policy = torch.load(source_policy, map_location='cpu', weights_only=False)
    metadata = json.loads((ROOT / 'reports/G07_full_racket_v5/teacher_contact_recovery_cold128_02/native_command_inputs.json').read_text())
    defaults, limits_value = metadata['default_joint_position_rad'], metadata['joint_limits_rad']
    default = torch.tensor(defaults, dtype=torch.float32)
    limits = torch.tensor(limits_value, dtype=torch.float32)
    checks = []
    for lane in [69, 112, 0]:
        name = f'lane{lane:04d}_episode000000'
        meta = json.loads((RUN / 'trajectory_traces' / (name + '.json')).read_text())
        data = RUN / 'trajectory_traces' / meta['data_file']
        assert sha(data) == meta['data_sha256']
        shutil.copyfile(data, snapshot / data.name)
        shutil.copyfile(RUN / 'trajectory_traces' / (name + '.json'), snapshot / (name + '.json'))
        controller = make_rally_controller(policy, 1, 'cpu')
        mixer = ResidualActionMixerV4(1, 'cpu', [.15] * 18)
        with np.load(snapshot / data.name) as arrays:
            public = unpack_public(arrays['packed_public'])
            assert hashlib.sha256(public.tobytes()).hexdigest() == meta['full_public794_sha256']
            max_base = max_native = 0.
            for observation, residual, base, native in zip(public, arrays['residual_actions'], arrays['base_actions'], arrays['native_actions']):
                obs = torch.from_numpy(observation[None])
                with torch.inference_mode():
                    calculated = controller.action({'policy': obs}, default, limits)
                    action = mixer.action(calculated, torch.from_numpy(residual[None]), obs, default, limits)
                max_base = max(max_base, float(abs(calculated.numpy()[0] - base).max()))
                max_native = max(max_native, float(abs(action.numpy()[0] - native).max()))
            assert max_base <= 5e-4 and max_native <= 5e-4
        checks.append(dict(env_id=lane, samples=meta['samples'], data_sha256=meta['data_sha256'],
            outcome=independent_episode(meta['raw_episode'], protocol), CPU_base_max_error=max_base,
            CPU_complete_command_max_error=max_native, public_reconstruction_bitwise_verified=True,
            actual_actor_iterations=meta['actor_iterations_encountered']))
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), actual_first128_complete=True,
        first128_airborne_contacts=2, first128_conditional_recovery=1, first128_complete_smash=0,
        original_first128_only_changed_fields=changed, original_initial_model_tensors_identical=True,
        warmup7_actor_and_log_std_unchanged=True, warmup7_checkpoint_is_canonical=False,
        common12_actual_trace_arrays_bitwise_identical=True, trace_checks=checks,
        corrected_native_TRAIN69=first[69], actor_learning_improvement_proven=False,
        initial_source_policy_sha256=sha(source_policy), source_script_sha256=sha(Path(__file__)),
        formal_cases_read=False, formal_acceptance=False, goal_complete=False)
    with (HERE / 'ACTUAL_NATIVE_AND_COMMAND_CHECK.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(first128_airborne_contacts=2, first128_recovery=1,
        first128_complete=0, only_changed_fields=changed, trace_checks=checks)))


if __name__ == '__main__':
    main()
