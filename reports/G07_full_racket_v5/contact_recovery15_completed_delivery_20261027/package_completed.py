"""Seal actually completed recovery-v5 learning and cold assessment, without live jobs."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import zipfile

MAIN = Path(__file__).resolve().parents[3]
ROOT = MAIN.parent / 'unitree_smash_contact_recovery'
HERE = Path(__file__).resolve().parent
REPORTS = MAIN / 'reports/G07_full_racket_v5'
TRAIN = REPORTS / 'residual_contact_recovery_n128_updates15_02'
COLD = REPORTS / 'residual_contact_recovery15_cold128_02'
OWNER = REPORTS / 'contact_recovery_queue02'
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from quadsmash.control.full_racket_train_evaluation_v4 import training_cases, summarize_training
from quadsmash.evaluation_v4 import independent_episode
from scripts.initialize_centered_policy_v4 import same_tree


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def main():
    assert read(OWNER / 'state.json')['state'] == 'COMPLETED_REVISED_SOURCE_TRAIN15_AND_COLD128'
    assert not Path('/proc/1939834').exists()
    protocol_path = ROOT / 'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json'
    protocol = read(protocol_path)
    protocol['_root'] = str(ROOT)
    cases, indices = training_cases(ROOT, protocol)
    summary = summarize_training(read(COLD / 'episodes.json'), cases, indices, protocol)
    assert all(read(COLD / 'metrics.json')[key] == value for key, value in summary.items())
    assert summary == read(OWNER / 'CONTACT_RECOVERY15_COLD128_RESULT.json')
    policy = torch.load(TRAIN / 'policy.pt', map_location='cpu', weights_only=False)
    source_path = MAIN / 'reports/G07_full_racket_v4/gain_body8_sequential_n128_01/best_candidate.pt'
    assert same_tree(policy['base_controller'], torch.load(source_path, map_location='cpu', weights_only=False))
    checkpoint = torch.load(TRAIN / 'checkpoint.pt', map_location='cpu', weights_only=False)
    assert checkpoint['iter'] == 15 and checkpoint['native']['canonical_restart']
    assert checkpoint['native']['total_timesteps'] == 245760
    assert not checkpoint['native']['environment']['tensors']['episode_length_buf'].any()
    assert checkpoint['infos']['residual_contract']['protocol_sha256'] == sha(protocol_path)
    assert read(TRAIN / 'canonical_contact_flight_state.json')['all_contact_flight_buffers_zero']
    assert read(TRAIN / 'canonical_contact_flight_state.json')['checkpoint_sha256'] == sha(TRAIN / 'checkpoint.pt')
    model = checkpoint['model_state_dict']
    assert int(model['actor_obs_normalizer.count']) == int(model['critic_obs_normalizer.count']) == 245760
    assert all(torch.isfinite(value).all() for value in model.values() if torch.is_tensor(value))
    seen, outcomes = [0] * 128, []
    rows = read(TRAIN / 'episodes.json')
    for row in rows:
        lane = row['env_id']
        assert row['ground_init'] and row['source_training_index'] == lane and row['episode_index'] == seen[lane]
        seen[lane] += 1
        outcomes.append(independent_episode(row, protocol))
    assert seen == checkpoint['native']['environment']['tensors']['episode_index'].tolist()
    counts = dict(completed_episodes=len(rows), air=sum(row['airborne_contact'] for row in outcomes),
        conditional_recovery_count=sum(row['conditional_recovery'] for row in outcomes),
        complete=sum(row['complete'] for row in outcomes), numerical_failures=sum(row['numerical_failures'] for row in rows))
    assert counts == read(TRAIN / 'metrics.json')['counts']
    assert counts == dict(completed_episodes=738, air=8, conditional_recovery_count=1, complete=0, numerical_failures=0)
    commands = read(OWNER / 'CONTACT_RECOVERY15_COLD128_CPU_COMMANDS.json')
    assert commands['policy_sha256'] == sha(TRAIN / 'policy.pt')
    assert commands['CPU_vs_actual_GPU_max_action_error'] <= commands['original_tolerance'] == 5e-4
    inputs = read(COLD / 'INPUTS.json')
    assert inputs['policy_sha256'] == read(COLD / 'metrics.json')['policy_sha256'] == sha(TRAIN / 'policy.pt')
    assert inputs['training_indices'] == list(range(128)) and not any(inputs['reset_noise_amplitudes'].values())
    assert inputs['reset_noise_seed'] == 724201 and inputs['protocol_sha256'] == sha(protocol_path)
    assert read(COLD / 'origin_layout.json')['actual_environment_origins_m'] == [[0., 0., 0.]] * 128
    with np.load(COLD / 'all_first_episodes.npz') as data:
        assert data['link_poses'].shape == (400, 128, 28, 7)
        assert all(np.isfinite(data[key]).all() for key in data.files)
    trace_check = read(HERE / 'ACTUAL_TRAINING_TRACE_AUDIT.json')
    assert trace_check['all_selected_trajectories_checked'] and trace_check['selected_episodes_checked'] == 14
    gpu_hours, snapshot_counts = 0., {}
    for run, cap in [(TRAIN, 245760), (COLD, 102400)]:
        job = read(run / 'job.json')
        assert job['status'] == 'SUCCEEDED' and job['exit_code'] == 0 and job['receipt_present'] and not job['timed_out']
        assert job['charged_env_steps'] == read(run / 'completed.json')['env_steps'] == cap
        assert not Path('/proc', str(job['pid'])).exists() and not Path('/proc', str(job['child_pid'])).exists()
        gpu_hours += job['charged_gpu_hours']
        for label in ['source', 'execution_source']:
            hashes = read(run / (label + '_hashes.json'))
            with zipfile.ZipFile(run / (label + '_snapshot.zip')) as zipped:
                assert zipped.testzip() is None and set(zipped.namelist()) == set(hashes)
                for name, digest in hashes.items():
                    assert hashlib.sha256(zipped.read(name)).hexdigest() == digest
                    if label == 'execution_source':
                        assert sha(ROOT / name) == digest
            snapshot_counts[run.name + '/' + label] = len(hashes)
    checked = dict(checked_at=datetime.now(timezone.utc).isoformat(), actual_completed_updates=15,
        actual_training_control_env_steps=245760, independently_checked_changing_policy_counts=counts,
        actual_cold_control_env_steps=102400, original128_cold_summary=summary,
        actual_combined_GPU_hours=gpu_hours, all128_training_episode_chains_checked=True,
        canonical_checkpoint_sha256=sha(TRAIN / 'checkpoint.pt'), actual_policy_sha256=sha(TRAIN / 'policy.pt'),
        canonical_contact_flight_buffers_zero=True, embedded_original_source_tree_unchanged=True,
        actual_valid_cold_public_rows=commands['actual_valid_public_rows'],
        CPU_vs_actual_cold_GPU_max_action_error=commands['CPU_vs_actual_GPU_max_action_error'],
        original_action_tolerance=5e-4, actual_training_traces_checked=14,
        source_snapshot_member_counts=snapshot_counts, new_policy_not_promoted=True,
        previous_goal_turn_classification='progress: delivered verified proportional historical video1ed3a47',
        current_turn_progress='completed revised15 and independent128 evidence; new measured swing9 running separately',
        formal_cases_read=False, formal_acceptance=False, goal_complete=False)
    save(HERE / 'ACTUAL_COMPLETED_DATA_CHECK.json', checked)
    payload, omitted = {}, []
    for label, run in [('training15', TRAIN), ('cold128', COLD)]:
        for path in sorted(run.rglob('*')):
            if not path.is_file() or path.suffix in ['.tmp', '.pyc', '.lock'] or '__pycache__' in path.parts:
                continue
            if '.partial.' in path.name:
                final = path.with_name(path.name.replace('.partial.', '.'))
                if final.is_file() and sha(path) == sha(final):
                    omitted.append(str(path))
                    continue
            payload[label + '/' + str(path.relative_to(run))] = path
    for name in ['run.py', 'PLAN.json', 'ACTUAL_REVISED_SOURCE_VERIFICATION.json',
                 'ACTUAL_TRAINING_RECEIPT.json', 'TEACHER_REVISED_COLD128_RESULT.json',
                 'TEACHER_REVISED_COLD128_CPU_COMMANDS.json', 'CONTACT_RECOVERY15_COLD128_RESULT.json',
                 'CONTACT_RECOVERY15_COLD128_CPU_COMMANDS.json', 'state.json']:
        payload['coordinator/' + name] = OWNER / name
    for name in ['package_completed.py', 'REPRODUCTION.md', 'check_actual_traces.py',
                 'ACTUAL_TRAINING_TRACE_AUDIT.json', 'ACTUAL_COMPLETED_DATA_CHECK.json']:
        payload['delivery/' + name] = HERE / name
    records = {name: dict(bytes=path.stat().st_size, sha256=sha(path)) for name, path in payload.items()}
    companion = read(REPORTS / 'contact_recovery_cold128_delivery_20261027/UPLOAD_RECEIPT.json')
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(), payloads=records,
        actual_training_counts=counts, original128_cold_summary=summary,
        current_policy_file='training15/policy.pt', canonical_checkpoint_file='training15/checkpoint.pt',
        current_policy_sha256=sha(TRAIN / 'policy.pt'), current_policy_not_promoted=True,
        required_runtime_companion={k: companion[k] for k in ['asset', 'url', 'bytes', 'sha256']},
        omitted_byte_identical_partial_files=omitted,
        excluded_live_job='measured_swing9_n128_01 and all subsequent candidate outcomes',
        fresh_native_replay_claimed=False, formal_cases_read=False, formal_acceptance=False)
    archive_path = HERE / 'G07_contact_recovery_v5_completed15_training_and_independent128_20261027.zip'
    with zipfile.ZipFile(archive_path, 'x', zipfile.ZIP_DEFLATED, compresslevel=4) as zipped:
        for name, path in payload.items():
            assert sha(path) == records[name]['sha256']
            zipped.write(path, name)
        zipped.writestr('MANIFEST.json', json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    with zipfile.ZipFile(archive_path) as zipped:
        assert zipped.testzip() is None and len(zipped.namelist()) == len(records) + 1
        for name, value in records.items():
            content = zipped.read(name)
            assert len(content) == value['bytes'] and hashlib.sha256(content).hexdigest() == value['sha256']
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), archive=str(archive_path),
        bytes=archive_path.stat().st_size, sha256=sha(archive_path), members=len(records) + 1,
        payloads_SHA_verified=len(records), all_CRC_passed=True,
        required_runtime_companion=manifest['required_runtime_companion'],
        formal_acceptance=False, goal_complete=False)
    save(HERE / 'ARCHIVE_CHECK.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
