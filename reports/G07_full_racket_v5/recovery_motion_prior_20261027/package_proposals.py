"""Package a real recovery reference and explicitly unexecuted policy constants."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import torch

MAIN = Path(__file__).resolve().parents[3]
ROOT = MAIN.parent / 'unitree_smash_contact_recovery'
HERE = Path(__file__).resolve().parent
REPORTS = MAIN / 'reports/G07_full_racket_v5'
SOURCE = MAIN / 'reports/G07_full_racket_v4/gain_body8_sequential_n128_01/best_candidate.pt'
TRAIN = REPORTS / 'residual_contact_recovery_n128_updates15_02'
sys.path.insert(0, str(ROOT))
from scripts.initialize_centered_policy_v4 import same_tree


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')


def main():
    checked = read(HERE / 'CPU_PRIOR_CHECK.json')
    assert checked['candidates_checked'] == 9 and checked['factory_sha256'] == sha(HERE / 'prior_controller.py')
    assert checked['CPU_complete_recovered_trace_command_error'] <= 5e-4
    spec = importlib.util.spec_from_file_location('packaged_recovery_prior_factory', HERE / 'prior_controller.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    seed, arrays = module.load_seed()
    source = torch.load(SOURCE, map_location='cpu', weights_only=False)
    family = dict(schema='shared-recovery-prior-proposals-v1', source_public_policy=source,
        source_policy_sha256=sha(SOURCE), parameters=source['parameters'].clone(),
        asset_sha256=seed['asset_sha256'], protocol_sha256=seed['protocol_sha256'],
        seed_metadata=seed, seed_arrays={key: torch.from_numpy(value.copy()) for key, value in arrays.items()},
        candidates=module.candidates(),
        factory_source_sha256={name: sha(HERE / name) for name in ['prior_controller.py', 'proposal_io.py']},
        seed_is_actual_native_record=True, seed_complete_smash=False,
        native_execution=False, source_performance_inherited=False, formal_acceptance=False,
        kind='Unexecuted shared public-clock/observed-phase recovery prior proposals over intact measured actors')
    family_path = HERE / 'recovery_prior_proposals.pt'
    assert not family_path.exists()
    torch.save(family, family_path)
    loaded = torch.load(family_path, map_location='cpu', weights_only=False)
    assert same_tree(loaded['source_public_policy'], source)
    assert all(torch.equal(loaded['seed_arrays'][key], torch.from_numpy(value)) for key, value in arrays.items())
    assert loaded['native_execution'] is False and loaded['source_performance_inherited'] is False
    # Validate the portable constructor on actual pre-hit and post-hit inputs.
    spec = importlib.util.spec_from_file_location('portable_proposal_io_check', HERE / 'proposal_io.py')
    io_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(io_module)
    controller = io_module.make_controller(loaded, 1, 1, 'cpu')
    with torch.inference_mode():
        assert torch.equal(controller.sample(loaded['seed_arrays']['clock_s']), loaded['seed_arrays']['residual_raw'])
    files = {}
    for name in ['prior_controller.py', 'proposal_io.py', 'prepare_seed.py', 'check_prior.py',
                 'SEED_RECEIPT.json', 'CPU_PRIOR_CHECK.json', 'recorded_recovery_prior.npz']:
        files['controllers/G07_full_racket_v5/recovery_motion_prior_20261027/' + name] = HERE / name
    for folder, name in [('early_tangent_intercept_20261027', 'early_lateral_controller.py'),
                         ('origin0_measured_swing_velocity_20261027', 'swing_controller.py')]:
        files['controllers/G07_full_racket_v5/' + folder + '/' + name] = REPORTS / folder / name
    for name in ['package_proposals.py', 'replay_fresh.py']:
        files['delivery/' + name] = HERE / name
    files['policy/recovery_prior_proposals.pt'] = family_path
    files['reference/source_original_policy.pt'] = SOURCE
    files['reference/actual_recovered_trace.npz'] = TRAIN / 'trajectory_traces/lane0069_episode000000.npz'
    files['reference/actual_recovered_trace.json'] = TRAIN / 'trajectory_traces/lane0069_episode000000.json'
    files['reference/native_command_inputs.json'] = REPORTS / 'teacher_contact_recovery_cold128_02/native_command_inputs.json'
    files['reference/initial_ground_states.json'] = TRAIN / 'initial_ground_states.json'
    for name in ['INPUTS.json', 'job.json', 'completed.json', 'execution_source_hashes.json', 'execution_source_snapshot.zip']:
        files['reference/completed_training/' + name] = TRAIN / name
    records = {name: dict(bytes=path.stat().st_size, sha256=sha(path)) for name, path in files.items()}
    companion = read(REPORTS / 'contact_recovery_cold128_delivery_20261027/UPLOAD_RECEIPT.json')
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(), payloads=records,
        proposal_count=9, family_file='policy/recovery_prior_proposals.pt',
        source_policy_sha256=sha(SOURCE), family_sha256=sha(family_path),
        seed_native_execution=True, seed_air_contact=True, seed_recovery=True, seed_complete_smash=False,
        proposed_family_native_execution=False, source_performance_inherited=False,
        required_runtime_companion={key: companion[key] for key in ['asset', 'url', 'bytes', 'sha256']},
        no_extra_GPU_job_started=True, formal_cases_read=False, formal_acceptance=False, goal_complete=False)
    archive_path = HERE / 'G07_actual_recovery_reference_and_unexecuted_prior_proposals_20261027.zip'
    with zipfile.ZipFile(archive_path, 'x', zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
        for name, path in files.items():
            assert sha(path) == records[name]['sha256']
            archive.write(path, name)
        archive.writestr('MANIFEST.json', json.dumps(manifest, indent=2) + '\n')
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None and len(archive.namelist()) == len(records) + 1
        for name, record in records.items():
            data = archive.read(name)
            assert len(data) == record['bytes'] and hashlib.sha256(data).hexdigest() == record['sha256']
    result = dict(created_at=datetime.now(timezone.utc).isoformat(), archive=str(archive_path),
        bytes=archive_path.stat().st_size, sha256=sha(archive_path), members=len(records) + 1,
        payloads_SHA_verified=len(records), family_sha256=sha(family_path),
        factory_and_source_and_seed_arrays_verified_after_serialization=True,
        proposed_family_native_execution=False, source_performance_inherited=False,
        required_runtime_companion=manifest['required_runtime_companion'],
        program_sha256=sha(Path(__file__)), formal_acceptance=False, goal_complete=False)
    save(HERE / 'ARCHIVE_CHECK.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
