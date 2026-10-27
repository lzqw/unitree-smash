"""Actual-source snapshots and cold-source checks for explicit recovery revision."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

from quadsmash.control.full_racket_origin0_residual_v4 import validate_frozen_baseline
from quadsmash.control.full_racket_train_evaluation_v4 import training_cases, summarize_training
from quadsmash.tasks.stand.contact_recovery_origin0_v5 import PROTOCOL, validate_recovery_protocol


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot_execution_sources(root, output):
    root, output = Path(root), Path(output)
    files = sorted([*root.glob('quadsmash/**/*.py'), *root.glob('scripts/*.py'),
                    *root.glob('configs/*.json'), *root.glob('assets/*.usd')])
    hashes = {str(path.relative_to(root)): sha(path) for path in files}
    with (output / 'execution_source_hashes.json').open('x') as stream:
        json.dump(hashes, stream, indent=2)
        stream.write('\n')
    with zipfile.ZipFile(output / 'execution_source_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, str(path.relative_to(root)))
    return dict(execution_worktree=str(root),
                execution_git_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                execution_source_manifest_sha256=sha(output / 'execution_source_hashes.json'),
                execution_source_files=len(files),
                guard_source_snapshot_scope='Main scheduler and ledger tree; this separate execution snapshot contains actual imported revision.')


def verify_recovery_source(path, checkpoint, evaluation, root, protocol_sha256):
    root, path, evaluation = Path(root), Path(path), Path(evaluation)
    validate_frozen_baseline(checkpoint)
    protocol_path = root / PROTOCOL
    if sha(protocol_path) != protocol_sha256:
        raise ValueError('Recovery protocol changed')
    protocol = json.loads(protocol_path.read_text())
    original = validate_recovery_protocol(protocol)
    if checkpoint['protocol_sha256'] != protocol['recovery_association']['source_protocol_sha256']:
        raise ValueError('Require the unchanged original-policy source, not relabeled historical weights')
    read = lambda name: json.loads((evaluation / name).read_text())
    inputs, metrics, job, completed = [read(name) for name in ['INPUTS.json', 'metrics.json', 'job.json', 'completed.json']]
    if not (job['status'] == 'SUCCEEDED' and job['exit_code'] == 0 and job['receipt_present']
            and job['charged_env_steps'] == 102400 and completed['env_steps'] == 102400):
        raise ValueError('Require the actual full128 cold assessment under the new association')
    if not (inputs['protocol_sha256'] == metrics['protocol_sha256'] == protocol_sha256
            and inputs['policy_sha256'] == metrics['policy_sha256'] == sha(path)):
        raise ValueError('Actual revised-source assessment weights or protocol differ')
    if not (inputs['source_checkpoint_protocol_sha256'] == checkpoint['protocol_sha256']
            and inputs['training_indices'] == list(range(128)) and inputs['formal_cases_read'] is False
            and inputs['origin_spacing_m'] == 0. and metrics['noise_level'] == 'none'
            and not any(inputs['reset_noise_amplitudes'].values())):
        raise ValueError('Require all original128, zero-noise, origin0 cases')
    if read('origin_layout.json')['actual_environment_origins_m'] != [[0., 0., 0.]] * 128:
        raise ValueError('Actual evaluated origins differ')
    for name, digest in read('execution_source_hashes.json').items():
        if name.startswith('quadsmash/') and sha(root / name) != digest:
            raise ValueError('Measured revision code changed: ' + name)
    if checkpoint['asset_sha256'] != original['asset']['sha256'] or sha(root / original['asset']['path']) != original['asset']['sha256']:
        raise ValueError('Physical asset changed')
    for name, digest in original['physical_configs'].items():
        if sha(root / name) != digest:
            raise ValueError('Physical config changed: ' + name)
    protocol['_root'] = str(root)
    cases, indices = training_cases(root, protocol)
    summary = summarize_training(read('episodes.json'), cases, indices, protocol)
    if summary['numerical_failures'] or any(metrics[key] != value for key, value in summary.items()):
        raise ValueError('Independent revised128 assessment disagrees')
    if metrics['actual_control_env_steps'] != completed['env_steps']:
        raise ValueError('Actual assessment steps differ')
    names = ['INPUTS.json', 'metrics.json', 'episodes.json', 'job.json', 'completed.json',
             'execution_source_hashes.json', 'origin_layout.json']
    return dict(source_policy_sha256=sha(path), source_evaluation=str(evaluation),
                original_source_protocol_sha256=checkpoint['protocol_sha256'],
                new_evaluation_protocol_sha256=protocol_sha256,
                source_original_TRAIN128_summary=summary,
                source_evidence_sha256={name: sha(evaluation / name) for name in names},
                source_weights_rewritten=False, formal_cases_read=False, formal_acceptance=False,
                both_fixed_source_actors_preserved=True)
