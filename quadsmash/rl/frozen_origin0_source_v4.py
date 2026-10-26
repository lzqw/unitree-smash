"""Require a completed original TRAIN128 cold assessment before fresh learning."""
import hashlib
import json
from pathlib import Path

from quadsmash.control.full_racket_origin0_residual_v4 import validate_frozen_baseline
from quadsmash.control.full_racket_train_evaluation_v4 import training_cases, summarize_training


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_frozen_origin0_source(path, checkpoint, evaluation, root, protocol_sha256):
    root, path, evaluation = Path(root), Path(path), Path(evaluation)
    validate_frozen_baseline(checkpoint)
    read = lambda name:json.loads((evaluation/name).read_text())
    inputs, metrics, job, completed = [read(name) for name in ['INPUTS.json', 'metrics.json', 'job.json', 'completed.json']]
    if not (job['status'] == 'SUCCEEDED' and job['exit_code'] == 0 and job['receipt_present']
            and job['charged_env_steps'] == 102400 and 0 < completed['env_steps'] <= 102400):
        raise ValueError('Require the successful complete native cold128 job and bounded actual steps')
    if not (inputs['protocol_sha256'] == metrics['protocol_sha256'] == checkpoint['protocol_sha256'] == protocol_sha256
            and inputs['policy_sha256'] == metrics['policy_sha256'] == sha(path)):
        raise ValueError('Source/evaluation weights or numerical protocol differ')
    if not (inputs['training_indices'] == list(range(128)) and inputs['formal_cases_read'] is False
            and inputs.get('origin_spacing_m') == 0. and metrics['noise_level'] == 'none'
            and all(value == 0 for value in inputs['reset_noise_amplitudes'].values())):
        raise ValueError('Require the full original128 zero-noise origin0 TRAIN assessment')
    origins = read('origin_layout.json')['actual_environment_origins_m']
    if len(origins) != 128 or any(row != [0., 0., 0.] for row in origins):
        raise ValueError('Actual evaluated origins differ from declared origin0')
    protocol_path = root/'configs/g07_full_racket_eval_v4_r3_origin0.json'
    if sha(protocol_path) != protocol_sha256:
        raise ValueError('Origin0 protocol changed')
    protocol = json.loads(protocol_path.read_text()); protocol['_root'] = str(root)
    if sha(root/protocol['asset']['path']) != protocol['asset']['sha256'] or checkpoint['asset_sha256'] != protocol['asset']['sha256']:
        raise ValueError('Full physical asset differs')
    for name, digest in protocol['physical_configs'].items():
        if sha(root/name) != digest:
            raise ValueError('Physical configuration changed: '+name)
    for name, digest in read('source_hashes.json').items():
        if name.startswith('quadsmash/') and sha(root/name) != digest:
            raise ValueError('Measured source code changed: '+name)
    cases, indices = training_cases(root, protocol)
    summary = summarize_training(read('episodes.json'), cases, indices, protocol)
    if summary['numerical_failures'] or any(metrics[key] != value for key, value in summary.items()):
        raise ValueError('Independent original128 assessment disagrees or contains numerical faults')
    if metrics['actual_control_env_steps'] != completed['env_steps']:
        raise ValueError('Actual evaluation step accounting differs')
    names = ['INPUTS.json', 'metrics.json', 'episodes.json', 'job.json', 'completed.json', 'source_hashes.json', 'origin_layout.json']
    return dict(source_policy_sha256=sha(path), source_evaluation=str(evaluation),
        source_original_TRAIN128_summary=summary, source_evidence_sha256={name:sha(evaluation/name) for name in names},
        scope='Completed native origin0 cold128 provenance only; fresh learner inherits no task score.',
        both_fixed_source_actors_preserved=True, formal_cases_read=False, formal_acceptance=False)
