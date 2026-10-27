"""Check executable queue wiring, sealed real-input evidence and CLI imports."""
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

HERE = Path(__file__).resolve().parent
MAIN = HERE.parents[2]
ROOT = MAIN.parent / 'unitree_smash_contact_recovery'
PROPOSALS = HERE.parent / 'windowed_swing_20261027'
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    cpu = json.loads((PROPOSALS / 'CPU_CONTROLLER_CHECK.json').read_text())
    proposal = json.loads((PROPOSALS / 'PROPOSAL.json').read_text())
    assert proposal['candidates'] == cpu['candidates']
    assert len(cpu['results']) == len(cpu['candidates']) == 7
    assert all(sha(path) == digest for path, digest in cpu['source_sha256'].items())
    assert cpu['results'][0]['maximum_normalized_action_change'] == 0.
    assert cpu['results'][0]['maximum_CPU_vs_original_GPU_action_difference'] <= 5e-4
    assert all(r['actual_public_rows'] == 4800 and r['pre_first_window_commands_bitwise_equal_rows'] >= 1601
        and r['maximum_arm_command_target_rate_rad_s'] <= 5.0005 for r in cpu['results'])
    generation = json.loads((HERE / 'ENTRY_GENERATION_CHECK.json').read_text())
    assert all(sha(path) == digest for group in ['source_sha256', 'generated_sha256']
               for path, digest in generation[group].items())
    names = ['run.py', 'source_support.py', 'prepare_entry.py', 'evaluate_windowed_swing.py', 'check_preparation.py']
    for name in names:
        ast.parse((HERE / name).read_text())
    queue = load('checked_window_queue', HERE / 'run.py')
    assert queue.SOURCE == HERE.parent / 'early_lateral7_n128_01/best_candidate.pt'
    assert queue.SOURCE_COLD == HERE.parent / 'early_lateral7_best_cold128_01'
    assert queue.SEARCH == HERE.parent / 'windowed_swing7_n128_01'
    assert queue.COLD == HERE.parent / 'windowed_swing7_best_cold128_01'
    assert queue.SOURCE_COLD != queue.COLD
    assert queue.PRIOR == HERE.parent / 'early_lateral_queue01'
    assert queue.factory().SCHEMA == 'g07-windowed-swing-contact-recovery-v5'
    support = load('checked_window_admission', HERE / 'source_support.py')
    live = HERE.parent / 'measured_swing9_n128_01'
    source = torch.load(live / 'candidate_000.pt', map_location='cpu', weights_only=False)
    try:
        support.verify_early_source(live / 'candidate_000.pt', source, live, ROOT,
            sha(ROOT / 'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json'))
    except AssertionError:
        pass
    else:
        raise AssertionError('Measured swing source cannot stand in for measured early-lateral source')
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    result = subprocess.run([sys.executable, str(HERE / 'evaluate_windowed_swing.py'), '--help'],
                            capture_output=True, text=True, env=environment)
    assert result.returncode == 0 and '--source-evaluation' in result.stdout, result.stderr[-2000:]
    (HERE / 'native_help.txt').write_text(result.stdout)
    record = dict(checked_at=datetime.now(timezone.utc).isoformat(),
        native_entry_help_exit_code=result.returncode, all_new_programs_AST_valid=True,
        all7_checked_on_actual4800_public_rows=True, every_native_CPU_evidence_field_checked=True,
        selected_source_cold_directory_distinct_from_future_output=True,
        actual_selected_early_source_validation_and_CPU_replay_pending_parent_exit=True,
        incompatible_measured_source_rejected=True,
        source_sha256={str(p): sha(p) for p in [*[HERE / n for n in names],
            PROPOSALS / 'window_controller.py', PROPOSALS / 'CPU_CONTROLLER_CHECK.json',
            PROPOSALS / 'PROPOSAL.json', HERE / 'ENTRY_GENERATION_CHECK.json']},
        new_native_execution=False, formal_acceptance=False)
    with (HERE / 'PRE_NATIVE_CPU_CHECK.json').open('x') as stream:
        stream.write(json.dumps(record, indent=2)+'\n')
    print(json.dumps(dict(native_entry_help_exit_code=result.returncode,
        selected_source_cold_directory_distinct_from_future_output=True,
        actual_selected_source_checks_pending=True, checked_programs=names)))


if __name__ == '__main__':
    main()
