"""Validate actual CPU evidence and a separate entry before registering a wait."""
from datetime import datetime, timezone
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

MAIN = Path(__file__).resolve().parents[3]
ROOT = MAIN.parent / 'unitree_smash_contact_recovery'
HERE = Path(__file__).resolve().parent
PROPOSALS = HERE.parent / 'early_tangent_intercept_20261027'
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    spec = importlib.util.spec_from_file_location('prepared_early_factory', PROPOSALS / 'early_lateral_controller.py')
    factory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(factory)
    cpu, proposal = [read(PROPOSALS / name) for name in ['CPU_CONTROLLER_CHECK.json', 'PROPOSAL.json']]
    assert cpu['factory_sha256'] == sha(PROPOSALS / 'early_lateral_controller.py')
    assert cpu['candidate_count'] == 9 and len(proposal['candidates']) == 7
    assert cpu['results'][0]['maximum_source_action_difference'] == 0.
    assert cpu['results'][0]['maximum_original_GPU_action_difference'] <= 5e-4
    assert cpu['all_nine_candidates_checked'] and cpu['source_actor_weights_and_normalizers_and_checkpoint_unchanged']
    assert proposal['candidates'] == [row for row in factory.candidates() if row['candidate_id'] in [0, 1, 2, 3, 5, 6, 7]]
    programs = [HERE / name for name in ['run.py', 'source_support.py', 'evaluate_early_lateral.py',
                                       'prepare_entry.py', 'check_preparation.py']]
    for path in programs:
        ast.parse(path.read_text(), filename=str(path))
    generation = read(HERE / 'ENTRY_GENERATION_CHECK.json')
    assert generation['source_sha256'] == sha(HERE.parent / 'measured_swing_queue01/evaluate_swing.py')
    assert generation['target_sha256'] == sha(HERE / 'evaluate_early_lateral.py')
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    result = subprocess.run([sys.executable, str(HERE / 'evaluate_early_lateral.py'), '--help'],
        capture_output=True, text=True, env=environment)
    assert result.returncode == 0 and '--source-evaluation' in result.stdout
    (HERE / 'native_help.txt').write_text(result.stdout)
    from source_support import verify_measured_source
    live = HERE.parent / 'measured_swing9_n128_01'
    source = torch.load(live / 'candidate_000.pt', map_location='cpu', weights_only=False)
    try:
        verify_measured_source(live / 'candidate_000.pt', source, live, ROOT,
            sha(ROOT / 'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json'))
    except FileNotFoundError as error:
        assert str(live / 'job.json') in str(error)
    else:
        raise AssertionError('Live incomplete search must not stand in for a completed cold assessment')
    record = dict(checked_at=datetime.now(timezone.utc).isoformat(),
        all_retained_candidate_definitions_match_CPU_checked_factory=True,
        retained_candidates=[0, 1, 2, 3, 5, 6, 7], all9_checked_on_actual4800_public_rows=True,
        every_native_entry_CPU_evidence_field_checked=True,
        native_entry_help_exit_code=result.returncode, all_new_programs_AST_valid=True,
        incomplete_live_search_rejected_as_completed_source=True, rejection_category='FileNotFoundError: completed job receipt absent',
        actual_selected_source_validation_and_CPU_replay_pending_parent_exit=True,
        source_sha256={str(path): sha(path) for path in [*programs,
            PROPOSALS / 'early_lateral_controller.py', PROPOSALS / 'CPU_CONTROLLER_CHECK.json', PROPOSALS / 'PROPOSAL.json']},
        new_native_execution=False, formal_acceptance=False)
    serialized = json.dumps(record, indent=2) + '\n'
    with (HERE / 'PRE_NATIVE_CPU_CHECK.json').open('x') as stream:
        stream.write(serialized)
    print(json.dumps(record))


if __name__ == '__main__':
    main()
