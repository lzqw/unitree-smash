"""Validate the real CPU evidence and separate native/queue wiring."""
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
PROPOSALS = HERE.parent / 'recovery_prior_native_20261027'
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
    assert len(cpu['results']) == len(cpu['candidates']) == 9
    assert cpu['factory_sha256'] == sha(PROPOSALS / 'prior_controller.py')
    assert cpu['results'][0]['maximum_source_action_difference'] == 0.
    assert cpu['results'][0]['maximum_original_GPU_action_difference'] <= 5e-4
    assert cpu['all9_action_streams_match_published_portable_proposals_bitwise']
    assert cpu['source_actor_and_normalizer_and_checkpoint_tree_unchanged']
    assert all(sha(path) == digest for path, digest in cpu['source_sha256'].items())
    assert all(r['actual_public_rows'] == 4800 and r['maximum_arm_target_rate_rad_s'] <= 5.0005
               and r['entire_stream_matches_previously_downloaded_family'] for r in cpu['results'])
    generation = json.loads((HERE / 'ENTRY_GENERATION_CHECK.json').read_text())
    assert all(sha(path) == digest for group in ['source_sha256', 'generated_sha256']
               for path, digest in generation[group].items())
    assert generation['generator_sha256'] == sha(HERE / 'prepare_entry.py')
    names = ['run.py', 'source_support.py', 'prepare_entry.py', 'evaluate_recovery_prior.py', 'check_preparation.py']
    for name in names:
        ast.parse((HERE / name).read_text())
    queue = load('checked_recovery_queue', HERE / 'run.py')
    assert queue.PRIOR == HERE.parent / 'windowed_swing_queue01'
    assert queue.SOURCE == HERE.parent / 'windowed_swing7_n128_01/best_candidate.pt'
    assert queue.SOURCE_COLD == HERE.parent / 'windowed_swing7_best_cold128_01'
    assert queue.SEARCH == HERE.parent / 'recovery_prior9_n128_01'
    assert queue.COLD == HERE.parent / 'recovery_prior9_best_cold128_01'
    assert queue.SOURCE_COLD != queue.COLD
    factory = queue.factory()
    assert factory.SCHEMA == 'g07-shared-native-recovery-prior-v5' and factory.candidates() == proposal['candidates']
    seed, arrays = factory.load_seed()
    assert seed['table_sha256'] == proposal['seed_table_sha256']
    assert arrays['residual_raw'].shape == (800, 18)
    # Execute the native main's actual CPU preconditions with the actual
    # evidence dictionaries; parsing alone cannot catch mismatched keys.
    entry_tree = ast.parse((HERE / 'evaluate_recovery_prior.py').read_text())
    main_node = next(n for n in entry_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    checked_assertions = 0
    scope = dict(cpu=cpu, proposal=proposal, sha=sha, PROPOSALS=PROPOSALS)
    for node in main_node.body:
        if isinstance(node, ast.Assert) and any(isinstance(n, ast.Name) and n.id == 'cpu' for n in ast.walk(node)):
            assert eval(compile(ast.Expression(node.test), str(HERE / 'evaluate_recovery_prior.py'), 'eval'), scope)
            checked_assertions += 1
    assert checked_assertions == 6
    support = load('checked_recovery_admission', HERE / 'source_support.py')
    live = HERE.parent / 'measured_swing9_n128_01'
    source = torch.load(live / 'candidate_000.pt', map_location='cpu', weights_only=False)
    try:
        support.verify_window_source(live / 'candidate_000.pt', source, live, ROOT,
            sha(ROOT / 'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json'))
    except AssertionError:
        pass
    else:
        raise AssertionError('An incomplete measured-swing source cannot replace completed windowed-source cold128')
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    result = subprocess.run([sys.executable, str(HERE / 'evaluate_recovery_prior.py'), '--help'],
                            capture_output=True, text=True, env=environment)
    assert result.returncode == 0 and '--source-evaluation' in result.stdout, result.stderr[-2000:]
    (HERE / 'native_help.txt').write_text(result.stdout)
    record = dict(checked_at=datetime.now(timezone.utc).isoformat(), native_entry_help_exit_code=result.returncode,
        all_new_programs_AST_valid=True, actual_native_CPU_preconditions_evaluated=checked_assertions,
        all9_checked_on_actual4800_public_rows=True, actual_window_source_checks_pending_parent_exit=True,
        all9_commands_bitwise_match_published_reference=True,
        selected_source_cold_directory_distinct_from_future_output=True,
        incompatible_measured_source_rejected=True,
        source_sha256={str(p): sha(p) for p in [*[HERE / n for n in names],
            PROPOSALS / 'prior_controller.py', PROPOSALS / 'CPU_CONTROLLER_CHECK.json',
            PROPOSALS / 'PROPOSAL.json', HERE / 'ENTRY_GENERATION_CHECK.json']},
        new_native_execution=False, formal_acceptance=False)
    with (HERE / 'PRE_NATIVE_CPU_CHECK.json').open('x') as stream:
        stream.write(json.dumps(record, indent=2)+'\n')
    print(json.dumps(dict(native_entry_help_exit_code=result.returncode,
        all9_action_streams_identical=True, native_CPU_preconditions_evaluated=checked_assertions,
        actual_selected_source_checks_pending=True)))


if __name__ == '__main__':
    main()
