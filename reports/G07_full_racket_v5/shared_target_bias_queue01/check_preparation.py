"""Check the preserved native loop, fail-closed gates and real-source admission."""
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = HERE.parents[2].parent/'unitree_smash_contact_recovery'
sys.path.insert(0, str(ROOT))
from source_support import STAGES, select_completed_stage, stage_paths, verify_selected_source, replay_selected_public


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def node_dump(tree, name):
    nodes = [x for x in ast.walk(tree) if isinstance(x, ast.FunctionDef) and x.name == name]
    assert len(nodes) == 1
    return ast.dump(nodes[0], include_attributes=False)


def tick_dump(tree):
    nodes = [x for x in ast.walk(tree) if isinstance(x, ast.For)
             and isinstance(x.target, ast.Name) and x.target.id == 'tick']
    assert len(nodes) == 1
    return ast.dump(nodes[0], include_attributes=False)


def main():
    import torch
    torch.set_num_threads(2)
    entry, prior = HERE/'evaluate_shared_target.py', BASE/'stance_recovery_queue01/evaluate_stance_recovery.py'
    original, derivative = [ast.parse(p.read_text()) for p in [prior, entry]]
    assert node_dump(original, 'generation_summary') == node_dump(derivative, 'generation_summary')
    assert tick_dump(original) == tick_dump(derivative)
    wanted = ['contact_recovery_origin0_cfg', 'FrozenPanelEnv', 'training_cases', 'training_noise',
              'snapshot_execution_sources', 'audit_composed_asset']
    for name in wanted:
        def calls(tree):
            return [ast.dump(x, include_attributes=False) for x in ast.walk(tree) if isinstance(x, ast.Call)
                    and isinstance(x.func, ast.Name) and x.func.id == name]
        assert calls(original) == calls(derivative), name
    generation = read(HERE/'ENTRY_GENERATION_CHECK.json')
    assert all(sha(Path(p)) == h for p, h in generation['source_sha256'].items())
    assert all(sha(Path(p)) == h for p, h in generation['generated_sha256'].items())
    # These strings test the pure source-selection gate only. They are not
    # fabricated rollout files, checkpoints or performance evidence.
    gate_checks = []
    for choose in range(len(STAGES)):
        states = {s[0]: {'state': ('SKIPPED_PARENT_SOURCE_STAGE' if i < choose else s[1])}
                  for i, s in enumerate(STAGES)}
        assert select_completed_stage(states) == STAGES[choose]
        gate_checks.append(dict(gate='legitimate_skip_falls_back_to_real_completed_stage', stage=STAGES[choose][0]))
    for status in ['FAILED_PRESERVED_FOR_REVIEW', 'WAITING_FOR_PARENT', 'RUNNING_SEARCH']:
        states = {s[0]: {'state': s[1]} for s in STAGES}
        states[STAGES[0][0]] = {'state': status}
        try:
            select_completed_stage(states)
        except ValueError:
            gate_checks.append(dict(gate='reject_nonterminal_or_failed_predecessor', status=status))
        else:
            raise AssertionError(status)
    try:
        select_completed_stage({s[0]: {'state': 'SKIPPED_PARENT_SOURCE_STAGE'} for s in STAGES})
    except ValueError:
        gate_checks.append(dict(gate='reject_no_completed_real_source'))
    else:
        raise AssertionError('All-skipped chain must not create a policy')
    protocol_path = ROOT/'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json'
    policy_path, cold = stage_paths(STAGES[-1])
    source = torch.load(policy_path, map_location='cpu', weights_only=False)
    actual = verify_selected_source(policy_path, source, cold, ROOT, sha(protocol_path))
    assert actual['source_stage'] == 'early_lateral_queue01' and actual['selected_candidate_id'] == 1
    assert actual['source_cold_summary']['airborne_contacts'] == 13
    factory_path = BASE/'current_intercept_diagnosis_20261027/native_factory.py'
    spec = importlib.util.spec_from_file_location('prepared_shared_target_native_factory', factory_path)
    factory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(factory)
    replay = replay_selected_public(policy_path, source, cold, ROOT, factory)
    original_cpu = read(BASE/'current_intercept_diagnosis_20261027/CPU_SHARED_TARGET_CHECK.json')
    assert len(replay['actual_selected_source_all5_public_checks']) == 5
    for item in replay['actual_selected_source_all5_public_checks']:
        original_result = next(r for r in original_cpu['results'] if r['run'] == 'early_lateral7_best_cold128_01'
                               and r['candidate']['candidate_id'] == item['candidate_id'])
        assert item['command_stream_sha256'] == original_result['command_stream_sha256']
    help_run = subprocess.run([sys.executable, str(entry), '--help'], cwd=HERE.parents[2],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), text=True, capture_output=True, timeout=60)
    with (HERE/'native_entry_help.stdout.log').open('x') as f:
        f.write(help_run.stdout+help_run.stderr)
    assert help_run.returncode == 0, help_run.stderr
    files = [Path(__file__), HERE/'run.py', HERE/'source_support.py', HERE/'stance_source_admission.py',
        entry, prior, HERE/'prepare_entry.py', HERE/'ENTRY_GENERATION_CHECK.json', protocol_path, factory_path,
        BASE/'current_intercept_diagnosis_20261027/NATIVE_PROPOSAL.json',
        BASE/'current_intercept_diagnosis_20261027/NATIVE_FACTORY_CPU_CHECK.json']
    record = dict(checked_at=datetime.now(timezone.utc).isoformat(), native_loop_and_scoring_AST_unchanged=True,
        native_environment_reset_noise_case_and_snapshot_calls_AST_unchanged=True,
        original128_no_labels_or_episode_ids_rewritten=True, skip_and_fail_closed_gates_checked=True,
        gate_checks=gate_checks, actual_early_independent_source_admitted=True,
        actual_early_all5_source_CPU_command_streams_match_prior_checks=True, actual_source_replay=replay,
        native_entry_help_exit_code=help_run.returncode, future_actual_source_still_pending=True,
        no_future_checkpoint_fabricated=True, new_native_execution=False, formal_acceptance=False,
        source_sha256={str(p): sha(p) for p in files})
    with (HERE/'PRE_NATIVE_CPU_CHECK.json').open('x') as f:
        f.write(json.dumps(record, indent=2)+'\n')
    print(json.dumps(dict(original_native_loop_and_scoring_AST_unchanged=True, gate_checks=len(gate_checks),
        actual_source_admitted=True, CPU_candidate_checks=5, native_entry_help_exit_code=0,
        new_native_execution=False)), flush=True)


if __name__ == '__main__':
    main()
