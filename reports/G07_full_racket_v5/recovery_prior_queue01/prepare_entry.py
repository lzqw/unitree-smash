"""Derive recovery-prior native trials while preserving the existing native loop."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / 'windowed_swing_queue01'
PROPOSALS = HERE.parent / 'recovery_prior_native_20261027'
SEED = HERE.parent / 'recovery_motion_prior_20261027'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def once(text, old, new):
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new)


def assignment(text, name, value):
    matches = [n for n in ast.parse(text).body if isinstance(n, ast.Assign)
               and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name]
    assert len(matches) == 1 and matches[0].lineno == matches[0].end_lineno
    lines = text.splitlines(True)
    lines[matches[0].lineno-1] = name+' = '+value+'\n'
    return ''.join(lines)


def main():
    cpu = json.loads((PROPOSALS / 'CPU_CONTROLLER_CHECK.json').read_text())
    assert len(cpu['results']) == len(cpu['candidates']) == 9
    assert all(sha(Path(p)) == h for p, h in cpu['source_sha256'].items())
    proposal = dict(prepared_at=datetime.now(timezone.utc).isoformat(), candidates=cpu['candidates'],
        factory_sha256=sha(PROPOSALS / 'prior_controller.py'), CPU_check_sha256=sha(PROPOSALS / 'CPU_CONTROLLER_CHECK.json'),
        seed_metadata_sha256=sha(SEED / 'SEED_RECEIPT.json'), seed_table_sha256=sha(SEED / 'recorded_recovery_prior.npz'),
        algorithm='Bounded shared18-joint residual traces, original.15rad mixer and5rad/s arm target limiter; absolute, causal planned-event or latched observed-air-hit phase alignment.',
        source_policy_selection_pending_actual_window7_and_cold128=True,
        physics_rules_cases_target_and_thresholds_unchanged=True,
        native_execution=False, seed_success_not_inherited=True, formal_acceptance=False)
    with (PROPOSALS / 'PROPOSAL.json').open('x') as stream:
        stream.write(json.dumps(proposal, indent=2)+'\n')
    text = (PRIOR / 'evaluate_windowed_swing.py').read_text()
    text = text.replace('verify_early_source', 'verify_window_source')
    text = text.replace('windowed_swing_20261027', 'recovery_prior_native_20261027')
    text = text.replace('window_controller.py', 'prior_controller.py')
    text = text.replace("SCHEMA = 'g07-windowed-swing-contact-recovery-v5'", "SCHEMA = 'g07-shared-native-recovery-prior-v5'")
    start = text.index("    assert cpu['source_sha256']")
    end = text.index("    if args.mode == 'search':", start)
    text = text[:start]+'''    assert cpu['factory_sha256'] == sha(PROPOSALS / 'prior_controller.py')
    assert len(cpu['results']) == len(proposal['candidates']) == 9
    assert cpu['results'][0]['maximum_source_action_difference'] == 0.
    assert cpu['results'][0]['maximum_original_GPU_action_difference'] <= 5e-4
    assert cpu['all9_action_streams_match_published_portable_proposals_bitwise']
    assert cpu['source_actor_and_normalizer_and_checkpoint_tree_unchanged']
'''+text[end:]
    text = text.replace("candidates == cpu['candidates'] and [r['candidate_id'] for r in candidates] == list(range(7))", "candidates == cpu['candidates'] == module().candidates()")
    text = text.replace("len(search_metrics['history']) == 7", "len(search_metrics['history']) == 9")
    text = text.replace('<= 716800', '<= 921600')
    text = text.replace("name='independent_cold_selected_windowed_swing',\n                           **checkpoint['windowed_swing']", "name='independent_cold_selected_recovery_prior',\n                           **checkpoint['recovery_prior']")
    text = once(text, "        factory = module()", """        factory = module()
        seed_metadata, seed_arrays = factory.load_seed()
        if args.mode == 'cold':
            assert checkpoint['seed_metadata_sha256'] == sha(HERE.parent / 'recovery_motion_prior_20261027/SEED_RECEIPT.json')
            assert checkpoint['seed_table_sha256'] == sha(HERE.parent / 'recovery_motion_prior_20261027/recorded_recovery_prior.npz')
            assert checkpoint['seed_arrays'].keys() == seed_arrays.keys()
            assert all(np.array_equal(checkpoint['seed_arrays'][k], value) for k, value in seed_arrays.items())""")
    text = once(text, "controller = factory.WindowedSwingController(base, 128, env.device, candidate['joint_velocity_rad_s'])",
        "controller = factory.RecoveryPriorController(base, seed_metadata, seed_arrays, 128, env.device,\n                **{key: candidate[key] for key in ['group', 'table', 'alignment', 'gain']})")
    text = once(text, "windowed_swing=dict(joint_velocity_rad_s=candidate['joint_velocity_rad_s'])",
        """recovery_prior={key: candidate[key] for key in ['group', 'table', 'alignment', 'gain']},
                    seed_metadata=seed_metadata, seed_arrays={key: value.copy() for key, value in seed_arrays.items()},
                    seed_metadata_sha256=sha(HERE.parent / 'recovery_motion_prior_20261027/SEED_RECEIPT.json'),
                    seed_table_sha256=sha(HERE.parent / 'recovery_motion_prior_20261027/recorded_recovery_prior.npz')""")
    text = once(text, "HERE.parent / 'origin0_measured_swing_velocity_20261027/swing_controller.py']",
        "HERE.parent / 'origin0_measured_swing_velocity_20261027/swing_controller.py',\n             HERE.parent / 'windowed_swing_20261027/window_controller.py',\n             HERE.parent / 'recovery_motion_prior_20261027/SEED_RECEIPT.json',\n             HERE.parent / 'recovery_motion_prior_20261027/recorded_recovery_prior.npz']")
    text = text.replace('Execute preparation-preserving windowed swing proposals', 'Execute shared recovery-prior proposals')
    text = text.replace('uses seven sequential', 'uses nine sequential')
    text = text.replace('windowed_swing_controller', 'native_recovery_prior_controller')
    ast.parse(text)
    with (HERE / 'evaluate_recovery_prior.py').open('x') as stream:
        stream.write(text)
    text = (PRIOR / 'run.py').read_text()
    values = dict(PRIOR="OUTPUTS / 'windowed_swing_queue01'", SOURCE="OUTPUTS / 'windowed_swing7_n128_01/best_candidate.pt'",
        SOURCE_COLD="OUTPUTS / 'windowed_swing7_best_cold128_01'", PROPOSALS="OUTPUTS / 'recovery_prior_native_20261027'",
        SEARCH="OUTPUTS / 'recovery_prior9_n128_01'", COLD="OUTPUTS / 'recovery_prior9_best_cold128_01'")
    for name, value in values.items():
        text = assignment(text, name, value)
    replacements = [
        ('evaluate_windowed_swing.py', 'evaluate_recovery_prior.py'),
        ('window_controller.py', 'prior_controller.py'),
        ('1957568', '1966524'),
        ('verify_early_source', 'verify_window_source'),
        ('WAITING_FOR_CONFIRMED_EARLY7_OWNER_AND_COLD128', 'WAITING_FOR_CONFIRMED_WINDOW7_OWNER_AND_COLD128'),
        ("assert read(PRIOR / 'state.json')['state'] == 'COMPLETED_EARLY_LATERAL7_AND_INDEPENDENT_COLD128'", "assert read(PRIOR / 'state.json')['state'] == 'COMPLETED_WINDOWED_SWING7_AND_INDEPENDENT_COLD128'"),
        ('RUNNING_WINDOWED_SWING_', 'RUNNING_RECOVERY_PRIOR_'),
        ('CHECKING_SELECTED_SOURCE_CPU_BEFORE_WINDOWED_SWING_NATIVE', 'CHECKING_SELECTED_SOURCE_CPU_BEFORE_RECOVERY_PRIOR_NATIVE'),
        ("state('COMPLETED_WINDOWED_SWING7_AND_INDEPENDENT_COLD128', final_summary", "state('COMPLETED_RECOVERY_PRIOR9_AND_INDEPENDENT_COLD128', final_summary"),
        ('list(range(7))', 'list(range(9))'),
        ("['results']) == 7", "['results']) == 9"),
        ('env_steps_cap=716800, max_seconds=18000', 'env_steps_cap=921600, max_seconds=21600'),
        ("job(SEARCH, 'search', SOURCE, 716800, 18000, expected)", "job(SEARCH, 'search', SOURCE, 921600, 21600, expected)"),
    ]
    for old, new in replacements:
        assert old in text, old
        text = text.replace(old, new)
    text = once(text, "    expected = dict(read(PRIOR / 'PLAN.json')['source_sha256'])", "    expected = dict(read(PRIOR / 'PLAN.json')['source_sha256'])\n    expected.update(read(PROPOSALS / 'CPU_CONTROLLER_CHECK.json')['source_sha256'])")
    text = once(text, "PROPOSALS / 'check_controller.py', PROPOSALS / 'CPU_CONTROLLER_CHECK.json', PROPOSALS / 'PROPOSAL.json']",
        "PROPOSALS / 'check_controller.py', PROPOSALS / 'CPU_CONTROLLER_CHECK.json', PROPOSALS / 'PROPOSAL.json',\n        PROPOSALS / 'prepare_factory.py', PROPOSALS / 'FACTORY_GENERATION_CHECK.json']")
    text = text.replace('actual_source_pending_early7_and_independent_cold', 'actual_source_pending_window7_and_independent_cold')
    text = text.replace('all7 early-lateral original128 trials', 'all7 windowed-swing original128 trials')
    text = text.replace('Quartic windowed swing velocity around the unchanged causal stroke center; original preparation, hit configuration, feedback, tool guards and5rad/s rate limit.', 'Shared bounded actual recovery residual traces with group and causal alignment ablations; original.15rad residual mixer and5rad/s arm rate limit.')
    text = text.replace('After early-lateral7/cold128 exits, execute7 windowed swing proposals.', 'After windowed-swing7/cold128 exits, execute9 shared recovery-prior proposals.')
    ast.parse(text)
    with (HERE / 'run.py').open('x') as stream:
        stream.write(text)
    receipt = dict(prepared_at=datetime.now(timezone.utc).isoformat(),
        source_sha256={str(PRIOR / n): sha(PRIOR / n) for n in ['evaluate_windowed_swing.py', 'run.py']},
        generated_sha256={str(HERE / n): sha(HERE / n) for n in ['evaluate_recovery_prior.py', 'run.py']},
        generator_sha256=sha(Path(__file__)), original_native_rollout_logging_and_scoring_loop_preserved=True,
        original128_cases_preserved=True, new_native_execution=False, formal_acceptance=False)
    with (HERE / 'ENTRY_GENERATION_CHECK.json').open('x') as stream:
        stream.write(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
