"""Create separately owned native/queue entries from the already reviewed loop."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORTS = HERE.parent
PRIOR = REPORTS / 'early_lateral_queue01'
PROPOSALS = REPORTS / 'windowed_swing_20261027'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(text, old, new):
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new)


def main():
    cpu = json.loads((PROPOSALS / 'CPU_CONTROLLER_CHECK.json').read_text())
    assert len(cpu['candidates']) == len(cpu['results']) == 7
    assert all(sha(Path(p)) == h for p, h in cpu['source_sha256'].items())
    proposal = dict(prepared_at=datetime.now(timezone.utc).isoformat(), candidates=cpu['candidates'],
        factory_sha256=sha(PROPOSALS / 'window_controller.py'), CPU_check_sha256=sha(PROPOSALS / 'CPU_CONTROLLER_CHECK.json'),
        algorithm='Quartic velocity envelope within the original public-forecast swing half-window; the feedforward hit position at the center and all pre-window preparation commands are unchanged on equal public input.',
        physics_rules_cases_target_and_thresholds_unchanged=True,
        source_policy_selection_pending_actual_early7_and_cold128=True,
        physical_tracking_or_contacts_or_recovery_proven=False, native_execution=False, formal_acceptance=False)
    with (PROPOSALS / 'PROPOSAL.json').open('x') as stream:
        stream.write(json.dumps(proposal, indent=2)+'\n')
    text = (PRIOR / 'evaluate_early_lateral.py').read_text()
    text = text.replace('verify_measured_source', 'verify_early_source')
    text = text.replace('early_tangent_intercept_20261027', 'windowed_swing_20261027')
    text = text.replace('early_lateral_controller.py', 'window_controller.py')
    text = text.replace("SCHEMA = 'g07-public-early-lateral-intercept-v5'", "SCHEMA = 'g07-windowed-swing-contact-recovery-v5'")
    start = text.index("    assert cpu['factory_sha256']")
    end = text.index("    if args.mode == 'search':", start)
    text = text[:start]+'''    assert cpu['source_sha256'][str(PROPOSALS / 'window_controller.py')] == sha(PROPOSALS / 'window_controller.py')
    assert len(cpu['results']) == len(proposal['candidates']) == 7
    assert cpu['results'][0]['maximum_normalized_action_change'] == 0.
    assert cpu['results'][0]['maximum_CPU_vs_original_GPU_action_difference'] <= 5e-4
    assert cpu['checkpoint_tree_unchanged'] and cpu['source_actor_and_normalizer_unchanged']
'''+text[end:]
    text = replace_once(text, "        assert cpu['all_nine_candidates_checked'] and cpu['source_actor_weights_and_normalizers_and_checkpoint_unchanged']\n", '')
    text = replace_once(text, "        assert candidates == [row for row in module().candidates() if row['candidate_id'] in [0, 1, 2, 3, 5, 6, 7]]", "        assert candidates == cpu['candidates'] and [r['candidate_id'] for r in candidates] == list(range(7))")
    text = replace_once(text, "name='independent_cold_selected_early_lateral',\n                           **checkpoint['early_lateral']", "name='independent_cold_selected_windowed_swing',\n                           **checkpoint['windowed_swing']")
    text = replace_once(text, "controller = factory.EarlyLateralController(base, 128, env.device, candidate['gain'], candidate['start_clock_s'])", "controller = factory.WindowedSwingController(base, 128, env.device, candidate['joint_velocity_rad_s'])")
    text = replace_once(text, "early_lateral=dict(gain=candidate['gain'], start_clock_s=candidate['start_clock_s'])", "windowed_swing=dict(joint_velocity_rad_s=candidate['joint_velocity_rad_s'])")
    text = replace_once(text, "HERE / 'source_support.py', HERE / 'SELECTED_SOURCE_CPU_CHECK.json']", "HERE / 'source_support.py', HERE / 'SELECTED_SOURCE_CPU_CHECK.json',\n             HERE.parent / 'early_tangent_intercept_20261027/early_lateral_controller.py',\n             HERE.parent / 'origin0_measured_swing_velocity_20261027/swing_controller.py']")
    text = text.replace('Execute public early-lateral preparation proposals', 'Execute preparation-preserving windowed swing proposals')
    text = text.replace('measured_swing_controller', 'windowed_swing_controller')
    ast.parse(text)
    with (HERE / 'evaluate_windowed_swing.py').open('x') as stream:
        stream.write(text)
    text = (PRIOR / 'run.py').read_text()
    replacements = [
        ('measured_swing_queue01', 'early_lateral_queue01'),
        ('measured_swing9_n128_01', 'early_lateral7_n128_01'),
        ('measured_swing9_best_cold128_01', 'early_lateral7_best_cold128_01'),
        ('early_tangent_intercept_20261027', 'windowed_swing_20261027'),
        ("\nSEARCH = OUTPUTS / 'early_lateral7_n128_01'", "\nSEARCH = OUTPUTS / 'windowed_swing7_n128_01'"),
        ("\nCOLD = OUTPUTS / 'early_lateral7_best_cold128_01'", "\nCOLD = OUTPUTS / 'windowed_swing7_best_cold128_01'"),
        ('evaluate_early_lateral.py', 'evaluate_windowed_swing.py'),
        ('early_lateral_controller.py', 'window_controller.py'),
        ('1951183', '1957568'),
        ('verify_measured_source', 'verify_early_source'),
        ('WAITING_FOR_CONFIRMED_SWING9_OWNER_AND_COLD128', 'WAITING_FOR_CONFIRMED_EARLY7_OWNER_AND_COLD128'),
        ('COMPLETED_MEASURED_SWING9_AND_INDEPENDENT_COLD128', 'COMPLETED_EARLY_LATERAL7_AND_INDEPENDENT_COLD128'),
        ('RUNNING_EARLY_LATERAL_', 'RUNNING_WINDOWED_SWING_'),
        ('CHECKING_SELECTED_SOURCE_CPU_BEFORE_EARLY_LATERAL_NATIVE', 'CHECKING_SELECTED_SOURCE_CPU_BEFORE_WINDOWED_SWING_NATIVE'),
        ('COMPLETED_EARLY_LATERAL7_AND_INDEPENDENT_COLD128\', final_summary', 'COMPLETED_WINDOWED_SWING7_AND_INDEPENDENT_COLD128\', final_summary'),
        ('[0, 1, 2, 3, 5, 6, 7]', 'list(range(7))'),
    ]
    for old, new in replacements:
        assert old in text, old
        text = text.replace(old, new)
    start = text.index('    files = [')
    end = text.index('    for path in files:', start)
    text = text[:start]+'''    files = [Path(__file__), HERE / 'evaluate_windowed_swing.py', HERE / 'prepare_entry.py',
        HERE / 'ENTRY_GENERATION_CHECK.json', HERE / 'source_support.py', HERE / 'PRE_NATIVE_CPU_CHECK.json',
        HERE / 'check_preparation.py', PRIOR / 'PLAN.json', PROPOSALS / 'window_controller.py',
        PROPOSALS / 'check_controller.py', PROPOSALS / 'CPU_CONTROLLER_CHECK.json', PROPOSALS / 'PROPOSAL.json']
'''+text[end:]
    text = replace_once(text, "    assert read(PROPOSALS / 'CPU_CONTROLLER_CHECK.json')['all_nine_candidates_checked']", "    assert len(read(PROPOSALS / 'CPU_CONTROLLER_CHECK.json')['results']) == 7")
    text = text.replace('actual_source_pending_all9_and_independent_cold', 'actual_source_pending_early7_and_independent_cold')
    text = text.replace('all9 original128 trials', 'all7 early-lateral original128 trials')
    text = text.replace('Earlier public world-Y material-hit-point feedback from the unchanged causal source forecast; original support guard and5rad/s target limiter.', 'Quartic windowed swing velocity around the unchanged causal stroke center; original preparation, hit configuration, feedback, tool guards and5rad/s rate limit.')
    text = text.replace(", excluded_CPU_overshoot_candidates=[4, 8]", '')
    text = text.replace('After measured swing9/cold128 exits, execute7 public early-lateral proposals.', 'After early-lateral7/cold128 exits, execute7 windowed swing proposals.')
    ast.parse(text)
    with (HERE / 'run.py').open('x') as stream:
        stream.write(text)
    receipt = dict(prepared_at=datetime.now(timezone.utc).isoformat(),
        source_sha256={str(PRIOR / name): sha(PRIOR / name) for name in ['evaluate_early_lateral.py', 'run.py']},
        generated_sha256={str(HERE / name): sha(HERE / name) for name in ['evaluate_windowed_swing.py', 'run.py']},
        original_native_rollout_logging_and_scoring_loop_preserved=True, original128_cases_preserved=True,
        new_native_execution=False, formal_acceptance=False)
    with (HERE / 'ENTRY_GENERATION_CHECK.json').open('x') as stream:
        stream.write(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
