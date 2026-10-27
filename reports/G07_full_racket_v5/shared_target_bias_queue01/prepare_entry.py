"""Derive5 reference-bias trials from the held original native episode loop."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent/'stance_recovery_queue01'
CORE = HERE.parent/'current_intercept_diagnosis_20261027'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def once(text, old, new):
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new)


def stance_admission():
    text = (PRIOR/'source_support.py').read_text().split('\ndef replay_selected_public(', 1)[0]
    text = once(text, 'def verify_strike_source(', 'def verify_stance_source(')
    text = once(text, "factory_path=reports/'free_floating_strike_20261027/native_factory.py'",
                "factory_path=reports/'public_stance_recovery_20261027/native_factory.py'")
    text = once(text, "source['schema']=='g07-public-momentum-strike-v5'",
                "source['schema']=='g07-public-stance-recovery-v5'")
    text = once(text, "core_path=reports/'free_floating_strike_20261027/strike_controller.py'",
                "core_path=reports/'public_stance_recovery_20261027/stance_controller.py'")
    text = once(text, "source['strike_core_sha256']", "source['stance_core_sha256']")
    text = once(text, "    free_core_path=reports/'free_floating_strike_20261027/free_floating.py'\n    assert source['free_floating_core_sha256']==sha(free_core_path)\n", '')
    text = text.replace("reports/'air_countermotion3_n128_01/best_candidate.pt'", "reports/'momentum_strike4_n128_01/best_candidate.pt'")
    text = text.replace("reports/'air_countermotion3_best_cold128_01'", "reports/'momentum_strike4_best_cold128_01'")
    text = once(text, "support_path=reports/'momentum_strike_queue01/source_support.py'",
                "support_path=reports/'stance_recovery_queue01/source_support.py'")
    text = once(text, 'support.verify_countermotion_source(', 'support.verify_strike_source(')
    text = once(text, "assert search==reports/'momentum_strike4_n128_01'", "assert search==reports/'stance_recovery4_n128_01'")
    text = once(text, 'model_path,core_path,free_core_path,support_path', 'model_path,core_path,support_path')
    text = text.replace('selected_strike_candidate_id', 'selected_stance_candidate_id')
    text = text.replace('previous_countermotion_source_check', 'previous_strike_source_check')
    text = text.replace('all4_actual_strike_search_summaries_independently_recomputed',
                        'all4_actual_stance_search_summaries_independently_recomputed')
    text = text.replace('Admit completed momentum-strike4/cold128 before any stance GPU job.',
                        'Admit completed stance4/cold128 before any shared-target GPU job.')
    ast.parse(text)
    return text


def main():
    cpu = read(CORE/'CPU_SHARED_TARGET_CHECK.json')
    binding = read(CORE/'NATIVE_FACTORY_CPU_CHECK.json')
    assert len(cpu['results']) == 10 and len(binding['results']) == 4
    assert all(sha(Path(p)) == h for record in [cpu, binding] for p, h in record['source_sha256'].items())
    candidates = [r['candidate'] for r in cpu['results'][:5]]
    assert candidates == [r['candidate'] for r in cpu['results'][5:]]
    text = (PRIOR/'evaluate_stance_recovery.py').read_text()
    text = text.replace('Execute public post-hit stance-leg recovery proposals', 'Execute shared y/z target-reference proposals')
    text = text.replace('uses four sequential ground resets', 'uses five sequential ground resets')
    text = text.replace('verify_strike_source', 'verify_selected_source')
    text = text.replace("PROPOSALS = MAIN / 'reports/G07_full_racket_v5/public_stance_recovery_20261027'",
                        "PROPOSALS = MAIN / 'reports/G07_full_racket_v5/current_intercept_diagnosis_20261027'")
    text = text.replace("SCHEMA = 'g07-public-stance-recovery-v5'", "SCHEMA = 'g07-public-shared-target-bias-v5'")
    text = text.replace('native_stance_recovery_controller', 'native_shared_target_controller')
    text = text.replace("PROPOSALS / 'CPU_CONTROLLER_CHECK.json'", "PROPOSALS / 'CPU_SHARED_TARGET_CHECK.json'")
    start = text.index("    assert factory_check['factory_sha256']")
    end = text.index("    if args.mode == 'search':", start)
    text = text[:start]+'''    assert factory_check['factory_sha256'] == sha(PROPOSALS / 'native_factory.py')
    assert len(cpu['results']) == 10 and len(proposal['candidates']) == 5
    assert all(r['bounded_finite'] and r['actors_and_normalizers_unchanged'] for r in cpu['results'])
    assert all(r['maximum_source_action_change'] == 0. and r['CPU_vs_actual_GPU_max_action_error'] <= 5e-4 for r in cpu['results'] if r['candidate']['candidate_id'] == 0)
    assert all(r['original_leg_commands_bitwise_preserved'] and r['causal_offsets_and_parameters_restored'] for r in cpu['results'])
    assert all(r['checked_core_command_stream_bitwise_identical'] and (r['actual_baseline_GPU_max_error'] is None or r['actual_baseline_GPU_max_error'] <= 5e-4) for r in factory_check['results'])
'''+text[end:]
    text = once(text, "        assert candidates == [r['candidate'] for r in cpu['results'][:4]] == [r['candidate'] for r in cpu['results'][4:8]] == [r['candidate'] for r in cpu['results'][8:]]",
                "        assert candidates == [r['candidate'] for r in cpu['results'][:5]] == [r['candidate'] for r in cpu['results'][5:]]")
    text = text.replace("assert len(search_metrics['history']) == 4", "assert len(search_metrics['history']) == 5")
    text = text.replace('<= 409600', '<= 512000')
    text = text.replace('independent_cold_selected_stance_recovery', 'independent_cold_selected_shared_target')
    text = text.replace("**checkpoint['stance_recovery']", "**checkpoint['shared_target_bias']")
    text = text.replace("['gain', 'horizontal_damping']", "['bias_world_m']")
    text = text.replace("PROPOSALS / 'stance_controller.py'", "PROPOSALS / 'shared_target_controller.py'")
    text = text.replace("checkpoint['stance_core_sha256']", "checkpoint['target_core_sha256']")
    text = text.replace('stance_recovery={key: candidate[key]', 'shared_target_bias={key: candidate[key]')
    text = text.replace('stance_core_sha256=sha(', 'target_core_sha256=sha(')
    anchor = "             HERE.parent / 'origin0_measured_swing_velocity_20261027/swing_controller.py']"
    addition = """             HERE.parent / 'origin0_measured_swing_velocity_20261027/swing_controller.py',
             HERE / 'stance_source_admission.py', HERE / 'ENTRY_GENERATION_CHECK.json',
             HERE.parent / 'public_stance_recovery_20261027/native_factory.py',
             HERE.parent / 'public_stance_recovery_20261027/stance_controller.py',
             HERE.parent / 'public_stance_recovery_20261027/CPU_CONTROLLER_CHECK.json',
             HERE.parent / 'stance_recovery_queue01/source_support.py',
             HERE.parent / 'momentum_strike_queue01/source_support.py',
             HERE.parent / 'air_countermotion_queue01/source_support.py',
             HERE.parent / 'recovery_prior_queue01/source_support.py',
             HERE.parent / 'windowed_swing_queue01/source_support.py',
             HERE.parent / 'early_lateral_queue01/source_support.py']"""
    text = once(text, anchor, addition)
    support = stance_admission()
    ast.parse(text)
    proposal = dict(prepared_at=datetime.now(timezone.utc).isoformat(), candidates=candidates,
        core_sha256=sha(CORE/'shared_target_controller.py'), factory_sha256=sha(CORE/'native_factory.py'),
        CPU_check_sha256=sha(CORE/'CPU_SHARED_TARGET_CHECK.json'),
        factory_CPU_check_sha256=sha(CORE/'NATIVE_FACTORY_CPU_CHECK.json'),
        actual_source_pending_existing_chain_and_its_independent128=True,
        original_TRAIN_contact_deficit_required=True, native_execution=False, formal_acceptance=False,
        original_physics_rules_cases_and_target_preserved=True)
    outputs = {'evaluate_shared_target.py': text, 'stance_source_admission.py': support}
    for name, content in outputs.items():
        with (HERE/name).open('x') as f:
            f.write(content)
    with (CORE/'NATIVE_PROPOSAL.json').open('x') as f:
        f.write(json.dumps(proposal, indent=2)+'\n')
    record = dict(prepared_at=datetime.now(timezone.utc).isoformat(), generator_sha256=sha(Path(__file__)),
        source_sha256={str(PRIOR/n): sha(PRIOR/n) for n in ['evaluate_stance_recovery.py', 'source_support.py']},
        generated_sha256={str(HERE/n): sha(HERE/n) for n in outputs},
        unchanged_native_tick_loop_reset_accounting_and_scoring_required=True,
        new_native_execution=False, formal_acceptance=False)
    with (HERE/'ENTRY_GENERATION_CHECK.json').open('x') as f:
        f.write(json.dumps(record, indent=2)+'\n')
    print(json.dumps(dict(generated=list(outputs), new_native_execution=False)))


if __name__ == '__main__':
    main()
