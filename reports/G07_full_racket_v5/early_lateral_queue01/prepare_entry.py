"""Derive a separate early-lateral native entry without editing live swing9 code."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'measured_swing_queue01/evaluate_swing.py'
TARGET = HERE / 'evaluate_early_lateral.py'


def main():
    text = SOURCE.read_text()
    replacements = [
        ('origin0_measured_swing_velocity_20261027', 'early_tangent_intercept_20261027'),
        ("SCHEMA = 'g07-measured-swing-contact-recovery-v5'", "SCHEMA = 'g07-public-early-lateral-intercept-v5'"),
        ('swing_controller.py', 'early_lateral_controller.py'),
        ("PROPOSALS / 'RESULT.json'", "PROPOSALS / 'PROPOSAL.json'"),
        ("cpu['factory_source_sha256']", "cpu['factory_sha256']"),
        ("assert cpu['original_source_unchanged_candidate_matches_actual_GPU']",
         "assert cpu['results'][0]['maximum_source_action_difference'] == 0. and cpu['results'][0]['maximum_original_GPU_action_difference'] <= 5e-4"),
        ("assert cpu['candidate_count'] == len(proposal['candidates']) == 9", "assert cpu['candidate_count'] == 9 and len(proposal['candidates']) == 7"),
        ('source_check = verify_recovery_source(args.source, checkpoint, args.source_evaluation, ROOT, sha(protocol_path))',
         'source_check = verify_measured_source(args.source, checkpoint, args.source_evaluation, ROOT, sha(protocol_path))'),
        ("assert cpu['source_checkpoint_sha256'] == sha(args.source)",
         "assert cpu['all_nine_candidates_checked'] and cpu['source_actor_weights_and_normalizers_and_checkpoint_unchanged']"),
        ("candidates = proposal['candidates']", "candidates = proposal['candidates']\n        assert candidates == [row for row in module().candidates() if row['candidate_id'] in [0, 1, 2, 3, 5, 6, 7]]"),
        ("assert len(search_metrics['history']) == 9", "assert len(search_metrics['history']) == 7"),
        ('<= 921600', '<= 716800'),
        ("candidates = [dict(candidate_id=selected['candidate_id'], name='independent_cold_selected_swing',\n                           overrides=checkpoint['runtime_overrides'])]",
         "candidates = [dict(candidate_id=selected['candidate_id'], name='independent_cold_selected_early_lateral',\n                           **checkpoint['early_lateral'])]"),
        ("factory.MeasuredSwingController(base, 128, env.device, candidate['overrides'])",
         "factory.EarlyLateralController(base, 128, env.device, candidate['gain'], candidate['start_clock_s'])"),
        ("runtime_overrides=candidate['overrides'], protocol_sha256=sha(protocol_path)",
         "early_lateral=dict(gain=candidate['gain'], start_clock_s=candidate['start_clock_s']), protocol_sha256=sha(protocol_path)"),
        ("HERE / 'run.py', HERE / 'PLAN.json', HERE / 'PRE_NATIVE_CPU_CHECK.json']",
         "HERE / 'run.py', HERE / 'PLAN.json', HERE / 'PRE_NATIVE_CPU_CHECK.json',\n             HERE / 'source_support.py', HERE / 'SELECTED_SOURCE_CPU_CHECK.json']"),
        ('import traceback\n', 'import traceback\n\nfrom source_support import verify_measured_source\n'),
        ('nine sequential ground resets', 'seven sequential ground resets'),
        ('Execute fixed measured swing proposals', 'Execute public early-lateral preparation proposals'),
    ]
    counts = []
    for before, after in replacements:
        count = text.count(before)
        assert count > 0, before
        text = text.replace(before, after)
        counts.append(dict(before=before, after=after, occurrences=count))
    assert 'MeasuredSwingController(' not in text and "candidate['overrides']" not in text
    assert "checkpoint['runtime_overrides']" not in text
    ast.parse(text, filename=str(TARGET))
    with TARGET.open('x') as stream:
        stream.write(text)
    record = dict(source=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        target=str(TARGET), target_sha256=hashlib.sha256(TARGET.read_bytes()).hexdigest(),
        replacements=counts, source_file_changed=False, native_execution=False)
    with (HERE / 'ENTRY_GENERATION_CHECK.json').open('x') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(entry_generated=True, replacements=len(counts), target_sha256=record['target_sha256'])))


if __name__ == '__main__':
    main()
