"""Admit V20 only after full V15 preservation and first-match feedback.

The model audit may start earlier than the nominal plan query. Its actual
first-match result is required; no fixed-time gate is added to the controller.
This is input/wiring validation, not native or formal task acceptance.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['entry', 'configuration', 'run', 'matches', 'feedback', 'preservation', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    read = lambda p: json.loads(Path(p).read_text())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if args.output.exists():
        raise ValueError('Fresh entry receipt required')
    text = args.entry.read_text()
    ast.parse(text)
    call = 'action=native_replay.action(tick,obs,action,finished)'
    if text.count(call) != 1 or 'native_pure_control_preserved_foot_replanning_v20' not in text:
        raise ValueError('Wrong recovery command owner')
    block = text[text.index(call):text.index('observation, _, done, _ = wrapped.step(action)')]
    if 'action = ' in block or 'action=' in block.split('\n', 1)[1]:
        raise ValueError('Another writer follows the recovery controller')
    cfg = read(args.configuration)
    library = Path(cfg['model_replanning_library'])
    matches = read(args.matches)
    inputs = read(args.feedback/'inputs.json')
    result = read(args.feedback/'result.json')
    if (result['state'] != 'COMPLETED_FIRST_MATCH_FEEDBACK_MODEL' or
            result['native_success'] or result['valid_expert_labels']):
        raise ValueError('Completed model-only first-match evidence required')
    if (not inputs['unchanged_acceptance'] or inputs['training'] or
            inputs['native_teleport'] or inputs['physical_substeps'] != 160):
        raise ValueError('Incorrect model verification contract')
    root = Path(__file__).resolve().parents[1]
    for filename in ['pure_control_preserved_foot_replanning_v20.py',
                     'audit_preserved_foot_replanning_v20.py']:
        path = root/'scripts'/filename
        if inputs['source_sha256'].get(str(path)) != sha(path):
            raise ValueError('Feedback did not execute the complete V20 controller')
    pins = {}
    for data in [matches, inputs, read(library)]:
        for filename,digest in data['source_sha256'].items():
            if sha(filename) != digest or (filename in pins and pins[filename] != digest):
                raise ValueError('Changed/conflicting audit input '+filename)
            pins[filename] = digest
    for path in [library, args.matches, Path(cfg['model_reference_library']),
            *[args.run/name for name in ['job.json', 'native_command_inputs.json',
                'pure_control_commands.npz', 'pure_control_diagnostics.jsonl']]]:
        if inputs['source_sha256'].get(str(path.resolve())) != sha(path):
            raise ValueError('Feedback audited different inputs: '+str(path))
    if read(args.run/'job.json')['status'] != 'SUCCEEDED':
        raise ValueError('Completed original native history required')
    source_cfgs = []
    for filename in inputs['source_sha256']:
        if Path(filename).name == 'controller.json':
            value = read(filename)
            if 'interface' in value:
                source_cfgs.append(value)
    if not any(c.get('controller') == 'pure_control_stance_foot_reference_v15' and
            c['interface'] == cfg['interface'] and
            sha(c['model_reference_library']) == sha(cfg['model_reference_library']) for c in source_cfgs):
        raise ValueError('Hardware interface or frozen prefix differs from the model audit')
    preservation = read(args.preservation)
    if (not preservation['passed'] or preservation['native_execution'] or
            preservation['native_success'] or preservation['valid_expert_labels']):
        raise ValueError('Model-only command preservation receipt required')
    for filename,digest in preservation['source_sha256'].items():
        if sha(filename) != digest or (filename in pins and pins[filename] != digest):
            raise ValueError('Changed/conflicting preservation input '+filename)
        pins[filename] = digest
    for path in [library, args.run/'pure_control_commands.npz', args.run/'pure_control_diagnostics.jsonl']:
        if preservation['source_sha256'].get(str(path.resolve())) != sha(path):
            raise ValueError('Preservation used another native history or candidate library')
    baseline_paths = [Path(f) for f in preservation['source_sha256'] if Path(f).name == 'best_verified_v15.json']
    if len(baseline_paths) != 1:
        raise ValueError('Verified V15 baseline receipt required')
    expected = set(read(baseline_paths[0])['clean_stable_cases'])
    if expected != {r['case'] for r in preservation['checks']} or not expected:
        raise ValueError('Missing protected success')
    for row in preservation['checks']:
        if row['last_tick'] != 799 or row['commands_checked'] != 800-row['first_tick'] or row['maximum_request_error_rad'] > 2e-6 or row['replanning_selected']:
            raise ValueError('Incomplete unchanged command replay')
    checks = []
    for match in matches['matches']:
        found = [r for r in result['results'] if r['case'] == match['case']]
        if len(found) != 1:
            raise ValueError('Missing actual first-match feedback')
        row = found[0]
        if (row['source_key'] != match['source_key'] or
                row['actual_first_match_tick'] != match['actual_tick'] or
                row['failure'] is not None or row['steps'] != 800-match['actual_tick'] or
                row['end_time_s'] != 8. or not row['model_terminal_window_passed'] or
                row['final_model_stable_s'] < .8 or
                row['maximum_native_prefix_readback_error_rad'] > 2e-6):
            raise ValueError('Actual-trigger feedback failed: '+str(match['case']))
        checks.append(row)
    candidates = {r['source_key'] for r in read(library)['references']}
    if not checks or candidates != {r['source_key'] for r in checks}:
        raise ValueError('Some candidate lacks first-match evidence')
    files = [Path(__file__), args.entry, args.configuration, args.matches,
             args.feedback/'inputs.json', args.feedback/'result.json', library, args.preservation]
    pins.update({str(p.resolve()):sha(p) for p in files})
    output = dict(passed=True, checks=checks, preserved_commands=preservation['checks'], single_final_native_writer=True,
        native_execution=False, valid_expert_labels=False,
        trigger_uses_actual_state_not_case_or_time=True, entry_sha256=sha(args.entry),
        source_sha256=pins, model_pass_is_not_native_acceptance=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    temporary.replace(args.output)
    print(json.dumps(dict(passed=True, actual_trigger_checks=len(checks), native_execution=False)))


if __name__ == '__main__':
    main()
