"""Prepare a full native V20 trial only after complete combined-reference evidence.

Copies the verified native entry unchanged. Never runs physics or promotes a
model result. Check mode is also used by the existing bounded native runner.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evidence(args):
    baseline = read(args.baseline); parent = read(args.parent_plan)
    cfg = read(args.proposal); old_cfg = read(baseline['configuration'])
    if baseline['version'] != 'pure_control_preserved_foot_replanning_v20':
        raise ValueError('Complete verified V20 baseline required')
    run = Path(baseline['native_run'])
    if read(run/'job.json')['status'] != 'SUCCEEDED' or Path(parent['output']).resolve() != run.resolve():
        raise ValueError('Parent is not the completed baseline native run')
    for key in set(cfg) | set(old_cfg):
        if key not in ['method', 'model_reference_library'] and cfg.get(key) != old_cfg.get(key):
            raise ValueError('Changed protected execution setting '+key)
    library = Path(cfg['model_reference_library']); bank = read(library)
    old_bank = read(old_cfg['model_reference_library'])
    if bank['references'][:len(old_bank['references'])] != old_bank['references']:
        raise ValueError('Changed original reference bank')
    added = bank['references'][len(old_bank['references']):]
    if not added or bank['native_success'] or bank['valid_expert_labels']:
        raise ValueError('New model-only references required')
    preservation = read(args.preservation)
    if not (args.feedback/'result.json').is_file():
        raise ValueError('Combined feedback has no completed result; native preparation withheld')
    inputs = read(args.feedback/'inputs.json'); result = read(args.feedback/'result.json')
    if (result['state'] != 'COMPLETED_COMBINED_INITIAL_FEEDBACK' or
            inputs['controller'] != 'V20' or inputs['forced_selection'] or
            inputs['physical_substeps'] != 160 or inputs['scoring_substeps'] != 16 or
            result['native_success'] or result['valid_expert_labels']):
        raise ValueError('Completed unforced fine-model V20 feedback required')
    if not preservation['passed'] or preservation['native_execution'] or preservation['valid_expert_labels']:
        raise ValueError('Verified recorded-command preservation required')
    if {r['case'] for r in preservation['checks']} != set(baseline['clean_stable_cases']):
        raise ValueError('Not all verified landing segments were protected')
    for row in preservation['checks']:
        if row['last_tick'] != 799 or row['commands_checked'] != 800-row['first_tick'] or row['maximum_request_error_rad'] > 2e-6:
            raise ValueError('Incomplete native command preservation')
    for data in [preservation, inputs]:
        if data['source_sha256'].get(str(args.proposal.resolve())) != sha(args.proposal):
            raise ValueError('Evidence used another configuration')
        if data['source_sha256'].get(str(library.resolve())) != sha(library):
            raise ValueError('Evidence used another reference bank')
        if data['source_sha256'].get(str(args.baseline.resolve())) != sha(args.baseline):
            raise ValueError('Evidence used another baseline')
    cmdfile = run/'pure_control_commands.npz'
    if inputs['source_sha256'].get(str(cmdfile.resolve())) != sha(cmdfile):
        raise ValueError('Feedback did not use this actual native history')
    by_source = {r['expected_initial_reference']: r for r in result['results']}
    if len(by_source) != len(result['results']) or set(by_source) != {r['source_key'] for r in added}:
        raise ValueError('Missing or repeated new-reference result')
    files = [Path(__file__), args.baseline, args.parent_plan, args.proposal, args.preservation,
        args.feedback/'inputs.json', args.feedback/'result.json', library,
        Path(cfg['model_replanning_library'])]
    checks = []
    for ref in added:
        row = by_source[ref['source_key']]
        path = args.feedback/('state_'+row['state_key'].replace(':', '_')+'.json')
        detail = read(path); initial = detail['initial_actual_state']; trace = detail['trajectory']
        if (row['failure'] is not None or not row['full_model_path'] or
                not row['model_terminal_window_passed'] or row['end_time_s'] != 8. or
                row['final_model_stable_s'] < .8 or row['steps'] != 800-initial['tick'] or
                len(trace) != row['steps'] or
                any(x['tick'] != initial['tick']+i for i, x in enumerate(trace)) or
                {k: v for k, v in detail.items() if k not in ['initial_actual_state', 'trajectory']} != row):
            raise ValueError('New reference lacks complete physically checked stable feedback')
        files.append(path); checks.append(row)
    pins = {}
    for data in [parent, baseline, bank, preservation, inputs]:
        for filename, digest in data['source_sha256'].items():
            if sha(filename) != digest or (filename in pins and pins[filename] != digest):
                raise ValueError('Changed or conflicting input '+filename)
            pins[filename] = digest
    pins.update({str(p.resolve()): sha(p) for p in files})
    return parent, cfg, pins, checks, preservation['checks']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['baseline', 'parent-plan', 'proposal', 'preservation', 'feedback', 'output', 'native-output']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    if args.native_output.exists() and not args.check_only:
        raise ValueError('Preserve existing native execution')
    parent, cfg, pins, checks, protected = evidence(args)
    command = list(parent['command'])
    old_entry = next(Path(x) for x in command if x.endswith('/evaluate_pure_control.py'))
    entry = args.output/'evaluate_pure_control.py'; configuration = args.output/'controller.json'
    if not args.check_only:
        args.output.mkdir(parents=True, exist_ok=False)
        entry.write_bytes(old_entry.read_bytes()); configuration.write_bytes(args.proposal.read_bytes())
    if sha(entry) != sha(old_entry) or sha(configuration) != sha(args.proposal):
        raise ValueError('Native entry or checked configuration changed')
    text = entry.read_text(); ast.parse(text)
    call = 'action=native_replay.action(tick,obs,action,finished)'
    if text.count(call) != 1 or 'native_pure_control_preserved_foot_replanning_v20' not in text:
        raise ValueError('Incorrect final command owner')
    block = text[text.index(call):text.index('observation, _, done, _ = wrapped.step(action)')]
    if 'action = ' in block or 'action=' in block.split('\n', 1)[1]:
        raise ValueError('Another writer follows pure control')
    pins.update({str(p.resolve()): sha(p) for p in [entry, configuration]})
    receipt = args.receipt if args.check_only else args.output/'prepared_entry_check.json'
    if receipt is None or receipt.exists(): raise ValueError('Fresh receipt required')
    receipt.write_text(json.dumps(dict(passed=True, new_reference_checks=checks,
        protected_command_checks=protected, source_sha256=pins, entry_sha256=sha(entry),
        native_execution=False, native_success=False, valid_expert_labels=False,
        single_final_writer=True, hardware_unchanged=True), indent=2)+'\n')
    if args.check_only:
        print(json.dumps(dict(passed=True, new_references=len(checks), protected_cases=len(protected)))); return
    command = [str(entry.resolve()) if x == str(old_entry) else x for x in command]
    command[command.index('--max-seconds')+1] = '7200'
    for i, item in enumerate(command[:-1]):
        if item == '--output': command[i+1] = str(args.native_output.resolve())
        if item == '--pure-control-configuration': command[i+1] = str(configuration.resolve())
    check = ['bash', 'scripts/python.sh', 'scripts/prepare_combined_initial_native_v1.py']
    for name in ['baseline', 'parent-plan', 'proposal', 'preservation', 'feedback', 'output', 'native-output']:
        check += ['--'+name, str(getattr(args, name.replace('-', '_')).resolve())]
    check += ['--check-only', '--receipt', str((args.output/'entry_check.json').resolve())]
    for p in [receipt, ROOT/'scripts/run_native_pure_control_long_v3.py', ROOT/'scripts/run_serial_pure_control_current_v4.py']:
        pins[str(p.resolve())] = sha(p)
    plan = dict(parent, command=command, source_sha256=pins, entry_sha256=sha(entry),
        scope=__doc__, output=str(args.native_output.resolve()), parent_plan=str(args.parent_plan.resolve()),
        declared_max_native_seconds=7200, check_command=check,
        post_command=['bash', 'scripts/python.sh', 'scripts/audit_native_pure_control_v2.py',
            '--run', str(args.native_output.resolve()), '--output', str((args.output/'native_audit.json').resolve())],
        model_accepted=False, training=False, valid_expert_labels=False)
    (args.output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    print(json.dumps(dict(plan=str(args.output/'plan.json'), native_execution=False)))


if __name__ == '__main__':
    main()
