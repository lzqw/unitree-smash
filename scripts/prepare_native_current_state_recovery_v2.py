"""Prepare a separately versioned native run with shared state representation."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['previous-preparation', 'previous-bundle', 'check', 'output', 'bundle-output']:
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if args.output.exists() or args.bundle_output.exists():
        raise ValueError('Fresh versioned outputs required')
    check = json.loads(args.check.read_text())
    if (check['native_execution'] or not check['exact_command_check_retained'] or
            check['physical_or_task_thresholds_changed'] or
            [r['control_tick'] for r in check['rows']] != [400, 550] or
            not all(r['corrected_command_exact'] and r['contradictory_state_rejected'] for r in check['rows'])):
        raise ValueError('Executed native-evidence representation regression required')
    previous = json.loads((args.previous_preparation / 'preparation.json').read_text())
    m = json.loads((args.previous_bundle / 'manifest.json').read_text())
    for pins in [m['source_sha256'], check['source_sha256']]:
        for path, expected in pins.items():
            if sha(path) != expected:
                raise ValueError('Changed admitted input: ' + path)
    if sha(args.previous_bundle / m['bundle_name']) != m['bundle_sha256']:
        raise ValueError('Changed original recovery bundle')
    source_entry, source_audit = Path(previous['entry']), Path(previous['audit'])
    if sha(source_entry) != previous['entry_sha256'] or sha(source_audit) != previous['audit_sha256']:
        raise ValueError('Previous preparation changed')
    text = source_entry.read_text()
    old = 'from scripts.native_current_state_recovery_v1 import NativeCurrentStateRecoveryV1 as NativeRecoverySequenceReplay'
    if text.count(old) != 1:
        raise ValueError('Unexpected native controller selection')
    text = text.replace(old,
        'from scripts.native_current_state_recovery_v2 import NativeCurrentStateRecoveryV2 as NativeRecoverySequenceReplay')
    ast.parse(text)
    args.output.mkdir(parents=True); args.bundle_output.mkdir(parents=True)
    entry = args.output / 'evaluate_current_state_v2.py'; entry.write_text(text)
    audit = args.output / 'audit_current_state_v2.py'; shutil.copyfile(source_audit, audit)
    m['current_state_planning']['state_representation'] = 'q/v-derived fields identical in planning and execution'
    files = [Path(__file__), args.check, args.previous_bundle / 'manifest.json',
        args.previous_preparation / 'preparation.json', root / 'scripts/native_current_state_recovery_v2.py']
    m['source_sha256'].update(check['source_sha256'])
    m['source_sha256'].update({str(p.resolve()): sha(p) for p in files})
    shutil.copyfile(args.previous_bundle / m['bundle_name'], args.bundle_output / m['bundle_name'])
    (args.bundle_output / 'manifest.json').write_text(json.dumps(m, indent=2) + '\n')
    result = dict(entry=str(entry.resolve()), audit=str(audit.resolve()), entry_sha256=sha(entry),
        audit_sha256=sha(audit), bundle=str(args.bundle_output.resolve()),
        bundle_manifest_sha256=sha(args.bundle_output / 'manifest.json'),
        source_sha256=m['source_sha256'], representation_check=str(args.check.resolve()),
        unchanged_plan_schedule=m['current_state_planning']['control_ticks'],
        scope=__doc__, native_execution=False, task_thresholds_changed=False,
        command_comparison_relaxed=False, valid_expert_labels=False)
    (args.output / 'preparation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['entry', 'audit', 'bundle']}))


if __name__ == '__main__':
    main()
