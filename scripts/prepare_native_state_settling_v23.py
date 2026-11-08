"""Prepare a separate full TRAIN128 test of state-driven support transition."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.check_native_state_settling_v23 import OLD, NEW
from scripts.retry_budgeted_settling_feedback_v1 import sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['parent-plan', 'feedback', 'fixture-run', 'output', 'native-output']:
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or a.native_output.exists():
        raise ValueError('Fresh native preparation required')
    read = lambda path: json.loads(Path(path).read_text())
    plan = read(a.parent_plan)
    for path, digest in plan['source_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Changed prepared baseline ' + path)
    command = plan['command']
    old = next(Path(x) for x in command if x.endswith('/evaluate_pure_control.py'))
    text = old.read_text()
    if text.count(OLD) != 1:
        raise ValueError('Expected V15 original final owner')
    cfg = read(command[command.index('--pure-control-configuration') + 1])
    cfg.update(controller='pure_control_state_settling_v23',
        method='full18 current-state-driven support transition; unchanged expanded initial reference bank')
    a.output.mkdir(parents=True)
    entry = a.output / 'evaluate_pure_control.py'
    entry.write_text(text.replace(OLD, NEW))
    config = a.output / 'controller.json'
    write(config, cfg)
    command = [str(entry.resolve()) if x == str(old) else x for x in command]
    command[command.index('--max-seconds') + 1] = '7200'
    for i, value in enumerate(command[:-1]):
        if value == '--output':
            command[i + 1] = str(a.native_output.resolve())
        elif value == '--pure-control-configuration':
            command[i + 1] = str(config.resolve())
    check = ['bash', 'scripts/python.sh', 'scripts/check_native_state_settling_v23.py',
        '--parent-plan', str(a.parent_plan.resolve()), '--entry', str(entry.resolve()),
        '--configuration', str(config.resolve()), '--feedback', str(a.feedback.resolve()),
        '--fixture-run', str(a.fixture_run.resolve()), '--output', str((a.output / 'entry_check.json').resolve())]
    subprocess.run([*check[:-1], str((a.output / 'prepared_entry_check.json').resolve())], cwd=ROOT, check=True, timeout=400)
    receipt = read(a.output / 'prepared_entry_check.json')
    plan['source_sha256'].update(receipt['source_sha256'])
    for f in [Path(__file__), a.output / 'prepared_entry_check.json']:
        plan['source_sha256'][str(f.resolve())] = sha(f)
    plan.update(command=command, scope=__doc__, declared_max_native_seconds=7200,
        output=str(a.native_output.resolve()), entry_sha256=sha(entry),
        parent_plan=str(a.parent_plan.resolve()), check_command=check,
        post_command=['bash', 'scripts/python.sh', 'scripts/audit_native_pure_control_v2.py',
            '--run', str(a.native_output.resolve()), '--output', str((a.output / 'native_audit.json').resolve())],
        model_accepted=False, training=False, valid_expert_labels=False,
        old_success_retention_unproven=True)
    write(a.output / 'plan.json', plan)
    print(json.dumps(dict(plan=str(a.output / 'plan.json'), native_execution=False)))


if __name__ == '__main__':
    main()
