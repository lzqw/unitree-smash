"""Prepare a full native comparison with checked actual first-match feedback.

Retains the complete frozen v14 prefix. No case/time gate or physical limit is
changed. This prepares a pinned diagnostic; it does not start or promote it.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['parent-plan', 'replanning-library', 'output', 'native-output',
                 'fixture-run', 'matches', 'feedback']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    read = lambda p: json.loads(Path(p).read_text())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    result = read(args.feedback/'result.json')
    if (result['state'] != 'COMPLETED_FIRST_MATCH_FEEDBACK_MODEL' or
            not result['results'] or
            not all(r['model_terminal_window_passed'] for r in result['results'])):
        raise ValueError('Completed successful actual-trigger model feedback required')
    command = [sys.executable, str(ROOT/'scripts/prepare_native_preserved_stance_replanning_v18.py')]
    for name in ['parent-plan', 'replanning-library', 'output', 'native-output', 'fixture-run']:
        command += ['--'+name, str(getattr(args, name.replace('-', '_')).resolve())]
    subprocess.run(command, cwd=ROOT, check=True)
    entry = args.output/'evaluate_pure_control.py'
    config = args.output/'controller.json'
    checker = ROOT/'scripts/check_first_match_native_entry_v1.py'
    arguments = ['--entry', str(entry.resolve()), '--configuration', str(config.resolve()),
        '--run', str(args.fixture_run.resolve()), '--matches', str(args.matches.resolve()),
        '--feedback', str(args.feedback.resolve())]
    preflight = args.output/'prepared_entry_check.json'
    subprocess.run([sys.executable, str(checker), *arguments, '--output', str(preflight.resolve())],
        cwd=ROOT, check=True)
    plan_path = args.output/'plan.json'
    plan = read(plan_path)
    plan.update(scope=__doc__, declared_max_native_seconds=7200,
        preparation_note='Prepared only; full actual first-match model feedback is not native success.',
        check_command=['bash', 'scripts/python.sh', str(checker), *arguments,
            '--output', str((args.output/'entry_check.json').resolve())])
    # Include both the admission evidence and its transitive source manifest.
    for filename,digest in read(preflight)['source_sha256'].items():
        if sha(filename) != digest:
            raise ValueError('Changed preflight source '+filename)
        if filename in plan['source_sha256'] and plan['source_sha256'][filename] != digest:
            raise ValueError('Conflicting preflight source '+filename)
        plan['source_sha256'][filename] = digest
    for path in [Path(__file__), checker, preflight]:
        plan['source_sha256'][str(path.resolve())] = sha(path)
    temporary = plan_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(plan, indent=2, allow_nan=False)+'\n')
    temporary.replace(plan_path)
    print(json.dumps(dict(plan=str(plan_path), native_execution=False,
        actual_first_match_verified=True, max_native_seconds=7200)))


if __name__ == '__main__':
    main()
