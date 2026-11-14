#!/usr/bin/env python3
"""Run one evaluation and retain SIGSEGV/nonzero/incomplete outcome evidence.

Pass the same arguments as evaluate_supermimic_v1.py, with the Isaac interpreter.
No automatic retries. stdout/stderr inherit the caller's logging destination.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quadsmash.rl.supermimic_run_lifecycle import inspect_process_result


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output', type=Path, required=True)
    args, _ = parser.parse_known_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('Refuse to reuse evaluation output: ' + str(output))
    command = [sys.executable, '-u', str(ROOT/'scripts/evaluate_supermimic_v1.py'), *sys.argv[1:]]
    child = subprocess.run(command)
    result = inspect_process_result(output, child.returncode)
    # The child may fail before it creates its output. Preserve that failure too.
    output.mkdir(parents=True, exist_ok=True)
    result['command'] = command
    (output/'process_result.json').write_text(json.dumps(result, indent=2)+'\n')
    print('SUPERMIMIC_PROCESS_RESULT='+json.dumps(result), flush=True)
    return result['diagnostic_exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
