"""Run the training-side native check for an opt-in resident Approach boundary."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from subprocess import check_output

from run_supermimic_approach_v2 import dump, paired_quick_result, quick_check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('boundary', 'baseline-policy', 'baseline-panel', 'execution-root'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--seed', type=int, default=882072)
    parser.add_argument('--required-free-mib', type=int, required=True,
                        help='Measured standalone evaluator peak GPU allocation plus headroom')
    args = parser.parse_args()
    if args.required_free_mib <= 0:
        parser.error('Provide a positive measured GPU memory requirement')
    directory = args.boundary.resolve(strict=True)
    boundary = json.loads((directory / 'ready.json').read_text())
    os.kill(boundary['training_pid'], 0)
    checkpoint = directory / 'policy.pt'
    with checkpoint.open('rb') as stream:
        if (checkpoint != Path(boundary['checkpoint']) or
                hashlib.file_digest(stream, 'sha256').hexdigest() != boundary['checkpoint_sha256']):
            raise ValueError('Resident checkpoint identity changed')
    if not (args.baseline_panel / 'completed.json').is_file():
        raise ValueError('Prepare a complete TRAIN64 baseline before using a resident evaluator')
    free = check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                        text=True).splitlines()
    if len(free) != 1 or min(int(free[0]), boundary['gpu']['free_bytes'] // 2**20) < args.required_free_mib:
        raise RuntimeError('This evaluator requires sufficient measured headroom on a single GPU')
    baseline = quick_check(args.baseline_policy, args.baseline_panel, str(args.execution_root), args.seed)
    panel = directory / 'quick64'
    candidate = quick_check(checkpoint, panel, str(args.execution_root), args.seed)
    pair = paired_quick_result(baseline, candidate, args.baseline_panel, panel)
    os.kill(boundary['training_pid'], 0)
    evidence = directory / 'paired.json'
    dump(evidence, dict(baseline=baseline, candidate=candidate, paired=pair))
    with evidence.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    dump(directory / 'decision.json', dict(update=boundary['update'],
        checkpoint_sha256=boundary['checkpoint_sha256'], paired_result_sha256=digest,
        action='stop' if pair['regressed'] else 'continue'))


if __name__ == '__main__':
    main()
