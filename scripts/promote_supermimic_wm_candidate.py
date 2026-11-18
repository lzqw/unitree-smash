#!/usr/bin/env python3
"""Promote a frozen WM candidate only after held-out error checks pass.

This command does not run Isaac or train a model.  It copies a candidate only
when the validation JSON contains the same-window anchor comparison and the
candidate passes the protected contact metrics.  Failure leaves the candidate
untouched and records a rejection decision.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

from quadsmash.rl.supermimic_wm_candidate_gate_v1 import compare_candidate_to_anchor


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation-json', type=Path, required=True)
    parser.add_argument('--candidate-name', required=True)
    parser.add_argument('--candidate-checkpoint', type=Path, required=True)
    parser.add_argument('--output-decision', type=Path, required=True)
    parser.add_argument('--promoted-checkpoint', type=Path)
    args = parser.parse_args()
    if args.output_decision.exists():
        raise FileExistsError(args.output_decision)
    report = json.loads(args.validation_json.read_text())
    results = report.get('results', {})
    if 'initial' not in results or args.candidate_name not in results:
        raise ValueError('validation JSON must contain initial and candidate results')
    decision = compare_candidate_to_anchor(results['initial'], results[args.candidate_name])
    decision.update(
        validation_json=str(args.validation_json.resolve()),
        candidate_name=args.candidate_name,
        candidate_checkpoint=str(args.candidate_checkpoint.resolve()),
        candidate_sha256=sha256(args.candidate_checkpoint),
        native_simulation=bool(report.get('native_simulation', False)),
    )
    if decision['passed']:
        if args.promoted_checkpoint is None:
            raise ValueError('A passed candidate requires --promoted-checkpoint')
        if args.promoted_checkpoint.exists():
            raise FileExistsError(args.promoted_checkpoint)
        args.promoted_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.candidate_checkpoint, args.promoted_checkpoint)
        decision['promoted_checkpoint'] = str(args.promoted_checkpoint.resolve())
        decision['promoted_sha256'] = sha256(args.promoted_checkpoint)
        decision['promotion_status'] = 'heldout_pass'
    else:
        decision['promotion_status'] = 'rejected_heldout_degradation_or_missing_evidence'
    args.output_decision.parent.mkdir(parents=True, exist_ok=True)
    args.output_decision.write_text(json.dumps(decision, indent=2) + '\n')
    print(json.dumps(decision, indent=2))
    if not decision['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
