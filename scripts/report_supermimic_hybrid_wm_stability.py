#!/usr/bin/env python3
"""Summarize a frozen hybrid ball-WM fit without promoting it.

The report requires a contact-edge improvement, no one-step degradation at the
adjacent event ticks, and no adjacent multi-step degradation. It does not
issue a native receipt and never changes a checkpoint.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scripts.fit_supermimic_action_ball import (
    autoregressive_validation_selection_score,
    event_validation_selection_score,
)


def _finite(value, name):
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f'Nonfinite or missing metric: {name}')
    return float(value)


def summarize(comparison: dict) -> dict:
    required = ('initial', 'candidate', 'initial_autoregressive', 'candidate_autoregressive')
    missing = [name for name in required if name not in comparison]
    if missing:
        raise ValueError(f'Missing comparison fields: {missing}')
    initial = comparison['initial']
    candidate = comparison['candidate']
    initial_ar = comparison['initial_autoregressive']
    candidate_ar = comparison['candidate_autoregressive']
    event = event_validation_selection_score(initial, candidate, adjacent_tolerance=1e-6)
    autoregressive = autoregressive_validation_selection_score(
        initial_ar, candidate_ar, adjacent_tolerance=1e-6)
    for offset in ('-1', '0', '1'):
        _finite(initial[offset]['velocity_rmse'], f'initial/{offset}')
        _finite(candidate[offset]['velocity_rmse'], f'candidate/{offset}')
    contact_improved = candidate['0']['velocity_rmse'] < initial['0']['velocity_rmse'] - 1e-6
    candidate_sha = comparison.get('candidate_sha256')
    if not isinstance(candidate_sha, str) or len(candidate_sha) != 64:
        raise ValueError('Candidate checkpoint hash is missing')
    accepted = bool(event['eligible'] and autoregressive['eligible'] and contact_improved)
    if not contact_improved:
        reason = 'contact_edge_not_improved'
    elif not event['eligible']:
        reason = event['reason']
    elif not autoregressive['eligible']:
        reason = autoregressive['reason']
    else:
        reason = 'stable_contact_neighborhood_candidate'
    return dict(
        schema='supermimic-hybrid-wm-stability-v1',
        prediction_status='STABLE_CANDIDATE' if accepted else 'NO_STABLE_CANDIDATE',
        accepted=accepted,
        reason=reason,
        candidate_sha256=candidate_sha,
        actor_updates=int(comparison.get('actor_updates', 0)),
        world_updates=int(comparison.get('world_updates', 0)),
        critic_updates=int(comparison.get('critic_updates', 0)),
        one_step=dict(
            initial={offset: initial[offset]['velocity_rmse'] for offset in ('-1', '0', '1')},
            candidate={offset: candidate[offset]['velocity_rmse'] for offset in ('-1', '0', '1')},
            event_selection=event,
            contact_improved=contact_improved,
        ),
        autoregressive=dict(
            selection=autoregressive,
            initial=initial_ar,
            candidate=candidate_ar,
        ),
        native_receipt_issued=False,
        scope='offline recorded-command prediction; no PhysX action perturbation or Actor update',
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = summarize(json.loads(args.comparison.read_text()))
    report['comparison_sha256'] = __import__('hashlib').sha256(
        args.comparison.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not report['accepted']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
