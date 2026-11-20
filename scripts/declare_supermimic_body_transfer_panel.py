#!/usr/bin/env python3
"""Freeze new launches, candidate identities and protected body metrics before native scoring."""
import argparse
import json
from pathlib import Path

from scripts.evaluate_supermimic_fresh_wm_panel import check_panel, digest
from scripts.freeze_full_racket_protocol_v4 import incoming_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('actor', 'anchor', 'contact-calibration', 'output'):
        parser.add_argument('--'+flag, type=Path, required=True)
    for flag in ('body', 'previous-panel'):
        parser.add_argument('--'+flag, action='append', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--cases-per-panel', type=int, default=16)
    args = parser.parse_args()
    if args.cases_per_panel < 1:
        parser.error('Each launch distribution needs at least one case')
    root = Path(__file__).resolve().parents[1]
    protocol = root/'configs/g07_full_racket_eval_v4.json'
    def weights(paths):
        return {str(p.resolve()):digest(p) for p in paths}
    cases = incoming_cases(args.cases_per_panel,args.seed,'A')
    cases += incoming_cases(args.cases_per_panel,args.seed+1,'B')
    panel = dict(schema='fresh-wm-holdout-v1', policy_training=False, seed=args.seed,
                 horizons=[1,2,4,8,16,32,64], cases=cases, protocol_sha256=digest(protocol),
                 weights=weights([args.anchor,args.actor]), prospective_body_comparison=dict(
                     bodies=weights(args.body), calibration_path=str(args.contact_calibration.resolve()),
                     calibration_sha256=digest(args.contact_calibration), previous_panels=weights(args.previous_panel),
                     transitions=['calibrated_analytic','initial_components'],
                     uniform_no_regression=['body_rmse','root_angular_mae'],
                     contact_gate='supermimic-wm-candidate-gate-v1'))
    check_panel(panel,protocol,root,[args.anchor,args.actor])
    with args.output.open('x') as stream:
        json.dump(panel,stream,indent=2,allow_nan=False)


if __name__ == '__main__':
    main()
