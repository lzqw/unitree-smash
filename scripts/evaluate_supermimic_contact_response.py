#!/usr/bin/env python3
"""Compare frozen contact rules against measured normal-or-friction response on all lanes."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from calibrate_supermimic_pilot_oracle import BallImpulsePriorV1, join, normal_after_sliding, predict, score, trajectory
from evaluate_supermimic_contact_prefix import digest


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pilot', 'calibration', 'external', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    pilot = torch.load(args.pilot, weights_only=True, map_location='cpu')
    frozen = json.loads(args.calibration.read_text())
    if pilot['split']['reference_pilot_sha256'] != frozen['pilot_sha256']:
        raise ValueError('Held-out source assignments differ from original calibration')
    sources = {}
    for row in pilot['rows']:
        path = Path(row['source'])
        ref = (row['split'], row['source_sha256'])
        if sources.setdefault(path, ref) != ref:
            raise ValueError('Split overlap or changed source')
    prefix = json.loads(args.external.read_text())
    if prefix['calibration_sha256'] != digest(args.calibration) or prefix['teacher_sha256'] in prefix['pilot_teachers']:
        raise ValueError('Unexpected external teacher identity')
    for entry in prefix['files']:
        path = Path(prefix['capture_path'])/f"spec_{entry['seed']}"/'trace.pt'
        if path in sources:
            raise ValueError('External evaluation overlaps pilot')
        sources[path] = ('external_prefix', entry['source_sha256'])
    prior = BallImpulsePriorV1().eval()
    params = frozen['fitted']
    groups = defaultdict(list)
    for path, (split, sha) in sorted(sources.items()):
        if digest(path) != sha:
            raise ValueError(f'Archive changed: {path}')
        native = torch.load(path, weights_only=True, map_location='cpu')
        trial = trajectory(native, prior, response_truth=True)
        native_normal_trial = trajectory(native, prior)
        groups[split, 'native_task_normal_after_sliding'].append(normal_after_sliding(
            native_normal_trial, prior, frozen['solver']['pair_contact_offset_m'], params['friction']))
        for rule, sliding in [('normal_rule', None), ('separating_overlap_friction', params['friction'])]:
            groups[split, rule].append(predict(trial, prior, frozen['solver']['pair_contact_offset_m'],
                                               solver=True, sliding_friction=sliding))
    results = {}
    for (split, rule), trials in groups.items():
        results.setdefault(split, {})[rule] = score(join(trials), params['normal_impulse_scale'], params['friction'])
    report = dict(scope='Measured racket response when normal or tangential force exceeds 0.001 N; native_task_normal_after_sliding uses the unchanged normal-only task label',
                  source_pilot_sha256=digest(args.pilot), calibration_sha256=digest(args.calibration),
                  external_receipt_sha256=digest(args.external), results=results, policy_training_admitted=False)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
