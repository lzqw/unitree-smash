#!/usr/bin/env python3
"""Evaluate a frozen analytic first-impact model on all trials of a new teacher prefix."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from calibrate_supermimic_pilot_oracle import BallImpulsePriorV1, join, predict, score, trajectory


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pilot', 'calibration', 'capture', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    frozen = json.loads(args.calibration.read_text())
    if digest(args.pilot) != frozen['pilot_sha256'] or frozen['solver']['clock'] != 'pre_step':
        raise ValueError('Model and training data identity mismatch')
    pilot = torch.load(args.pilot, map_location='cpu', weights_only=True)
    sources = {Path(row['source']).parent.parent for row in pilot['rows']}
    training_teachers = {json.loads((src/'provenance.json').read_text())['teacher_sha256'] for src in sources}
    meta = json.loads((args.capture/'provenance.json').read_text())
    if meta['teacher_sha256'] in training_teachers:
        raise ValueError('Teacher archive overlaps pilot')
    teacher = Path(meta['args'][meta['args'].index('--teacher') + 1])
    if digest(teacher) != meta['teacher_sha256']:
        raise ValueError('Teacher archive changed')
    paths = [args.capture/f'spec_{i}'/'trace.pt' for i in range(meta['seed'], meta['seed']+meta['batches'])]
    if set(paths) != set(args.capture.glob('spec_*/trace.pt')):
        raise ValueError('Missing or unexpected validation batch')
    prior = BallImpulsePriorV1().eval()
    rows, files = [], []
    for seed, path in enumerate(paths, meta['seed']):
        archive = torch.load(path, weights_only=True, map_location='cpu')
        if archive['seed'] != seed or archive['parent_lane'].shape != (meta['lanes'],):
            raise ValueError('Unexpected validation trial identity or batch size')
        trial = trajectory(archive, prior)
        rows.append(predict(trial, prior, frozen['solver']['pair_contact_offset_m'], solver=True))
        files.append(dict(seed=seed, source_sha256=digest(path)))
    fitted = frozen['fitted']
    metrics = score(join(rows), fitted['normal_impulse_scale'], fitted['friction'])
    report = dict(scope='Frozen native-racket first impact on every trial from a different teacher prefix',
                  teacher=str(teacher), teacher_sha256=digest(teacher), pilot_teachers=sorted(training_teachers),
                  calibration_sha256=digest(args.calibration), capture_path=str(args.capture),
                  files=files, metrics=metrics, policy_training_admitted=False,
                  contact_bptt='detached', v13_residual_gradient='detached')
    with args.output.open('x') as stream:
        json.dump(report, stream, allow_nan=False, indent=2)
        stream.write('\n')
    print(json.dumps(metrics))


if __name__ == '__main__':
    main()
