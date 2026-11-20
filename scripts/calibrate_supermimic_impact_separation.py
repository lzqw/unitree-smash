#!/usr/bin/env python3
"""Test within-substep gap sensitivity of the normal impulse on disjoint native launch families."""
import argparse
import json
from pathlib import Path

import torch

from audit_supermimic_surface_spin_transfer import evaluate
from evaluate_supermimic_contact_rollout import BallImpulsePriorV1, sha
from quadsmash.rl.supermimic_racket_material_contract_v1 import validate_contract


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('collection', 'material', 'calibration', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    cfg = json.loads(a.calibration.read_text())
    prov = json.loads((a.collection/'nominal_a/provenance.json').read_text())
    teacher = Path(prov['args'][prov['args'].index('--teacher') + 1])
    if sha(teacher) != prov['teacher_sha256']:
        raise ValueError('Changed native teacher')
    asset = torch.load(teacher, mmap=True, weights_only=True)['task_contract']['asset']['sha256']
    material = torch.tensor(validate_contract(json.loads(a.material.read_text()), asset)['effective'])
    prior = BallImpulsePriorV1().eval()
    def score(split, gain):
        settings = dict(cfg, fitted=dict(cfg['fitted'], separation_velocity_gain=gain))
        return evaluate(a.collection, a.collection/(split+'.json'), split, prior, settings, material,
                        True, False, compound=True)

    gains = [0., .025, .05, .1, .2, .4, .8, 1.]
    train = {str(g):score('train', g) for g in gains}
    base = train['0.0']
    def ok(r):
        return (r['absolute_rmse'] <= base['absolute_rmse'] and
                r['all_signed']['absolute_rmse'] <= base['all_signed']['absolute_rmse'])
    choice = min((g for g in gains if ok(train[str(g)])), key=lambda g:train[str(g)]['balanced_loss'])
    dev = {str(g):score('development', g) for g in dict.fromkeys((0., choice))}
    with a.output.open('x') as f:
        json.dump(dict(train=train, selected_gain=choice, development=dev, material_sha256=sha(a.material),
                       source_calibration_sha256=sha(a.calibration), split_hashes={k:sha(a.collection/(k+'.json'))
                       for k in ('train', 'development')}, final_holdout_opened=False, policy_updates=0,
                       scope='Fit separation divided by physics dt at predicted contacts using calibration families only',
                       contact_gradients_admitted=False), f, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
