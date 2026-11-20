#!/usr/bin/env python3
"""Test propagation of multi-shape racket impulses without supplying the native shuttle at later substeps."""
import argparse
import json
from pathlib import Path

import torch

from audit_supermimic_surface_spin_transfer import evaluate
from evaluate_supermimic_contact_rollout import BallImpulsePriorV1, sha
from quadsmash.rl.supermimic_racket_material_contract_v1 import validate_contract


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ('collection', 'material', 'calibration', 'output'):
        cli.add_argument('--' + key, type=Path, required=True)
    a = cli.parse_args()
    torch.set_num_threads(2)
    provenance = json.loads((a.collection/'nominal_a'/'provenance.json').read_text())
    args = provenance['args']
    teacher = Path(args[args.index('--teacher') + 1])
    if sha(teacher) != provenance['teacher_sha256']:
        raise ValueError('Changed native teacher')
    asset = torch.load(teacher, mmap=True, weights_only=True)['task_contract']['asset']['sha256']
    material = torch.tensor(validate_contract(json.loads(a.material.read_text()), asset)['effective'])
    cfg = json.loads(a.calibration.read_text())
    model = BallImpulsePriorV1().eval()
    splits = ('train', 'development')
    variants = {}
    for split in splits:
        report = a.collection/(split+'.json')
        def score(*, table, spin, slip, compound=False):
            return evaluate(a.collection, report, split, model, cfg, table, spin, slip, compound=compound)
        variants[split] = dict(
            anchor=score(table=None, spin=False, slip=True),
            single=score(table=material, spin=True, slip=False),
            compound=score(table=material, spin=True, slip=False, compound=True))
    decision = {}
    for split, result in variants.items():
        base, candidate = result['anchor'], result['compound']
        v = candidate['stable_signed']
        decision[split] = dict(
            absolute=candidate['absolute_rmse'] <= base['absolute_rmse'],
            all_signed=candidate['all_signed']['absolute_rmse'] <= base['all_signed']['absolute_rmse'],
            stable_improvement=v['absolute_rmse'] < base['stable_signed']['absolute_rmse'],
            direction=v['cosine_median'] >= .8 and v['cosine_p10'] >= .5,
            amplitude=.5 <= v['gain_median'] <= 2.)
    output = dict(scope='Measured native racket, shuttle initialized once per case; spin propagated after impact',
                  results=variants, checks=decision, calibration_sha256=sha(a.calibration), material_sha256=sha(a.material),
                  split_sha256={s:sha(a.collection/(s+'.json')) for s in splits},
                  candidate_eligible_for_reserved_test=all(all(check.values()) for check in decision.values()),
                  final_holdout_opened=False, policy_updates=0, contact_gradients_admitted=False)
    with a.output.open('x') as f:
        json.dump(output, f, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
