"""Calibrate first-contact incidence on native training lanes and score a frozen fit."""
import argparse
import json
from pathlib import Path

import torch


from scripts.dynamimic_body_fault import calibrate, scores
from scripts.fit_dynamimic_recovery_wm import read_data


def evaluate(data, coefficients, params, selected):
    tick, lane, logit = scores(data, coefficients)
    hazard = torch.zeros_like(data['active'], dtype=torch.float32)
    hazard[tick, lane] = torch.sigmoid(params[0] * logit + params[1])
    report = dict(expected_onsets=float(hazard[:, selected].sum()),
                  observed_onsets=int((data['active'] & data['body'])[:, selected].sum()))
    for event in ('body', 'strict', 'relaxed'):
        mask = data['active'] & data[event]
        lanes = (mask.any(0) & selected).nonzero().flatten()
        end = mask.int().argmax(0)[lanes]
        # Only full active H48 windows contribute, so a shorter history cannot dilute risk.
        times = end[None, :] - torch.arange(48)[:, None]
        valid = ((times >= 0) & data['active'][times.clamp_min(0), lanes]).all(0)
        risk = 1 - (1 - hazard[times[:, valid], lanes[valid]]).prod(0)
        report[event] = dict(cases=len(lanes), full_windows=len(risk),
                             median_risk=float(risk.quantile(.5)) if len(risk) else None)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('train', type=Path)
    parser.add_argument('score', type=Path)
    args = parser.parse_args()
    extra = ('active', 'strict', 'relaxed')
    data, _ = read_data(args.train, extra)
    coefficients, params = calibrate(data)
    if args.train.resolve() != args.score.resolve():
        data, _ = read_data(args.score, extra)
    held = torch.arange(data['active'].shape[1]) % 16 < 2
    print(json.dumps(dict(calibration=params.tolist(), held=evaluate(data, coefficients, params, held)),
                     indent=2))


if __name__ == '__main__':
    main()
