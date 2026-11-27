"""Calibrate first-contact incidence on native training lanes and score a frozen fit."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from scripts.dynamimic_body_fault import features, fit_native_body_fault
from scripts.fit_dynamimic_recovery_wm import read_data
from quadsmash.rl.dynamimic_recovery_wm_v1 import local_robot_state


def scores(data, coefficients):
    tick, lane = data['active'].nonzero().T
    q, v = local_robot_state(data['after'][tick, lane])
    x = features(q, v, data['feet'][tick, lane])
    return tick, lane, x @ coefficients[:-1] + coefficients[-1]


def calibrate(data):
    coefficients, _ = fit_native_body_fault(data)
    tick, lane, logit = scores(data, coefficients)
    train = lane % 16 >= 2
    y = data['body'][tick[train], lane[train]].float()
    params = torch.tensor([.5, -5.], requires_grad=True)
    optimizer = torch.optim.LBFGS([params], max_iter=80, line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(params[0] * logit[train] + params[1], y)
        loss.backward()
        return loss

    optimizer.step(closure)
    return coefficients, params.detach()


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
