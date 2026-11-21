#!/usr/bin/env python3
"""Evaluate one frozen slip calibration on identical G07 cases at three perturbation amplitudes."""
import argparse
import json
from pathlib import Path

import torch

from calibrate_supermimic_toi_slip import TOISlip, body_transfer
from evaluate_supermimic_contact_rollout import sha
from fit_supermimic_native_racket_toi_head import free_rollout, g07
from quadsmash.rl.supermimic_ball_impulse_prior_v1 import BallImpulsePriorV1


def read(path):
    return json.loads(path.read_text())


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('fit', 'physics-calibration', 'development', 'body', 'online-body', 'source-stage', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    fit = read(args.fit)
    digests = read(args.development/'completed.json')
    if (fit['final_holdout_opened'] or fit['calibration_sha256'] != sha(args.physics_calibration)
            or fit['g07_sha256'] != digests['wide'] or fit['online_body_racket']['base_body_sha256'] != sha(args.body)):
        raise ValueError('Calibration or baseline differs from the wide-scale evaluation')
    prior = BallImpulsePriorV1(normal_impulse_scale=read(args.physics_calibration)['fitted']['normal_impulse_scale']).eval()
    model = TOISlip(fit['selected']['friction'], fit['selected']['sticking'])
    training_lanes = set(fit['g07_training']['commands']['lanes'])
    scores = {}
    stable_ids = []
    reference = None
    for divisor, name in ((1, 'wide'), (2, 'half'), (4, 'quarter')):
        root = args.development/name
        if sha(root/'status.json') != digests[name]:
            raise ValueError('Changed native development collection')
        startup = read(root/'startup.json')
        spec = dict(startup['specs'][0]['spec'])
        epsilon = spec.pop('epsilon_request')
        identity = (startup['source_sha256'], spec, epsilon*divisor)
        if reference is None:
            reference = identity
        if identity != reference or set(spec['lanes']) & training_lanes:
            raise ValueError('Expected the same held lanes, directions, timing and source policy at all scales')
        _, _, noise, groups, seq, cmds = g07(root, prior)
        oracle = free_rollout(model, prior, seq, groups, noise, cmds)
        body = body_transfer(root, args.body, prior, model, groups, noise, seq, cmds, args.online_body, args.source_stage)
        if body['replay_world_sha256'] != fit['online_body_racket']['replay_world_sha256']:
            raise ValueError('Different Body response model')
        if name == 'wide' and (oracle['candidate'] != fit['g07']['candidate'] or body['metrics'] != fit['online_body_racket']['metrics']):
            raise ValueError('Frozen weights do not reproduce initial evaluation')
        observable = {row['lane'] for row in oracle['cases'] if row['mode'] == 'racket_stable'
                      and row['candidate']['cosine'] is not None and row['executed_target_change_rad'] > 0}
        stable_ids.append(observable)
        scores[name] = dict(epsilon_request=epsilon, stable_observable_lanes=sorted(observable),
                            groups={k:groups.count(k) for k in sorted(set(groups))},
                            native_racket=oracle['candidate'], body_racket=body['metrics'])
    result = dict(scope='Previously registered development cases at different amplitudes of the same requested direction',
                  calibration_sha256=sha(args.fit), source_status_sha256=digests, scales=scores,
                  unique_observable_stable_lanes=sorted(set.union(*stable_ids)),
                  observable_and_stable_at_every_scale=sorted(set.intersection(*stable_ids)),
                  optimizer_updates=0, final_holdout_opened=False)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
