"""Compare signed racket response for a Body overlay and its exact parent."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from fit_supermimic_approach_body_response import body_predictions
from fit_supermimic_precontact_body import sha256
from quadsmash.rl.supermimic_physical_loop_loading_v1 import tensor_digest
from quadsmash.rl.supermimic_physical_world_v2 import PhysicalWorldV2
from quadsmash.rl.supermimic_policy_execution_v1 import contract
from quadsmash.rl.supermimic_racket_dynamics_v1 import racket_state
from quadsmash.rl.supermimic_training_probe_replay import training_windows
from run_supermimic_approach_v2 import dump


@torch.no_grad()
def response(body, windows, launch):
    horizons = (4, 8, 16)
    terms = dict(center=[0, 1, 2], normal=[3, 6, 9], linear_velocity=[12, 13, 14],
                 angular_velocity=[15, 16, 17])
    samples = {horizon: {name: [] for name in terms} for horizon in horizons}
    for lanes, signed in windows:
        trajectories = {
            name: [(racket_state(estimate, body.geometry), racket_state(actual, body.geometry))
                   for estimate, actual in body_predictions(body, signed[name], launch[list(lanes)])]
            for name in ('minus', 'plus')
        }
        for horizon in horizons:
            predicted, measured = [plus - minus for minus, plus in zip(
                trajectories['minus'][horizon - 1], trajectories['plus'][horizon - 1], strict=True)]
            for name, indices in terms.items():
                samples[horizon][name].append((predicted[:, indices], measured[:, indices]))
    return {h: {name: summarize(rows) for name, rows in channels.items()} for h, channels in samples.items()}


def summarize(rows):
    predicted, measured = (torch.cat(part) for part in zip(*rows, strict=True))
    signal = measured.norm(dim=-1) > 1e-7
    error = (predicted - measured).norm(dim=-1)
    if not torch.any(signal):
        return dict(samples=len(error), directional_samples=0, mean_error=float(error.mean()))
    cosine = F.cosine_similarity(predicted[signal], measured[signal])
    gain = predicted[signal].norm(dim=-1) / measured[signal].norm(dim=-1)
    return dict(samples=len(error), directional_samples=int(signal.sum()), mean_error=float(error.mean()),
                mean_direction_cosine=float(cosine.mean()), median_gain=float(gain.median()))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for option in ('checkpoint', 'candidate', 'output'):
        cli.add_argument('--' + option, type=Path, required=True)
    args = cli.parse_args()
    torch.set_num_threads(4)
    base = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    candidate = torch.load(args.candidate, map_location='cpu', weights_only=True)
    if (sha256(args.checkpoint) != candidate['parent_checkpoint_sha256']
            or tensor_digest(candidate['model'].items()) != candidate['body_sha256']):
        raise ValueError('Body candidate identity does not match its parent or its weights')
    task = base['metadata']['task_contract']
    suite = Path(task['suite']['path'])
    if sha256(suite) != task['suite']['sha256']:
        raise ValueError('Launch suite changed')
    launch = torch.tensor([case['launch']['position_m'] + case['launch']['velocity_mps']
                           for case in json.loads(suite.read_text())['cases']])
    body = PhysicalWorldV2(task['asset']['sha256']).eval()
    body.load_state_dict({k.removeprefix('physical_model.'): v for k, v in base['world'].items()
                          if k.startswith('physical_model.')}, strict=True)
    physics_contract = {key: value for key, value in task.items() if key != 'approach_ball_scoring'}
    windows = []
    for source in base['metadata']['wm_action_collections']:
        pairs, receipt = training_windows(Path(source['path']), physics_contract, contract(),
                                          base['config']['wm_holdout_lanes'])
        if (receipt['status_sha256'], receipt['spec_branches']) != (source['status_sha256'], source['spec_branches']):
            raise ValueError('Registered signed command collection changed')
        windows.extend(pairs)
    before = response(body, windows, launch)
    body.body.load_state_dict(candidate['model'], strict=True)
    after = response(body, windows, launch)
    dump(args.output, dict(candidate_body_sha256=candidate['body_sha256'], windows=len(windows),
                           baseline=before, candidate=after))


if __name__ == '__main__':
    main()
