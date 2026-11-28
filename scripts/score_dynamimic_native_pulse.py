"""Compare one native Recovery intervention with unchanged PPO01 continuation."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from scripts.fit_dynamimic_recovery_critic import NATIVE_RETURN_SCHEMA, gae, make_tail, read_rollout


@torch.no_grad()
def score(args):
    before, a = read_rollout(args.baseline, args.checkpoint)
    after, b = read_rollout(args.intervention, args.checkpoint,
                            expected_schema='dynamimic-native-recovery-pulse-v1')
    for key in ('cases', 'companion_sha256', 'origins', 'dt', 'lane_approach', 'episode_indices'):
        if a[key] != b[key]:
            raise ValueError(f'Native conditions differ in {key}')
    for name in ('initial_states.json', 'actuator_contract.json', 'task_contract.json'):
        if json.loads((args.baseline/name).read_text()) != json.loads((args.intervention/name).read_text()):
            raise ValueError(f'Native conditions differ in {name}')
    tick, horizon = b['pulse']['start_tick'], b['pulse']['duration_ticks']
    stop = tick + horizon
    n = len(a['cases'])
    exact = before['active'][tick] & after['active'][tick]
    offset = 0
    for name_a, name_b in zip(a['chunks'], b['chunks']):
        blocks = [torch.load(path/'trajectories'/name, mmap=True, weights_only=True)
                  for path, name in ((args.baseline, name_a), (args.intervention, name_b))]
        if len(blocks[0]['active']) != len(blocks[1]['active']) and offset + min(len(x['active']) for x in blocks) <= tick:
            raise ValueError('Prefix chunks do not align')
        length = len(blocks[0]['active'])
        for key in ('before', 'final_joint_targets', 'request', 'obs'):
            end = min(length, tick - offset + int(key in ('before', 'obs')))
            if end > 0:
                equal = blocks[0][key][:end] == blocks[1][key][:end]
                exact &= equal.flatten(2).all(-1).all(0)
        offset += length
        if offset > tick:
            break
    if offset <= tick:
        raise ValueError('Both native archives must include the intervention boundary')
    tail, source = make_tail(args.checkpoint, 'cpu')
    weights = torch.load(args.critic/'critic.pt', weights_only=True, map_location='cpu')
    fit = json.loads((args.critic/'fit.json').read_text())
    if (fit['native_checkpoint'] != a['checkpoint_sha256'] or weights['source'] != source
            or weights.get('return_schema') != NATIVE_RETURN_SCHEMA):
        raise ValueError('Native tail checkpoint mismatch')
    tail.load_native_fit(weights)
    predictions, returns = [], []
    for data in (before, after):
        reward, live = data['reward'], data['active']
        full = gae(reward, torch.zeros_like(reward), live, data['done']|data['body']|data['net'], lam=1.)
        prefix = (.995**torch.arange(horizon)[:,None] * reward[tick:stop] * live[tick:stop]).sum(0)
        predictions.append(prefix + .995**horizon * live[stop] * tail(data['obs'][stop]))
        returns.append(full[tick])
    pred, native = predictions[1]-predictions[0], returns[1]-returns[0]
    report = dict(pulse=b['pulse'], incumbent=a['checkpoint_sha256'], eligible=int(exact.sum()),
                  sha256_critic=hashlib.sha256((args.critic/'critic.pt').read_bytes()).hexdigest(), groups={})
    held = torch.arange(n)%16 < 2
    for label, split in [('training', ~held), ('held', held)]:
        sel = exact & split
        x, y = pred[sel], native[sel]
        large = y.abs() > 1
        report['groups'][label] = dict(cases=int(sel.sum()), changes_over_one_return_unit=int(large.sum()),
            correct_direction=int((x[large]*y[large]>0).sum()),
            predicted_rms=float(x.square().mean().sqrt()) if len(x) else None,
            native_rms=float(y.square().mean().sqrt()) if len(y) else None,
            body_before=int(before['body'][tick:].logical_and(before['live'][tick:]).any(0)[sel].sum()),
            body_after=int(after['body'][tick:].logical_and(after['live'][tick:]).any(0)[sel].sum()))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'intervention', 'checkpoint', 'critic', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(score(args), indent=2)+'\n')
