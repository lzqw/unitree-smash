"""Compare same-policy Critics with measured RTG on either side of the first post-contact landing."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch

from quadsmash.rl.supermimic_horizon_learner_v2 import optimizer_step
from quadsmash.rl.supermimic_evidence_binding_v1 import complete_native_rtg
from quadsmash.rl.supermimic_physical_loop_loading_v1 import tensor_digest
from quadsmash.rl.supermimic_v1 import MLP


def select_landing_rows(ticks, touchdown, max_distance):
    """Select the closest recorded start on each side of the support transition."""
    pre = [t for t in ticks if 0 <= touchdown - t <= max_distance]
    post = [t for t in ticks if 0 < t - touchdown <= max_distance]
    return {'before_support': max(pre, default=None), 'after_support': min(post, default=None)}


def ranking(values, success):
    differences = values[success][:, None] - values[~success][None, :]
    if not differences.numel():
        return None
    return float(((differences > 0).float() + .5 * (differences == 0)).mean())


def offline_comparison(checkpoint_path, inputs, targets, general, landing_rows, steps=8, readout_path=None):
    """Use identical starting parameters and Adam slots for both offline fit arms."""
    checkpoint = torch.load(checkpoint_path, mmap=True, map_location='cpu', weights_only=False)
    generator = torch.Generator().manual_seed(3117)
    def draw(pool, n):
        return pool[torch.randint(len(pool), (n,), generator=generator)]
    batches = []
    for _ in range(steps):
        shared = draw(general, 64)
        batches.append((torch.cat([shared, draw(general, 64)]),
                        torch.cat([shared, *(draw(group, 32) for group in landing_rows)])))
    predictions = {}
    for arm, name in enumerate(('uniform_rtg_fit', 'landing_rtg_fit')):
        model = MLP(inputs.shape[-1], (512, 256, 128), 1)
        model.load_state_dict(checkpoint['critic'])
        optimizer = torch.optim.Adam(model.parameters())
        # Optimizer tensors must not alias the initial checkpoint between arms.
        optimizer.load_state_dict(copy.deepcopy(checkpoint['critic_optimizer']))
        for batch in batches:
            ids = batch[arm]
            loss = (model(inputs[ids]).flatten() - targets[ids]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer_step(optimizer, model.parameters())
        with torch.no_grad():
            predictions[name] = torch.cat([model(x).flatten() for x in inputs.split(128)])
    # Solve only the last affine layer against complete native returns. Each phase
    # has fixed mass. Regularize slopes to the inherited readout in feature units.
    model = MLP(inputs.shape[-1], (512, 256, 128), 1).eval()
    model.load_state_dict(checkpoint['critic'])
    with torch.no_grad():
        features = torch.cat([model.net[:-1](x).double() for x in inputs.split(128)])
        base = model(inputs).flatten().double()
        ids = torch.cat([general, *landing_rows])
        weights = torch.cat([torch.full((len(group),), mass/len(group), dtype=torch.double)
                             for mass,group in zip((.5,.25,.25), (general,*landing_rows),strict=True)])
        z = features[ids]
        residual = targets[ids].double()-base[ids]
        center = weights @ z
        bias = weights @ residual
        z = z-center
        cov = z.T @ (weights[:,None]*z)
        ridge = cov.trace()/cov.shape[0]
        if ridge <= 0:
            raise ValueError('Native critic features have zero variation')
        slope = torch.linalg.solve(cov+ridge*torch.eye(len(center),dtype=z.dtype), z.T@(weights*(residual-bias)))
        model.net[-1].weight.add_(slope.float().reshape(1,-1))
        model.net[-1].bias.add_((bias-center@slope).float())
        predictions['landing_linear_readout'] = torch.cat([model(x).flatten() for x in inputs.split(128)])
        if readout_path is not None:
            from quadsmash.rl.supermimic_physical_run_guard_v1 import atomic_torch_save
            atomic_torch_save(dict(critic=model.state_dict(), source_critic_sha256=tensor_digest(checkpoint['critic'].items()),
                                   actor_sha256=tensor_digest(checkpoint['actor'].items()), ridge=float(ridge),
                                   training_only=True), readout_path)
    return predictions


def audit(run, refresh, before, after, max_distance=32, fit=False, landing_inputs=None, readout_path=None):
    directory = run / refresh
    bank = torch.load(directory / 'native_critic_labels.pt', map_location='cpu', weights_only=True)
    episodes = directory / 'complete_episodes.pt'
    with episodes.open('rb') as f:
        if hashlib.file_digest(f, 'sha256').hexdigest() != bank['source_sha256']:
            raise ValueError('RTG bank source archive changed')
    archive = torch.load(episodes, map_location='cpu', weights_only=True)
    data, receipt = archive['tensors'], archive['receipt']
    records = json.loads((directory / 'episodes_summary.json').read_text())
    held = set(json.loads((run / 'config.json').read_text())['wm_holdout_lanes'])
    identities = [tuple(map(int, item)) for item in bank['identity']]
    lookup = {key: i for i, key in enumerate(identities)}
    actor = bank['actor_sha256']
    if (len(lookup) != len(identities) or receipt['actor_sha256'] != actor
            or tensor_digest(bank['label_actor_state'].items()) != actor
            or archive['policy_id'] != bank['policy_id']):
        raise ValueError('Native snapshot identity mismatch')
    case_keys = {(r['env_id'], r['episode_index']) for r in records}
    if (len(case_keys) != len(records) or case_keys != set(enumerate(receipt['episode_ids']))
            or any(r['actor_sha256'] != actor or r['policy_id'] != bank['policy_id']
                   or r['policy_execution'] != bank['policy_execution'] for r in records)):
        raise ValueError('Terminal episode or execution contract mismatch')
    gamma, = {r['native_return_gamma'] for r in records}
    rtg, complete = complete_native_rtg(data, gamma)
    lanes, eps, ticks = torch.tensor(identities).T
    if not (complete[ticks, lanes].all() and torch.equal(data['episode'][ticks, lanes], eps)
            and torch.equal(ticks, bank['stage']) and torch.equal(rtg[ticks, lanes].float(), bank['rtg'])):
        raise ValueError('Archived return disagrees with complete native suffix')
    n_original = len(identities)
    if landing_inputs is not None:
        extra = torch.load(landing_inputs, map_location='cpu', weights_only=True)
        if (extra['actor_sha256'] != actor or extra['source_sha256'] != bank['source_sha256']
                or extra['reproduced_sparse_inputs'] <= 0):
            raise ValueError('Additional event-aligned critic inputs differ from the archive')
        new = []
        for i, key in enumerate(extra['identity']):
            key = tuple(key)
            lane, ep, t = key
            if key in lookup:
                continue
            if not complete[t,lane] or int(data['episode'][t,lane]) != ep:
                raise ValueError('Unknown event-aligned target')
            lookup[key] = len(identities)
            identities.append(key)
            new.append(i)
        added = torch.tensor(new, dtype=torch.long)
        lanes, eps, ticks = torch.tensor(identities).T
        bank['inputs'] = torch.cat((bank['inputs'], extra['inputs'][added]))
        bank['rtg'] = rtg[ticks,lanes].float()
        bank['head_masks']['recovery'] = torch.cat((bank['head_masks']['recovery'], extra['recovery'][added]))
    feet = data['feet'].bool().any(-1)
    landing = (~feet & data['next_feet'].bool().any(-1) & (data['event_state'][..., 0] > .5)
               & (data['event_state'][..., 1] > .5))
    values = {}
    for name, path in [('before_refresh', before), ('after_refresh', after)]:
        checkpoint = torch.load(path, mmap=True, map_location='cpu', weights_only=False)
        if tensor_digest(checkpoint['actor'].items()) != actor:
            raise ValueError('Critic checkpoint belongs to another actor')
        critic = MLP(bank['inputs'].shape[-1], (512, 256, 128), 1).eval()
        critic.load_state_dict(checkpoint['critic'])
        with torch.no_grad():
            values[name] = torch.cat([critic(x).flatten() for x in bank['inputs'].split(128)])
        del checkpoint
    stages = {'before_support': [], 'after_support': []}
    training_stages = {k: [] for k in stages}
    eligible = []
    for record in records:
        lane, episode = record['env_id'], record['episode_index']
        if not record['qualified_airborne_string_contact']:
            continue
        if lane in held:
            eligible.append(lane)
        crossings = torch.where(landing[:receipt['lengths'][lane], lane])[0]
        if not len(crossings):
            continue
        touchdown = int(crossings[0])
        valid = ticks[(lanes == lane) & bank['head_masks']['recovery'].bool()].tolist()
        if landing_inputs is not None:
            valid = [tick for tick in valid if tick in (touchdown, touchdown+1)]
        for stage, tick in select_landing_rows(valid, touchdown, max_distance).items():
            if tick is None:
                continue
            i = lookup[(lane, episode, tick)]
            if lane not in held:
                training_stages[stage].append(i)
                continue
            stages[stage].append(dict(lane=lane, episode=episode, case=record['case_id'], tick=tick,
                                      landing_transition=touchdown, recovered=bool(record['recovery_success']),
                                      native_rtg=float(bank['rtg'][i]), **{k:float(v[i]) for k,v in values.items()}))
    general = torch.where(~torch.isin(lanes[:n_original], torch.tensor(sorted(held))))[0]
    if fit:
        groups = [torch.tensor(ids, dtype=torch.long) for ids in training_stages.values()]
        if not all(len(g) for g in groups) or any(torch.isin(lanes[g], torch.tensor(sorted(held))).any() for g in groups):
            raise ValueError('Both landing phases need eligible training cases outside the holdout')
        fitted = offline_comparison(after, bank['inputs'], bank['rtg'], general, groups, readout_path=readout_path)
        values.update(fitted)
        for rows in stages.values():
            for row in rows:
                i = lookup[(row['lane'], row['episode'], row['tick'])]
                row.update({name:float(pred[i]) for name, pred in fitted.items()})
    scores = {}
    for stage, rows in stages.items():
        success = torch.tensor([r['recovered'] for r in rows], dtype=torch.bool)
        target = torch.tensor([r['native_rtg'] for r in rows])
        scores[stage] = dict(cases=len(rows), recovered=int(success.sum()), target_auc=ranking(target, success))
        for name in values:
            pred = torch.tensor([r[name] for r in rows])
            scores[stage][name] = dict(mae=float((pred-target).abs().mean()) if rows else None,
                                       recovery_auc=ranking(pred, success))
    original_held = torch.isin(lanes[:n_original], torch.tensor(sorted(held)))
    overall = {}
    for head in ('recovery','approach'):
        selected = torch.where(original_held & bank['head_masks'][head][:n_original].bool())[0]
        overall[head] = dict(rows=len(selected), mae={k:float((v[selected]-bank['rtg'][selected]).abs().mean())
                                                     for k,v in values.items()} if len(selected) else {})
    return dict(actor_sha256=actor, source_sha256=bank['source_sha256'], policy=bank['policy_id'],
                original_holdout=overall,
                scope='Frozen refresh training-run held lanes. One gated snapshot per episode per phase.',
                extra_aligned_inputs=len(identities)-n_original, offline_updates_per_arm=8 if fit else 0, training_rows={k:len(v) for k,v in training_stages.items()},
                max_distance_ticks=max_distance, eligible_contact_lanes=sorted(eligible), scores=scores,
                cases=stages)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('run', 'before', 'after', 'output'):
        parser.add_argument('--'+arg, type=Path, required=True)
    parser.add_argument('--refresh', required=True)
    parser.add_argument('--compare-offline-fit', action='store_true')
    parser.add_argument('--landing-inputs', type=Path)
    parser.add_argument('--save-readout', type=Path)
    args = parser.parse_args()
    if args.save_readout and not args.compare_offline_fit:
        parser.error('--save-readout requires --compare-offline-fit')
    report = audit(args.run, args.refresh, args.before, args.after, fit=args.compare_offline_fit, landing_inputs=args.landing_inputs, readout_path=args.save_readout)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report['scores'], indent=2))
