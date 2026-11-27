"""Keep successful native actions while retaining BPTT changes near observed failures."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from quadsmash.rl.supermimic_v1 import load_ppo_initialized_actor
from scripts.dynamimic_recovery_validation import policies
from scripts.fit_dynamimic_recovery_critic import root_termination
from scripts.fit_dynamimic_recovery_wm import read_data
from scripts.train_dynamimic_recovery_bptt import mixed_request, recovery_start_groups, stratified_starts


def ridge_correction(features, desired_change, success, phase_case, training):
    """Give successes and pre-fall cases equal mass, with equal weight per case-phase."""
    counts = phase_case[training].bincount(minlength=int(phase_case.max())+1)
    weight = features.new_zeros(len(features))
    for group in (success, ~success):
        selected = training & group
        if not selected.any():
            raise ValueError('Both successful and pre-fall native cases are required')
        inverse = counts[phase_case[selected]].to(features).reciprocal()
        weight[selected] = .5 * inverse / inverse.sum()
    weighted = features.T * weight
    gram = weighted @ features
    gram.diagonal().add_(.001)
    return torch.linalg.solve(gram, weighted @ (desired_change * success[:, None]))


def change_metrics(original, corrected, phase_case):
    counts = phase_case.bincount()
    weights = counts[phase_case].to(original).reciprocal()
    weights /= weights.sum()
    a = (weights * original.square().mean(1)).sum()
    b = (weights * corrected.square().mean(1)).sum()
    dot = (weights * (original * corrected).mean(1)).sum()
    return dict(before_rms=float(a.sqrt()), after_rms=float(b.sqrt()),
                directional_gain=float(dot/a.clamp_min(1e-12)),
                direction_cosine=float(dot/(a*b).clamp_min(1e-24).sqrt()))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('native', 'checkpoint', 'companion', 'bptt', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    data, manifest = read_data(args.native, ('active','done','net','relaxed','strict','obs','release_time','request'))
    behavior = digest(args.checkpoint)
    if (behavior != manifest['checkpoint_sha256'] or digest(args.companion) != manifest['companion_sha256']
            or json.loads((args.bptt/'training.json').read_text())['native'] != behavior):
        raise ValueError('Training archive or BPTT policy identity differs')
    checkpoint = torch.load(args.checkpoint, weights_only=False, mmap=True, map_location='cpu')
    source = Path(checkpoint['metadata']['init_policy_checkpoint']['path'])
    if digest(source) != checkpoint['metadata']['init_policy_checkpoint']['sha256']:
        raise ValueError('Frozen policy initialization changed')
    height = data['after'][...,2] + torch.tensor(manifest['origins'])[:,2]
    data['root_terminated'] = root_termination(data['done'], height)
    falls = data['body'] | data['root_terminated']
    success = ~((falls | data['net']) & data['live']).any(0) & (
        (data['relaxed'] | data['strict']) & data['live']).any(0)
    impending = F.pad(falls & data['live'], (0, 0, 0, 47)).unfold(0,48,1).any(-1)
    held = torch.arange(256)%16 < 2
    tick,lane = (data['active'] & ~held & (success | impending)).nonzero().T
    touchdown = torch.where(data['landed'] & data['live'], torch.arange(len(data['live']))[:,None],
                            len(data['live'])).amin(0)
    phase = torch.bucketize(tick-touchdown[lane], torch.tensor([0,48]), right=True)
    keys = 3*lane + phase
    ticks = [int(p.stem.split('_')[-1]) for p in (args.native/'trajectories').glob('g07_*.pt')]
    checks = stratified_starts(recovery_start_groups(data,ticks,held),128,torch.Generator().manual_seed(9917))
    args.output.mkdir(parents=True,exist_ok=False)
    report = dict(behavior=behavior, companion=digest(args.companion), successful_cases=int(success[~held].sum()),
                  training_cases=int(lane.unique().numel()), scope='native training-side action replay only', candidates={})
    for label,budget in [('conservative',.020),('moderate',.035)]:
        original_path = args.bptt/(label+'.pt')
        parent, other, candidate, _ = policies(args.checkpoint, original_path, args.companion,
                                               lambda: load_ppo_initialized_actor(source,'cpu')[0])
        head = list(candidate.actor.recovery.children())
        def requests(actor,t,l):
            return mixed_request(actor,other,data['obs'][t,l], l%2==1,data['release_time'][t,l])
        features, raw, target = [], [], []
        for ids in torch.arange(len(tick)).split(512):
            t,l = tick[ids],lane[ids]
            normalized = candidate.actor.normalizer(data['obs'][t,l])
            x = normalized
            for layer in head[:-1]:
                x = layer(x)
            features.append(F.pad(x,(0,1),value=1.))
            target.append(parent.actor.recovery(normalized) - candidate.actor.recovery(normalized))
            raw.append(requests(candidate,t,l))
            torch.testing.assert_close(requests(parent,t,l).clamp(-1,1),data['request'][t,l],rtol=0,atol=2e-6)
        x, y = torch.cat(features).double(),torch.cat(target).double()
        raw = torch.cat(raw)
        nominal = data['request'][tick,lane]
        ready = (data['obs'][tick,lane,775]-data['release_time'][tick,lane]) >= .04
        gate = torch.cat((torch.ones(len(tick),12),ready[:,None].expand(-1,6)),1)
        old = raw.clamp(-1,1)-nominal
        folds = []
        for fold in range(5):
            train = lane%5 != fold
            delta = ridge_correction(x,y,success[lane],keys,train)
            new = (raw + gate*(x@delta).float()*candidate.actor.recovery_scale).clamp(-1,1)-nominal
            for group,flag in [('success',success[lane]),('prefall',~success[lane])]:
                mask = ~train & flag
                if not mask.any():
                    continue
                folds.append(dict(fold=fold, group=group, cases=len(lane[mask].unique()),
                                  **change_metrics(old[mask],new[mask],keys[mask])))
        delta = ridge_correction(x,y,success[lane],keys,torch.ones_like(lane,dtype=torch.bool)).float()
        head[-1].weight.add_(delta[:-1].T)
        head[-1].bias.add_(delta[-1])
        expected = (raw + gate * (x @ delta.double()).float() * candidate.actor.recovery_scale).clamp(-1, 1)
        maximum_error = 0.
        for ids in torch.arange(len(tick)).split(512):
            actual = requests(candidate, tick[ids], lane[ids]).clamp(-1, 1)
            torch.testing.assert_close(actual, expected[ids], atol=2e-6, rtol=0)
            maximum_error = max(maximum_error, float((actual - expected[ids]).abs().max()))
        t,l = checks.T
        rms = float((requests(candidate,t,l).clamp(-1,1)-requests(parent,t,l).clamp(-1,1)).square().mean().sqrt())
        if not 0 < rms <= budget:
            raise ValueError(f'{label} action budget failed: {rms}')
        saved = torch.load(original_path,weights_only=False,mmap=True,map_location='cpu')
        state = candidate.state_dict()
        changed = {k for k,v in state.items() if not torch.equal(v,saved['actor'][k])}
        if changed != {'actor.recovery.4.weight','actor.recovery.4.bias'}:
            raise ValueError(f'Unexpected retention parameters: {changed}')
        dest = args.output/(label+'.pt')
        torch.save(dict(saved,actor=state),dest)
        report['candidates'][label] = dict(before=digest(original_path),after=digest(dest),rms=rms,budget=budget,
                                          replay_max_error=maximum_error, folds=folds)
    (args.output/'fit.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
