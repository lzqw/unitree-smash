"""Persistent Recovery tail value. All TD targets come from complete native PhysX episodes."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from quadsmash.rl.supermimic_v1 import load_ppo_initialized_actor
from scripts.native_recovery_ppo import explained_variance, gae


class NativeTailValue(nn.Module):
    def __init__(self, normalizer):
        super().__init__()
        self.normalizer = normalizer.eval()
        for parameter in self.normalizer.parameters():
            parameter.requires_grad_(False)
        self.value = nn.Sequential(nn.Linear(794, 256), nn.LayerNorm(256), nn.Tanh(),
                                   nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 1))
        nn.init.zeros_(self.value[-1].weight)
        nn.init.zeros_(self.value[-1].bias)

    def forward(self, obs):
        return self.value(self.normalizer(obs).clamp(-10, 10)).squeeze(-1)


def read_rollout(root, checkpoint):
    directory = root/'trajectories'
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest['schema'] != 'dynamimic-native-recovery-v1':
        raise ValueError('Tail value requires a completed native Recovery archive')
    if manifest['checkpoint_sha256'] != hashlib.sha256(checkpoint.read_bytes()).hexdigest():
        raise ValueError('Critic rollout checkpoint mismatch')
    keys = ('obs', 'active', 'reward', 'done', 'body', 'net', 'live')
    parts = {k:[] for k in keys}
    for name in manifest['chunks']:
        block = torch.load(directory/name, weights_only=True, mmap=True)
        for k in keys:
            parts[k].append(block[k])
    data = {k:torch.cat(v) for k,v in parts.items()}
    if not torch.all((data['done'] & data['live']).sum(0) == 1):
        raise ValueError('Expected exactly one native episode end per lane')
    return data, manifest


def make_tail(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    source = ckpt['metadata']['init_policy_checkpoint']
    path = Path(source['path'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != source['sha256']:
        raise ValueError('Source actor changed')
    actor, _ = load_ppo_initialized_actor(path, torch.device(device))
    actor.load_state_dict(ckpt['actor'], strict=True)
    return NativeTailValue(actor.actor.normalizer).to(device), source['sha256']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('native','checkpoint','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(67319)
    data, manifest = read_rollout(args.native, args.checkpoint)
    tail, source = make_tail(args.checkpoint, args.device)
    opt = torch.optim.AdamW(tail.value.parameters(), lr=3e-4, weight_decay=1e-4)
    if args.resume:
        saved = torch.load(args.resume, weights_only=True, map_location=args.device)
        if saved['source'] != source:
            raise ValueError('Frozen actor used for normalization differs')
        tail.value.load_state_dict(saved['critic'])
        opt.load_state_dict(saved['optimizer'])
    active = data['active']
    held = torch.arange(active.size(1)) % 16 < 2
    train = active & ~held
    valid = active & held
    if not train.any() or not valid.any():
        raise ValueError('No eligible transitions in train or held-out split')
    with torch.no_grad():
        values = torch.zeros_like(data['reward'])
        values[active] = torch.cat([tail(z.to(args.device)).cpu() for z in data['obs'][active].split(1024)])
        done = data['done'] | data['body'] | data['net']
        targets = values + gae(data['reward'], values, active, done, lam=.95 if args.resume else 1.)
    obs, truth = data['obs'][train], targets[train]
    for _ in range(3):
        for ids in torch.randperm(len(obs)).split(1024):
            pred = tail(obs[ids].to(args.device))
            loss = F.smooth_l1_loss(pred, truth[ids].to(args.device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(tail.value.parameters(), 1., error_if_nonfinite=True)
            opt.step()
    with torch.no_grad():
        predictions = torch.cat([tail(z.to(args.device)).cpu() for z in data['obs'][valid].split(1024)])
    report = dict(source=source, native_checkpoint=manifest['checkpoint_sha256'], training_frames=len(obs),
                  held_frames=len(predictions), held_explained_variance_before=explained_variance(values[valid], targets[valid]),
                  held_explained_variance_after=explained_variance(predictions, targets[valid]),
                  held_mae=float((predictions-targets[valid]).abs().mean()), returned_targets='native TD(lambda)')
    torch.save(dict(critic=tail.value.state_dict(), optimizer=opt.state_dict(), source=source), args.output/'critic.pt')
    (args.output/'fit.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
