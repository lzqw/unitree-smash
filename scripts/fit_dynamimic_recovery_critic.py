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


# Native JumpEnv terminates at this world-frame root height independently of contact.
ROOT_FAILURE_HEIGHT = .12
FALL_COST = 15.
NATIVE_RETURN_SCHEMA = 'recovery-native-low-root-terminal-v1'


def root_termination(done, root_height_w):
    return done & (root_height_w < ROOT_FAILURE_HEIGHT)


class NativeTailValue(nn.Module):
    def __init__(self, normalizer):
        super().__init__()
        self.normalizer = normalizer.eval()
        for parameter in self.normalizer.parameters():
            parameter.requires_grad_(False)
        self.value = nn.Sequential(nn.Linear(794, 256), nn.LayerNorm(256), nn.Tanh(),
                                   nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 1))
        self.register_buffer('return_scale', torch.ones(()))
        self.register_buffer('return_mean', torch.zeros(()))
        nn.init.zeros_(self.value[-1].weight)
        nn.init.zeros_(self.value[-1].bias)

    def forward(self, obs):
        z = self.normalizer(obs).clamp(-10, 10)
        return self.return_mean + self.return_scale * self.value(z).squeeze(-1)

    def load_native_fit(self, saved):
        self.value.load_state_dict(saved['critic'])
        self.return_scale.fill_(saved.get('return_scale', 1.))
        self.return_mean.fill_(saved.get('return_mean', 0.))
        if not torch.isfinite(self.return_mean) or not self.return_scale.isfinite() or self.return_scale <= 0:
            raise ValueError('Invalid native return scaling')


def read_rollout(root, checkpoint):
    directory = root/'trajectories'
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest['schema'] != 'dynamimic-native-recovery-v1':
        raise ValueError('Tail value requires a completed native Recovery archive')
    if manifest['checkpoint_sha256'] != hashlib.sha256(checkpoint.read_bytes()).hexdigest():
        raise ValueError('Critic rollout checkpoint mismatch')
    keys = ('obs', 'active', 'reward', 'done', 'body', 'net', 'live')
    parts = {k:[] for k in keys}
    root_events = []
    for name in manifest['chunks']:
        block = torch.load(directory/name, weights_only=True, mmap=True)
        root_events.append(root_termination(block['done'], block['after'][..., 2]))
        for k in keys:
            parts[k].append(block[k])
    data = {k:torch.cat(v) for k,v in parts.items()}
    # Raw native rewards already charge the first body or net event. A low-root
    # terminal may occur before either sensor, while the lane is still eligible.
    root_only = torch.cat(root_events) & data['active'] & ~data['body'] & ~data['net']
    data['reward'] = torch.where(root_only, -FALL_COST, data['reward'])
    data['root_only_terminal'] = root_only
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
    parser.add_argument('--history-native', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(67319)
    data, manifest = read_rollout(args.native, args.checkpoint)
    current_lanes = data['active'].shape[1]
    if args.history_native:
        from scripts.fit_dynamimic_recovery_wm import verify_replay_cases
        historical, previous = read_rollout(args.history_native, args.checkpoint)
        if previous['companion_sha256'] != manifest['companion_sha256']:
            raise ValueError('Historical Critic targets require the same companion policy')
        verify_replay_cases([manifest['cases'], previous['cases']])
        current_ids = {case['case_id'] for case in manifest['cases']}
        if any(case['case_id'] in current_ids for case in previous['cases']):
            raise ValueError('Historical Critic archive repeats current cases')
        # Keep episodes in separate lanes when computing GAE.
        duration = max(len(historical['active']), len(data['active']))
        for key in data:
            columns = []
            for tensor in (data[key], historical[key]):
                pad_shape = (duration - len(tensor), *tensor.shape[1:])
                columns.append(torch.cat((tensor, tensor.new_zeros(pad_shape))))
            data[key] = torch.cat(columns, dim=1)
        del historical
    tail, source = make_tail(args.checkpoint, args.device)
    opt = torch.optim.AdamW(tail.value.parameters(), lr=3e-4, weight_decay=1e-4)
    matching_return = False
    if args.resume:
        saved = torch.load(args.resume, weights_only=True, map_location=args.device)
        if saved['source'] != source:
            raise ValueError('Frozen actor used for normalization differs')
        matching_return = saved.get('return_schema') == NATIVE_RETURN_SCHEMA
        tail.load_native_fit(saved)
        opt.load_state_dict(saved['optimizer'])
    active = data['active']
    held = torch.arange(active.size(1)) % 16 < 2
    train = active & ~held
    valid = active & held & (torch.arange(active.size(1)) < current_lanes)
    if not train.any() or not valid.any():
        raise ValueError('No eligible transitions in train or held-out split')
    with torch.no_grad():
        values = torch.zeros_like(data['reward'])
        values[active] = torch.cat([tail(z.to(args.device)).cpu() for z in data['obs'][active].split(1024)])
        done = data['done'] | data['body'] | data['net']
        targets = values + gae(data['reward'], values, active, done, lam=.95 if matching_return else 1.)
    obs, truth = data['obs'][train], targets[train]
    if not args.resume:
        # Normalize the regression target once. All GAE, reports and Actor tail
        # values stay in native return units; later blocks retain the same scale.
        tail.return_mean.copy_(truth.mean())
        tail.return_scale.copy_(truth.std().clamp_min(1.))
    for _ in range(3):
        for ids in torch.randperm(len(obs)).split(1024):
            pred = tail(obs[ids].to(args.device))
            loss = F.smooth_l1_loss((pred-truth[ids].to(args.device))/tail.return_scale, torch.zeros_like(pred))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(tail.value.parameters(), 1., error_if_nonfinite=True)
            opt.step()
    with torch.no_grad():
        predictions = torch.cat([tail(z.to(args.device)).cpu() for z in data['obs'][valid].split(1024)])
    report = dict(return_schema=NATIVE_RETURN_SCHEMA, source=source, native_checkpoint=manifest['checkpoint_sha256'], training_frames=len(obs),
                  held_frames=len(predictions),
                  root_only_terminal_train=int(data['root_only_terminal'][:,~held].sum()),
                  root_only_terminal_held=int((data['root_only_terminal'] & valid).sum()), held_explained_variance_before=explained_variance(values[valid], targets[valid]),
                  held_explained_variance_after=explained_variance(predictions, targets[valid]),
                  held_mae=float((predictions-targets[valid]).abs().mean()), returned_targets='native TD(lambda)')
    if args.history_native:
        report['history_native'] = str(args.history_native)
    torch.save(dict(return_schema=NATIVE_RETURN_SCHEMA, critic=tail.value.state_dict(), optimizer=opt.state_dict(), source=source,
                    return_mean=float(tail.return_mean), return_scale=float(tail.return_scale)), args.output/'critic.pt')
    (args.output/'fit.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
