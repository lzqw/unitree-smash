#!/usr/bin/env python3
"""Fit a phase-774 recovery actor from actual successful native traces.

Only episodes with an observed airborne contact and native conditional recovery
are used. The result is a portable residual actor for an opt-in recovery-phase
teacher; it is not claimed to be a closed-loop success result by itself.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quadsmash.rl.episode_trace_v4 import unpack_public
from quadsmash.rl.offline_actor_export_v1 import synchronize_offline_actor


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(794, 256), nn.ELU(),
                                 nn.Linear(256, 128), nn.ELU(),
                                 nn.Linear(128, 64), nn.ELU(),
                                 nn.Linear(64, 18))

    def forward(self, x):
        return self.net(x)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--traces', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=882104)
    args = parser.parse_args()
    if args.output.exists() or args.updates < 1:
        raise ValueError('Require a fresh output and positive updates')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state = checkpoint['model_state_dict']
    if checkpoint.get('native', {}).get('canonical_restart') is not True:
        raise ValueError('Require a canonical native checkpoint')
    mean = state['actor_obs_normalizer._mean'].float().clone()
    std = state['actor_obs_normalizer._std'].float().clamp_min(1e-6).clone()
    actor = Actor()
    actor.load_state_dict({'net.' + k[len('actor.'):]: v for k, v in state.items()
                           if k.startswith('actor.')}, strict=True)
    initial = Actor(); initial.load_state_dict(actor.state_dict()); initial.eval()

    manifest = []
    for meta_path in sorted(args.traces.glob('trajectory_traces/*.json')):
        meta = json.loads(meta_path.read_text())
        outcome = meta.get('original_outcomes', {})
        if not (outcome.get('airborne_contact') and outcome.get('conditional_recovery')):
            continue
        manifest.append(meta)
    if not manifest:
        raise ValueError('No native airborne-contact recovery-success traces')

    observations, targets = [], []
    selected_rows = 0
    for meta in manifest:
        data_path = args.traces / 'trajectory_traces' / meta['data_file']
        if sha(data_path) != meta['data_sha256']:
            raise ValueError('Trace data hash mismatch')
        with np.load(data_path, allow_pickle=False) as data:
            public = unpack_public(data['packed_public']).astype(np.float32)
            actions = data['residual_actions'].astype(np.float32)
        if public.shape[1:] != (794,) or actions.shape != (len(public), 18):
            raise ValueError('Trace interface mismatch')
        mask = public[:, 774] > 0.5
        if mask.any():
            observations.append(public[mask]); targets.append(actions[mask]); selected_rows += int(mask.sum())
    if not observations:
        raise ValueError('No phase-774 observations in recovery-success traces')
    x = torch.from_numpy(np.concatenate(observations))
    y = torch.from_numpy(np.concatenate(targets))
    z = (x - mean) / std
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
    batch = min(1024, len(z))
    history = []
    for update in range(1, args.updates + 1):
        ids = torch.randint(len(z), (batch,))
        pred = actor(z[ids]); target = y[ids]
        fit = (pred - target).square().mean()
        with torch.no_grad(): anchor = initial(z[ids])
        loss = fit + 0.05 * (pred - anchor).square().mean()
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0); optimizer.step()
        if update % 250 == 0 or update == args.updates:
            with torch.inference_mode():
                err = actor(z) - y
                row = dict(update=update, loss=float(loss), rmse=float(err.square().mean().sqrt()),
                           p95=float(err.abs().flatten().quantile(.95)), max=float(err.abs().max()))
                history.append(row); print(json.dumps(row), flush=True)

    actor.eval()
    probe = x[::max(1, len(x)//4096)]
    result = synchronize_offline_actor(checkpoint, actor, mean, std, probe)
    blob = io.BytesIO()
    scripted = torch.jit.script(nn.Sequential(
        nn.Identity(), actor))
    # Export the exact normalized wrapper used by the portable teacher.
    class Normalized(nn.Module):
        def __init__(self, m, s, a):
            super().__init__(); self.register_buffer('mean', m); self.register_buffer('std', s); self.actor = a
        def forward(self, obs): return self.actor((obs - self.mean) / self.std)
    torch.jit.save(torch.jit.script(Normalized(mean, std, actor)), blob)
    result['residual_actor_jit'] = blob.getvalue()
    result['residual_training_metadata'] = dict(
        result.get('residual_training_metadata', {}), phase774_recovery_distillation=True,
        recovery_trace_episodes=len(manifest), recovery_trace_phase774_rows=selected_rows,
        recovery_trace_updates=args.updates,
        recovery_trace_manifest_sha256=sha(args.traces / 'INPUTS.json')
        if (args.traces / 'INPUTS.json').exists() else None,
        native_execution=False, formal_acceptance=False)
    args.output.mkdir(parents=True)
    torch.save(result, args.output / 'policy.pt')
    report = dict(schema='phase774-recovery-actor-from-native-traces-v1',
                  source_checkpoint=str(args.checkpoint), source_checkpoint_sha256=sha(args.checkpoint),
                  trace_root=str(args.traces), episodes=len(manifest), phase774_rows=selected_rows,
                  updates=args.updates, history=history, native_execution=False,
                  closed_loop_success_tested=False, policy_sha256=sha(args.output / 'policy.pt'))
    (args.output / 'RESULT.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__': main()
