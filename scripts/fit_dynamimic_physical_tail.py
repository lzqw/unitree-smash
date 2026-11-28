"""Fit a training-only Recovery value on physical states from complete native episodes."""
import argparse
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from quadsmash.rl.dynamimic_recovery_wm_v1 import local_robot_state, rotation
from scripts.fit_dynamimic_recovery_critic import NATIVE_RETURN_SCHEMA, gae, read_rollout
from scripts.fit_dynamimic_recovery_wm import read_data, verify_replay_cases
from scripts.run_supermimic_recovery_baseline import dump


def physical_features(q, v, feet, elapsed, landed):
    # Local root height and court-relative x are available in both recorded and imagined states.
    orientation = rotation(q[:, 3:7])[:, [0, 2]].flatten(1)
    return torch.cat((q[:, :1], (q[:, 2:3]-.2)/.15, orientation,
                      torch.asinh(v[:, :3]), torch.asinh(v[:, 3:6]/2),
                      q[:, 7:], torch.asinh(v[:, 6:]/5), feet.to(q),
                      elapsed[:, None]/8, landed[:, None].to(q)), 1)


class PhysicalTail(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(56, 96), nn.SiLU(), nn.Linear(96, 64),
                                    nn.SiLU(), nn.Linear(64, 1))
        self.register_buffer('center', torch.zeros(()))
        self.register_buffer('scale', torch.ones(()))

    def forward(self, x):
        return self.center + self.scale * self.layers(x).squeeze(-1)


def samples(directory, checkpoint):
    data, manifest = read_rollout(directory, checkpoint)
    physical, _ = read_data(directory)
    active = data['active']
    t, l = active.nonzero().T
    truth = gae(data['reward'], torch.zeros_like(data['reward']), active,
                data['done'] | data['body'] | data['net'], lam=1.)[t,l]
    q,v = local_robot_state(physical['before'][t,l])
    # Feet in the stored boundary are post-step. The observation has pre-step feet.
    z = data['obs'][t,l]
    landed = F.pad(physical['landed'][:-1], (0,0,1,0))[t,l]
    x = physical_features(q,v,z[:,57:61],z[:,775],landed)
    first_fault = active & (data['body'] | data['root_only_terminal'] | data['net'])
    near_fault = F.pad(first_fault,(0,0,0,47)).unfold(0,48,1).any(-1)[t,l]
    return manifest, (x, truth, l, near_fault)


def errors(model, data, mask):
    x, truth, _, near = data
    with torch.no_grad():
        diff = model(x[mask]) - truth[mask]
    return dict(frames=len(diff), mae=float(diff.abs().mean()),
                prefall_frames=int(near[mask].sum()), prefall_bias=float(diff[near[mask]].mean()),
                prefall_mae=float(diff[near[mask]].abs().mean()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('native','history-native','checkpoint','output'):
        p.add_argument('--'+key, type=Path, required=True)
    args = p.parse_args()
    torch.manual_seed(67319)
    m, current = samples(args.native,args.checkpoint)
    old, previous = samples(args.history_native,args.checkpoint)
    verify_replay_cases([m['cases'],old['cases']])
    if (m['companion_sha256'] != old['companion_sha256']
            or {c['case_id'] for c in m['cases']} & {c['case_id'] for c in old['cases']}):
        raise ValueError('Expected disjoint native cases and matching behaviors')
    x,y,l,_ = (torch.cat(values) for values in zip(current,previous,strict=True))
    train=l%16>=4
    x,y=x[train],y[train]
    model=PhysicalTail()
    model.center.copy_(y.mean())
    model.scale.copy_(y.std().clamp_min(1.))
    opt=torch.optim.AdamW(model.parameters(),lr=.001)
    dev = (current[2] % 16 >= 2) & (current[2] % 16 < 4)
    best_error=float('inf')
    best=None
    best_epoch=0
    args.output.mkdir(parents=True,exist_ok=False)
    for epoch in range(12):
        for i in torch.randperm(len(y)).split(768):
            loss=F.smooth_l1_loss((model(x[i])-y[i])/model.scale,torch.zeros_like(y[i]))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        stats=errors(model,current,dev)
        # Equal weight for overall fidelity and the 48 ticks before a native fault.
        score=.5*(stats['mae']+stats['prefall_mae'])
        if score<best_error:
            best_error=score
            best={k:v.detach().clone() for k,v in model.state_dict().items()}
            best_epoch=epoch+1
    model.load_state_dict(best)
    torch.save(dict(model=best,return_schema=NATIVE_RETURN_SCHEMA,
                    native_checkpoint=m['checkpoint_sha256']),args.output/'critic.pt')
    dump(args.output/'fit.json',dict(native_checkpoint=m['checkpoint_sha256'], epochs=12,
        selected_epoch=best_epoch,training_frames=len(y),development=errors(model,current,dev),
        held=errors(model,current,current[2]%16<2),return_schema=NATIVE_RETURN_SCHEMA))


if __name__=='__main__':
    main()
