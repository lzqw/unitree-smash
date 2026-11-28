"""Compare two Recovery changes on the same native observations with equal weight per case."""
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


def action_comparison(old_delta, new_delta, lanes):
    ids = lanes.unique()
    weights = 1 / (len(ids) * lanes.bincount()[lanes].double())
    old, new = old_delta.double(), new_delta.double()
    def energy(x):
        return (weights * x.square().mean(1)).sum()
    e_old, e_new = energy(old), energy(new)
    dot = (weights * (old * new).mean(1)).sum()
    mean = weights @ new
    return dict(cases=len(ids), ticks=len(lanes), old_rms=float(e_old.sqrt()),
                new_rms=float(e_new.sqrt()), change_rms=float(energy(old-new).sqrt()),
                direction_cosine=float(dot/(e_old*e_new).sqrt()) if e_old*e_new>0 else None,
                constant_vector_energy_share=float(mean.square().mean()/e_new) if e_new>0 else None,
                leg_rms=float(energy(new[:,:12]).sqrt()), arm_rms=float(energy(new[:,12:]).sqrt()))


@torch.no_grad()
def compare(args):
    d, m = read_data(args.native, ('active','done','net','strict','obs','release_time','request'))
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    if digest(args.baseline) != m['checkpoint_sha256'] or digest(args.companion) != m['companion_sha256']:
        raise ValueError('Native archive and supplied policies differ')
    init = torch.load(args.baseline, mmap=True, weights_only=False, map_location='cpu')['metadata']['init_policy_checkpoint']
    if digest(Path(init['path'])) != init['sha256']:
        raise ValueError('Frozen initialization changed')
    def load():
        return load_ppo_initialized_actor(Path(init['path']), 'cpu')[0]
    incumbent = policies(args.baseline, args.previous, args.companion, load)
    new = policies(args.baseline, args.candidate, args.companion, load)[2:]
    fail = (d['body'] | root_termination(d['done'], d['after'][...,2]+torch.tensor(m['origins'])[:,2])) & d['live']
    impending = F.pad(fail & d['active'], (0,0,0,47)).unfold(0,48,1).any(-1)
    time = torch.arange(len(d['live']))[:,None]
    landing = torch.where(d['landed'] & d['live'], time+1, len(time)+1).amin(0)
    late = time-landing>=100
    clean = ~((fail | d['net']) & d['live']).any(0) & (d['strict'] & d['live']).any(0)
    out = dict(previous=digest(args.previous), candidate=digest(args.candidate), baseline=m['checkpoint_sha256'], groups={})
    for name, mask in [('early_prefall',impending & ~late), ('late_prefall',impending & late),
                       ('clean_strict',clean.expand_as(fail))]:
        for split in ('training','held'):
            for parity, app in enumerate(('primary','h8d')):
                lane = torch.arange(len(clean))
                select = (lane%16<2) == (split=='held')
                t,l = (d['active'] & mask & select & (lane%2==parity)).nonzero().T
                if not len(t):
                    continue
                changes = []
                for idx in torch.arange(len(t)).split(512):
                    x,y = t[idx],l[idx]
                    def request(policy):
                        return policy._gated_distribution(d['obs'][x,y],d['gate'][x,y],
                            release_time=d['release_time'][x,y]).loc.clamp(-1,1)
                    baseline = request(incumbent[parity])
                    torch.testing.assert_close(baseline,d['request'][x,y],rtol=0,atol=2e-6)
                    changes.append(torch.stack([request(incumbent[parity+2])-baseline,request(new[parity])-baseline]))
                old_delta,new_delta = torch.cat(changes,dim=1)
                out['groups'][f'{name}_{split}_{app}'] = action_comparison(old_delta,new_delta,l)
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('native','baseline','previous','candidate','companion','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    args.output.write_text(json.dumps(compare(args),indent=2)+'\n')
