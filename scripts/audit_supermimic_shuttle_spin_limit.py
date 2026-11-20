#!/usr/bin/env python3
"""Measure the pre-solve angular speed limit from consecutive genuine shuttle states."""
import argparse
import json
from pathlib import Path
import torch
from evaluate_supermimic_contact_rollout import sha


def spin_samples(trace):
    t=trace['tensors']
    omega=t['shuttle_state_w'][...,10:13]
    active=torch.stack([t[k][1:].flatten(2).abs().amax(-1)>.001 for k in
        ('shuttle_total_normal_force_w','racket_friction_force_w','net_force_matrix_w','ground_force_w')]).any(0)
    before,after=omega[:-1],omega[1:]
    # Use observed force-free angular deceleration to estimate the limiting speed.
    drop=(before.norm(dim=-1)-after.norm(dim=-1)>1.) & ~active
    return before,after,active,drop


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wide',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    torch.set_num_threads(2)
    design=json.loads((a.wide/'design.json').read_text())
    hashes={}
    limit=None
    stats={}
    for split in ('train','development'):
        audit=json.loads((a.wide/(split+'.json')).read_text())
        if audit['design_sha256']!=sha(a.wide/'design.json') or split!=audit['split']:
            raise ValueError('Different launch family assignment')
        examples=[];contact=[];declines=[];free=0
        for seed,part in design['assignments'].items():
            if part!=split:continue
            # Nominal once per family. Repeated nominal does not add independent support.
            path=a.wide/'nominal_a'/f'spec_{seed}'/'trace.pt'
            hashes[str(path)]=sha(path)
            if hashes[str(path)]!=audit['archives'][str(path)]:raise ValueError('Changed nominal trajectory')
            trial=torch.load(path,mmap=True,weights_only=True)
            before,after,active,drop=spin_samples(trial['raw']['substep_trace'])
            declines.append(after[drop].norm(dim=-1))
            examples.append((before[~active],after[~active]))
            contact.append((before[active].norm(dim=-1),after[active].norm(dim=-1)))
            free+=int((~active).sum())
        drops=torch.cat(declines)
        if split=='train':
            if drops.numel()==0:raise ValueError('No observed angular deceleration in calibration')
            limit=float(drops.median())
        before,after=(torch.cat(x) for x in zip(*examples))
        pre,post=(torch.cat(x) for x in zip(*contact))
        capped=before*(limit/before.norm(dim=-1,keepdim=True).clamp_min(1.e-10)).clamp(max=1.)
        stats[split]=dict(force_free=free,force_free_speed_drop_rows=len(drops),drop_final_speed_range_rad_s=
            [float(drops.min()),float(drops.max())] if len(drops) else [],
            uncapped_free_component_rmse=float((after-before).square().mean().sqrt()),
            capped_free_component_rmse=float((after-capped).square().mean().sqrt()),
            active_substeps=len(post),contact_ends_above_limit=int((post>limit+1.).sum()),
            contact_starts_above_limit=int((pre>limit+1.).sum()),maximum_post_contact_speed_rad_s=float(post.max()))
    with a.output.open('x') as out:
        json.dump(dict(limit_rad_s=limit,source_sha256=hashes,groups=stats,policy_updates=0,
                       contact_gradients_admitted=False),out,indent=2,allow_nan=False)
        out.write('\n')


if __name__=='__main__':main()
