#!/usr/bin/env python3
"""Measure how much native pair friction is explained by incoming contact slip."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def observations(archive, mass, radius):
    trace = archive['raw']['substep_trace']
    t = trace['tensors']
    dt = trace['physics_dt_s']
    normal_force = t['racket_force_w'][1:]
    friction = t['racket_friction_force_w'][1:]
    threshold = .001
    loaded = normal_force.norm(dim=-1) > threshold
    sliding = friction.norm(dim=-1) > threshold
    isolated = t['ground_force_w'][1:].norm(dim=-1) <= threshold
    isolated &= t['net_force_matrix_w'][1:].flatten(2,-2).norm(dim=-1).amax(-1) <= threshold
    step, lane = torch.where(isolated & (loaded | sliding))
    force = normal_force[step,lane]
    normal = F.normalize(force, dim=-1)
    load = force.norm(dim=-1) * dt / mass
    incoming = t['shuttle_state_w'][step,lane]
    point = t['racket_contact_position_w'][step+1,lane]
    point = torch.where(torch.isfinite(point).all(-1,keepdim=True), point, incoming[:,:3]-radius*normal)
    racket_arm = point - t['racket_position_w'][step,lane]
    racket_velocity = t['racket_linear_velocity_w'][step,lane] + torch.cross(
        t['racket_angular_velocity_w'][step,lane], racket_arm, dim=-1)
    relative = incoming[:,7:10] + torch.cross(incoming[:,10:13], -radius*normal, dim=-1) - racket_velocity
    slip = relative - (relative*normal).sum(-1,keepdim=True)*normal
    return dict(load=load, slip=slip, native=friction[step,lane]*(dt/mass),
                loaded=loaded[step,lane], active_friction=sliding[step,lane])


def evaluate(data, mu, gain):
    slip = data['slip']
    pred = -F.normalize(slip, dim=-1)*torch.minimum(gain*slip.norm(dim=-1),mu*data['load'])[:,None]
    native = data['native']
    mask = data['active_friction'] & data['loaded'] & (pred.norm(dim=-1)>1e-9)
    cosine = F.cosine_similarity(pred[mask],native[mask])
    ratio = native[mask].norm(dim=-1)/data['load'][mask]
    return dict(substeps=len(slip), loaded_with_direction=int(mask.sum()),
                friction_without_normal=int((data['active_friction'] & ~data['loaded']).sum()),
                vector_rmse_mps=float((native-pred).square().sum(-1).mean().sqrt()),
                zero_rmse_mps=float(native.square().sum(-1).mean().sqrt()),
                slip_direction_cosine_median=float(cosine.median()) if len(cosine) else None,
                observed_friction_to_normal_median=float(ratio.median()) if len(ratio) else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    physics = json.loads((Path(__file__).resolve().parents[1]/'configs/shuttle_physics_v2.json').read_text())
    pilot = torch.load(args.pilot,weights_only=True,map_location='cpu')
    sources = {}
    for row in pilot['rows']:
        identity = row['source_sha256'], row['split']
        if sources.setdefault(row['source'],identity) != identity:
            raise ValueError('Conflicting source identity or split')
    groups = {s:[] for s in ('train','validation','test')}
    if not pilot['split']['unseen_spec_holdout']:
        raise ValueError('Expected source-held-out pilot')
    for path, (digest, split) in sources.items():
        if sha(path) != digest:
            raise ValueError(f'Corrupt source {path}')
        obs = observations(torch.load(path,map_location='cpu',weights_only=True,mmap=True),
                           physics['mass_kg'],physics['collision_radius_m'])
        groups[split].append(obs)
    joined = {s:{k:torch.cat([record[k] for record in records]) for k in records[0]}
              for s,records in groups.items()}
    grid = [(mu, gain) for mu in (.04,.08,.16,.25,.4,.6,.8,1.)
            for gain in (.1,2/7,.5,1.)]
    mu, gain = min(grid,key=lambda pair:evaluate(joined['train'],*pair)['vector_rmse_mps'])
    out = dict(pilot_sha256=sha(args.pilot), archives=len(sources), fit_split='train',
               scope='Isolated native contact substeps in the contributing archives. Measured normal load and contact point are used. Friction-only substeps remain in the absolute error.',
               fitted=dict(mu=mu,gain=gain), splits={k:dict(baseline=evaluate(d,.16,2/7),
               fitted=evaluate(d,mu,gain)) for k,d in joined.items()}, contact_gradient_admitted=False)
    with args.output.open('x') as f:
        json.dump(out,f,indent=2,allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    main()
