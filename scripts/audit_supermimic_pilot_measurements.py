#!/usr/bin/env python3
"""Audit noise, interval length and impulse against sparse native contact data."""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch


def distribution(values):
    value = torch.as_tensor(values, dtype=torch.float64)
    value = value[torch.isfinite(value)]
    if not len(value):
        return {'count': 0}
    return dict(count=len(value), minimum=float(value.min()), p10=float(value.quantile(.1)),
                median=float(value.median()), p90=float(value.quantile(.9)), maximum=float(value.max()))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--pilot', required=True, type=Path)
    cli.add_argument('--output', required=True, type=Path)
    args = cli.parse_args()
    pilot = torch.load(args.pilot, weights_only=True, map_location='cpu')
    contacts = [r for r in pilot['rows'] if r['contact_object'] == 'racket']
    negatives = [r for r in pilot['rows'] if r['contact_object'] == 'no_contact']
    if len({(r['source_sha256'], r['lane']) for r in negatives}) != len(negatives):
        raise ValueError('Repeated negative capture/lane')
    sources = {r['source']: r['source_sha256'] for r in pilot['rows']}
    if len(sources) != pilot['source_count']:
        raise ValueError('Source missing from selected pilot')
    sensitivity = {v: Counter() for v in (.0001, .001, .005, .01)}
    no_contact = []
    momentum_error = []
    local_gaps = []
    for path, digest in sources.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Invalid source hash for {path}')
        trace = torch.load(path, weights_only=True, map_location='cpu')['raw']['substep_trace']
        t = trace['tensors']
        dt = trace['physics_dt_s']
        f = t['racket_force_w']
        norm = f.norm(dim=-1)
        absent = torch.ones_like(norm, dtype=torch.bool)
        for i, detail in enumerate(trace['manifold_frames']):
            absent[i, detail['lane'][detail['filter_index'] == 0]] = False
        no_contact.append(norm[absent])
        v = t['shuttle_state_w'][..., 7:10]
        free = v[:-1] + dt * (v.new_tensor([0.,0.,-9.81]) - .025442307692307694 * v[:-1].norm(dim=-1,keepdim=True) * v[:-1])
        dv_impulse = .0052 * (v[1:] - free)
        sensor_impulse = dt * (f + t['racket_friction_force_w'])
        for threshold, counts in sensitivity.items():
            mask = norm > threshold
            start = mask.clone()
            start[1:] &= ~mask[:-1]
            extended = torch.cat([mask.new_zeros(1, mask.shape[1]), mask, mask.new_zeros(1, mask.shape[1])])
            begins = (extended[1:] & ~extended[:-1]).T.nonzero()
            ends = (~extended[1:] & extended[:-1]).T.nonzero()
            lengths = ends[:,1] - begins[:,1]
            counts['intervals'] += int(start.sum())
            counts['active_substeps'] += int(mask.sum())
            counts['longest_interval'] = max(counts['longest_interval'], int(lengths.max()) if len(lengths) else 0)
            counts['false_active_without_manifold'] += int((mask & absent).sum())
            counts['force_free_substeps'] += int((~mask).sum())
            counts['sum_active_impulse_magnitudes_ns'] += float(sensor_impulse.norm(dim=-1)[mask].sum())
        isolated = (norm[1:] > .001) & (t['ground_force_w'][1:].norm(dim=-1) < .001)
        isolated &= t['net_force_matrix_w'][1:].flatten(2,-2).norm(dim=-1).amax(-1) < .001
        errors = (dv_impulse - sensor_impulse[1:]).norm(dim=-1) / dv_impulse.norm(dim=-1).clamp_min(1e-8)
        momentum_error.append(errors[isolated])
    noise = torch.cat(no_contact)
    mass_by = defaultdict(list)
    for row in contacts:
        mass = row['effective_mass_kg']
        ratio_residual = mass - .0052
        mass_by['all'].append(mass)
        mass_by['residual_to_shuttle_mass_kg'].append(ratio_residual)
        mass_by['region/' + row['contact_region']].append(mass)
        part = row['racket_part'] or 'unknown'
        mass_by['part/' + ('frame' if part.startswith('Frame_') else part)].append(mass)
        for bucket in row['bucket'].split(':'):
            if bucket.startswith(('vn','vt')):
                mass_by[bucket].append(mass)
        if row['previous_peak_gap_m'] is not None:
            local_gaps.append(row['previous_peak_gap_m'])
    report = dict(pilot_sha256=hashlib.sha256(args.pilot.read_bytes()).hexdigest(), archives=len(sources),
                  unique_negative_trials=len(negatives), matched_manifold_events=sum(bool(r['manifold_points']) for r in contacts),
                  threshold_sensitivity=sensitivity, noise_on_native_manifold_absent_steps={
                      'count':len(noise), **{f'p{q}':float(noise.quantile(q/100)) for q in (50,95,99)}},
                  threshold_n=.001, threshold_basis='Frozen task force threshold tested against independent absence of PhysX manifold witnesses.',
                  effective_mass_definition='Derived from native impulse and pre-impact speed at fixed restitution 0.72.',
                  inferred_effective_mass_kg={key:distribution(value) for key,value in mass_by.items()},
                  repeated_contact_peak_separation_m=distribution(local_gaps),
                  isolated_racket_momentum_relative_error=distribution(torch.cat(momentum_error)))
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
