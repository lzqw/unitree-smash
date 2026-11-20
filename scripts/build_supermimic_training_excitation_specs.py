#!/usr/bin/env python3
"""Predeclare train-only paired command windows for the shared world model."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from quadsmash.rl.supermimic_native_pair_capture_v1 import validate_spec


def training_lanes(exclusions):
    held = set().union(*exclusions)
    if not held or any(type(i) is not int or i not in range(128) for i in held):
        raise ValueError('Invalid reserved lanes')
    lanes = sorted(set(range(128)) - held)
    if not lanes:
        raise ValueError('No training lanes remain')
    return lanes


def make_spec(lanes, event_tick, lead, hold, epsilon, seed):
    generator = torch.Generator().manual_seed(seed)
    arm = torch.nn.functional.normalize(torch.randn(len(lanes), 6, generator=generator), dim=-1)
    direction = torch.cat((torch.zeros(len(lanes), 12), arm), dim=-1)
    start = event_tick - lead
    return validate_spec(dict(
        schema='native-paired-prefix-batch-spec-v2', tick=start+15, action_tick=start,
        action_hold_ticks=hold, head='approach', lanes=lanes, epsilon_request=epsilon,
        request_directions=direction.tolist(), direction_seed=seed,
        capture_wm_window=True, capture_substeps=False, data_role='training_action_identification',
        reference_event_tick=event_tick, reference_event_lead_ticks=lead))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-verification', type=Path, required=True)
    p.add_argument('--probe-holdout-manifest', type=Path, required=True)
    p.add_argument('--packed-data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--direction-seed', type=int, default=20261121)
    a = p.parse_args()
    reference = json.loads(a.reference_verification.read_bytes())
    holdout = json.loads(a.probe_holdout_manifest.read_bytes())
    lanes = training_lanes([reference['excluded_lanes'], holdout['validation_lanes']])
    package = torch.load(a.packed_data, weights_only=True, map_location='cpu', mmap=True)
    data = package['tensors']
    occurs = data['next_event_state'][:,lanes,1] > data['event_state'][:,lanes,1]
    count = occurs.sum(dim=1)
    ticks = sorted(torch.where(count > 0)[0].tolist(), key=lambda t:(-int(count[t]), t))
    schedule = [(4,4,.01), (2,2,.005), (8,8,.02), (1,2,.002)]
    if len(ticks) < len(schedule):
        raise ValueError('Insufficient distinct observed contact ticks on training lanes')
    specs = [make_spec(lanes,t,l,h,e,a.direction_seed+i)
             for i,(t,(l,h,e)) in enumerate(zip(ticks,schedule))]
    a.output.mkdir(parents=True, exist_ok=False)
    for i,spec in enumerate(specs):
        (a.output/f"spec_{i:02d}_tick{spec['tick']:03d}.json").write_text(json.dumps(spec,indent=2)+'\n')
    def digest(path):
        with path.open('rb') as f:
            return hashlib.file_digest(f,'sha256').hexdigest()
    manifest = dict(purpose='training action identification; not a native contact validation panel',
                    training_lanes=lanes, withheld_lanes=sorted(set(range(128))-set(lanes)),
                    packed_sha256=digest(a.packed_data), reference_sha256=digest(a.reference_verification),
                    probe_holdout_sha256=digest(a.probe_holdout_manifest), windows=len(specs),
                    window_steps=16, seed=a.direction_seed)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':
    main()
