#!/usr/bin/env python3
"""Build read-only native probe specs from recorded event edges.

This script only selects control ticks and writes diagnostic specs.  It never
creates counterfactual states, runs Isaac, or labels a policy successful.
"""
import argparse
import json
from pathlib import Path

import torch


def make_event_aligned_spec(row, head, epsilon, capture_substeps=False, action_lead_ticks=0):
    """Represent exactly the lanes whose recorded edge selected this tick."""
    if action_lead_ticks < 0 or action_lead_ticks >= row['tick']:
        raise ValueError('Action lead must leave a positive action tick')
    lanes = list(row['lanes'])
    directions = []
    for lane in lanes:
        direction = [0.0] * 18
        direction[12 + (lane % 6)] = 1.0
        directions.append(direction)
    return dict(
        schema='native-paired-prefix-batch-spec-v2',
        tick=row['tick'],
        head=head,
        lanes=lanes,
        epsilon_request=epsilon,
        request_directions=directions,
        capture_substeps=bool(capture_substeps),
        action_tick=int(row['tick']) - int(action_lead_ticks),
        event_alignment=dict(
            contact_validation_lanes=list(row['contact']),
            frame_validation_lanes=list(row['frame']),
            source='recorded event edge only',
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packed-data', type=Path, required=True)
    parser.add_argument('--split-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-specs', type=int, default=5)
    parser.add_argument('--epsilon', type=float, default=0.001)
    parser.add_argument('--head', choices=('approach', 'recovery'), default='approach')
    parser.add_argument('--capture-substeps', action='store_true',
                        help='Record a bounded native substep trace for each probe.')
    parser.add_argument('--action-lead-ticks', type=int, default=0,
                        help='Apply the perturbation before the recorded event tick.')
    args = parser.parse_args()
    if args.max_specs < 1 or args.epsilon <= 0 or args.action_lead_ticks < 0:
        raise ValueError('max-specs and epsilon must be positive, with nonnegative action lead')
    if args.output.exists():
        raise FileExistsError(args.output)

    package = torch.load(args.packed_data, map_location='cpu', weights_only=True)
    tensors = package['tensors']
    if tensors['event_state'].ndim != 3 or tensors['event_state'].shape[2] <= 6:
        raise ValueError('Packed archive lacks the event-state fields needed for alignment')
    split = json.loads(args.split_manifest.read_text())
    validation = sorted(set(int(x) for x in split['validation_lanes']))
    if not validation or min(validation) < 0 or max(validation) >= 128:
        raise ValueError('Validation split is not an original-128 lane split')

    event_edges = {
        'contact': tensors['next_event_state'][:, :, 1] > tensors['event_state'][:, :, 1],
        'frame': tensors['next_event_state'][:, :, 6] > tensors['event_state'][:, :, 6],
    }
    candidates = []
    for tick in range(1, tensors['event_state'].shape[0] - 1):
        contact_ids = [lane for lane in validation if bool(event_edges['contact'][tick, lane])]
        frame_ids = [lane for lane in validation if bool(event_edges['frame'][tick, lane])]
        ids = sorted(set(contact_ids + frame_ids))
        if ids:
            candidates.append(dict(tick=tick, lanes=ids, contact=contact_ids, frame=frame_ids))

    # Greedily cover new held-out event lanes, then prefer contact over frame
    # when coverage is tied.  This is only a deterministic spec-selection rule.
    chosen = []
    covered = set()
    remaining = list(candidates)
    while remaining and len(chosen) < args.max_specs:
        remaining.sort(key=lambda row: (
            len(set(row['lanes']) - covered),
            len(row['contact']),
            len(row['frame']),
            -row['tick']), reverse=True)
        row = remaining.pop(0)
        chosen.append(row)
        covered.update(row['lanes'])

    # Keep the intervention lane set equal to the recorded event-aligned
    # candidates.  Expanding every spec to all 128 lanes dilutes a contact
    # probe with unrelated no-contact lanes and makes the native direction
    # gate measure the wrong population.
    args.output.mkdir(parents=True)
    for index, row in enumerate(chosen):
        spec = make_event_aligned_spec(row, args.head, args.epsilon,
                                       args.capture_substeps, args.action_lead_ticks)
        (args.output / f'spec_{index:02d}_tick{row["tick"]:03d}.json').write_text(
            json.dumps(spec, indent=2) + '\n'
        )
    manifest = dict(
        schema='event-aligned-native-probe-spec-set-v1',
        packed_data_sha256=__import__('hashlib').sha256(args.packed_data.read_bytes()).hexdigest(),
        split_manifest_sha256=__import__('hashlib').sha256(args.split_manifest.read_bytes()).hexdigest(),
        head=args.head,
        epsilon_request=args.epsilon,
        capture_substeps=bool(args.capture_substeps),
        action_lead_ticks=int(args.action_lead_ticks),
        validation_lanes=validation,
        selected_ticks=chosen,
        covered_validation_lanes=sorted(covered),
        native_response_support_is_not_certified=True,
    )
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
