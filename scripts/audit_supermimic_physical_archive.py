#!/usr/bin/env python3
"""Read committed native archive chunks on CPU; no simulation or optimization."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from quadsmash.rl.supermimic_physical_observation_v1 import PhysicalPolicyInputs, PhysicalPolicyObservationV1
from quadsmash.rl.supermimic_full_body_geometry_v1 import FullBodyGeometryV1


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    manifest = json.loads((args.archive/'manifest.json').read_text())
    if not manifest['chunks']:
        raise ValueError('No committed native data chunks')
    mapper = PhysicalPolicyObservationV1().float()
    geometry = FullBodyGeometryV1(manifest['task_contract']['asset']['sha256'])
    indices = [manifest['body_names'].index(name) for name in geometry.names]
    maximum = torch.zeros(794)
    absolute_sum = torch.zeros(794, dtype=torch.float64)
    body_max = {key: torch.zeros(len(indices)) for key in ('position', 'velocity', 'angular_velocity')}
    phases = torch.zeros(5, dtype=torch.long)
    events = torch.zeros(8, dtype=torch.long)
    samples = ticks = terminal = 0
    omissions = {}
    for entry in manifest['chunks']:
        path = args.archive/entry['file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('Chunk changed: '+str(path))
        chunk = torch.load(path, map_location='cpu', weights_only=True)
        for row in chunk['rows']:
            s = row['before']
            native = row['observation']
            # The archive was captured after observe_native inserted the
            # current frame. Supply its preceding ten frames to the pure map.
            history_before = torch.cat((s['history'][:, 1:], torch.zeros_like(s['history'][:, :1])), 1)
            physical = PhysicalPolicyInputs(**s['physical'])
            reconstructed, _ = mapper(physical, history_before)
            error = (reconstructed-native).abs()
            if not torch.isfinite(error).all():
                raise ValueError('Nonfinite physical reconstruction')
            maximum = torch.maximum(maximum, error.amax(0))
            absolute_sum += error.double().sum(0)
            phases += torch.bincount(native[:, 770:775].argmax(1), minlength=5)
            events += ((row['after']['event_state'] > .5) & (s['event_state'] <= .5)).sum(0)
            if not torch.equal(s['episode_index'], row['after']['episode_index']):
                raise ValueError('Transition crosses reset')
            for side in ('before', 'after'):
                record = row[side]; p = record['physical']
                fk = geometry(p['joint_position'], p['joint_velocity'], p['root_position_w'],
                              p['root_quaternion_w'], p['root_link_velocity_w'], p['root_angular_velocity_w'])
                for key, native_key in (('position', 'body_position_w'), ('velocity', 'body_linear_velocity_w'),
                                        ('angular_velocity', 'body_angular_velocity_w')):
                    delta = (fk[key]-record[native_key][:, indices]).abs()
                    if not torch.isfinite(delta).all():
                        raise ValueError('Nonfinite body geometry')
                    body_max[key] = torch.maximum(body_max[key], delta.amax((0, 2)))
            for memory in ('executor_before', 'executor_after'):
                omissions.update(row[memory]['omitted'])
            samples += len(native); ticks += 1
            terminal += int((row['terminated'] | row['timeout']).sum())
    result = dict(scope='Committed-chunk CPU audit; not held-out dynamics validation or task scoring',
                  archive=str(args.archive.resolve()), chunks=len(manifest['chunks']), ticks=ticks,
                  transitions=samples, terminal_transitions=terminal, phases=phases.tolist(),
                  event_transitions=events.tolist(),
                  observation_max_abs=float(maximum.max()), observation_max_channel=int(maximum.argmax()),
                  observation_channel_max=maximum.tolist(), observation_channel_mae=(absolute_sum/samples).tolist(),
                  full_body_max_by_name={name:{key:float(value[i]) for key,value in body_max.items()}
                                         for i,name in enumerate(geometry.names)}, executor_omissions=omissions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('chunks','ticks','transitions','terminal_transitions',
                                                 'phases','observation_max_abs','observation_max_channel')}))


if __name__ == '__main__':
    main()
