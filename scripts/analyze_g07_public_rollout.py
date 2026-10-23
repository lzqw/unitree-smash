#!/usr/bin/env python3
"""Audit recorded public trajectories on CPU without assigning missing outcomes."""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    path = args.input/'public_dataset.pt'
    if not path.exists():
        path = args.input/'public_dataset.partial.pt'
    data = torch.load(path, map_location='cpu', weights_only=False)
    observation, action, active = [data[key] for key in
                                   ('observations', 'actions', 'active_first_episode')]
    if observation.ndim != 3 or observation.shape[-1] != 794:
        raise ValueError('Expected the recorded public794 observation contract')
    if action.shape != (*observation.shape[:2], 18) or active.shape != observation.shape[:2]:
        raise ValueError('Recorded time/environment axes differ')
    if active.dtype != torch.bool:
        raise ValueError('The first-episode mask must be explicit boolean data')
    outcomes = {}
    if data['completed']:
        receipt = json.loads((args.input/'completed.json').read_text())
        rows = json.loads((args.input/'episodes.json').read_text())
        if len(rows) != observation.shape[1] or receipt['episodes'] != len(rows):
            raise ValueError('Completed dataset/outcome count mismatch')
        outcomes = {row['scenario']: row for row in rows}
    manifest = json.loads((args.input/'eval_manifest.json').read_text())
    if data['checkpoint_sha256'] != manifest['checkpoint_sha256']:
        raise ValueError('Recorded action source checkpoint differs from run manifest')
    if data['panel_sha256'] != manifest['launch_panel_sha256']:
        raise ValueError('Recorded cases differ from run manifest')
    dt = data['control_dt_s']
    lanes = []
    for lane in range(observation.shape[1]):
        def first(index):
            found = torch.nonzero(active[:, lane] & (observation[:, lane, index]>.5)).flatten()
            return int(found[0]) if len(found) else None
        contact, landing = first(773), first(774)
        summary = dict(lane=lane, source_case=data['source_cases'][lane],
                       active_frames=int(active[:, lane].sum()),
                       first_public_post_contact_time_s=None if contact is None else contact*dt,
                       first_public_landing_time_s=None if landing is None else landing*dt)
        if landing is not None:
            states = []
            for lookback in (5, 3, 1, 0):
                index = max(0, landing-lookback)
                frame = observation[index, lane]
                gravity, velocity = frame[6:9], frame[:3]
                states.append(dict(time_s=index*dt, lookback_control_steps=lookback,
                                   world_vertical_velocity_mps=float(-(gravity*velocity).sum()),
                                   public_gravity_body=gravity.tolist(),
                                   body_velocity_mps=velocity.tolist(),
                                   body_angular_velocity_rad_s=frame[3:6].tolist(),
                                   foot_contact=frame[57:61].tolist(),
                                   joint_velocity_norm_rad_s=float(frame[27:45].norm())))
            summary['measured_states_near_first_public_landing'] = states
        if lane in outcomes:
            row = outcomes[lane]
            summary['completed_outcome'] = {key: row[key] for key in
                ('complete_smash', 'airborne_contact', 'recovery_success',
                 'robot_net_contact', 'numerical_failures')}
        lanes.append(summary)
    result = dict(kind='CPU audit of actual public-input/action collection; no new physical policy score',
                  source=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  dataset_completed=data['completed'], observation_shape=list(observation.shape),
                  action_shape=list(action.shape), active_frames=int(active.sum()),
                  finite_observations=bool(torch.isfinite(observation).all()),
                  finite_actions=bool(torch.isfinite(action).all()),
                  maximum_absolute_action=float(action.abs().max()),
                  no_teacher_action_labels=True, formal_acceptance=False,
                  clock_resolution_s=dt, lanes=lanes)
    output = args.input/'PUBLIC_DATA_AUDIT.json'
    temporary = output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, indent=2)+'\n')
    temporary.replace(output)
    print(json.dumps({key:value for key,value in result.items() if key!='lanes'}), flush=True)


if __name__ == '__main__':
    main()
