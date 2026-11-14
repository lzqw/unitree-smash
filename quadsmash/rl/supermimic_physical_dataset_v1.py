"""Compact native physical states and reset-safe supervised windows.

The 55D vector is robot/ball kinematics, NOT the whole Markov state. Contact,
task and executor memory remain separate. Models consume actual final joint
targets, never label residual requests as if those were native PD targets.
"""
import hashlib
import json
from pathlib import Path

import torch


STATE_FIELDS = (
    ('root_position_w', 3), ('root_quaternion_w', 4),
    ('root_link_velocity_w', 3), ('root_angular_velocity_w', 3),
    ('joint_position', 18), ('joint_velocity', 18),
    ('ball_position_w', 3), ('ball_velocity_w', 3),
)
STATE_DIM = sum(width for _, width in STATE_FIELDS)


def encode_physical(physical):
    n = len(physical['joint_position'])
    for name, width in STATE_FIELDS:
        if physical[name].shape != (n, width):
            raise ValueError('Physical field shape: '+name)
    return torch.cat([physical[name] for name, _ in STATE_FIELDS], -1)


def decode_physical(vector):
    if vector.shape[-1] != STATE_DIM:
        raise ValueError('Expected55D physical vector')
    return dict(zip((name for name, _ in STATE_FIELDS),
                    vector.split([width for _, width in STATE_FIELDS], -1)))


def valid_windows(episode, terminated, timeout, steps):
    """[T,N] starts; a terminal may end a window, but cannot occur inside it."""
    if episode.ndim != 2 or terminated.shape != episode.shape or timeout.shape != episode.shape:
        raise ValueError('Expected [time,lane] episode/terminal arrays')
    if steps < 1 or steps > len(episode):
        raise ValueError('Window length outside recorded data')
    e = episode.unfold(0, steps, 1)
    done = (terminated | timeout).unfold(0, steps, 1)
    return (e == e[..., :1]).all(-1) & (~done[..., :-1]).all(-1)


def split_lanes(count, validation_count, seed):
    if not 0 < validation_count < count:
        raise ValueError('Both train and validation require complete lanes')
    order = torch.randperm(count, generator=torch.Generator().manual_seed(seed))
    return order[validation_count:].sort().values, order[:validation_count].sort().values


def pack_archive(directory):
    """Read only manifest-committed chunks; preserve pre-reset next states.

    Auxiliary archive data is retained at source rather than pretending the
    compact tensor package contains every controller state or substep event.
    """
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    if not manifest['chunks']:
        raise ValueError('No committed physical transitions')
    records = []
    previous_tick = None
    for chunk in manifest['chunks']:
        path = directory/chunk['file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != chunk['sha256']:
            raise ValueError('Archive hash mismatch: '+chunk['file'])
        rows = torch.load(path, map_location='cpu', weights_only=True)['rows']
        for row in rows:
            before, after = row['before'], row['after']
            tick = before['control_tick']
            if previous_tick is not None and tick != previous_tick + 1:
                raise ValueError('Missing native tick between chunks')
            if after['control_tick'] != tick + 1 or not torch.equal(before['episode_index'], after['episode_index']):
                raise ValueError('Invalid physical transition boundary')
            previous_tick = tick
            record = dict(state=encode_physical(before['physical']),
                          next_state=encode_physical(after['physical']),
                          observation=row['observation'],
                          residual_request=row['requested_action'],
                          base_action=row['base_action'],
                          final_action=row['final_action'],
                          final_joint_targets=row['final_joint_targets'],
                          reward=row['reward'], terminated=row['terminated'], timeout=row['timeout'],
                          episode=before['episode_index'],
                          elapsed=before['physical']['elapsed_s'],
                          foot_contact=before['physical']['foot_contact'],
                          next_foot_contact=after['physical']['foot_contact'],
                          ball_present=before['physical']['ball_present'],
                          next_ball_present=after['physical']['ball_present'],
                          event_state=before['event_state'], next_event_state=after['event_state'])
            # Clone the selected arrays so no retained view pins the large
            # executor/geometry dictionaries while reading later chunks.
            records.append({key:value.clone() for key,value in record.items()})
    return dict(schema='supermimic-physical-supervision-v1',
                source_archive=str(directory.resolve()), source_manifest=manifest,
                state_fields=STATE_FIELDS,
                limitation='Requires source archive for executor/task memory; no substep contact labels',
                tensors={key:torch.stack([row[key] for row in records]) for key in records[0]})
