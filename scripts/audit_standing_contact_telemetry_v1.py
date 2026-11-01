"""Compare additional native contact reads with the unchanged standing metric."""
import json
from pathlib import Path

import numpy as np

from scripts.audit_whole_body_standing_v1 import audit
from scripts.standing_contact_telemetry_v1 import path_identity
from scripts.run_landing_validation_sequence_v1 import sha


def audit_telemetry(source):
    source = Path(source)
    result = audit(source)
    meta = json.loads((source/'contact_telemetry_metadata.json').read_text())
    events = json.loads((source/'first_nonfoot_contact_details.json').read_text())
    with np.load(source/'native_contact_telemetry.npz') as data:
        frames = data['physics_frame']; clock = data['clock_s']
        nonfoot = data['nonfoot_ground_force_world_n']; foot = data['foot_net_force_world_n']
    with np.load(source/'physics_support_metrics.npz') as data:
        original_frames = data['physics_frame']; metrics = data['values']
    if (not np.array_equal(frames, original_frames) or not np.array_equal(clock, metrics[:, :, 0])
            or not np.isfinite(nonfoot).all() or not np.isfinite(foot).all()):
        raise ValueError('Telemetry is not aligned with the actual original metric samples')
    total = np.linalg.norm(nonfoot, axis=-1).sum(-1)
    if not np.allclose(total, metrics[:, :, 7], atol=1e-4, rtol=2e-6):
        raise ValueError('Per-body records do not reproduce the original aggregate')
    ordering = []
    offsets = {}; offset = 0
    for sensor in meta['sensors']:
        count = sensor['reported_force_matrix_shape'][1]
        offsets[sensor['name']] = (offset, count)
        offset += count
        expected = [(i, body) for i in range(nonfoot.shape[1]) for body in sensor['body_names']]
        actual = [path_identity(p) for p in sensor['sensor_paths']]
        ordering.append(dict(name=sensor['name'],
            paths_match_original_reshape_order=actual == expected,
            sensor_and_rigid_body_views_match=sensor['sensor_paths'] == sensor['body_view_paths'],
            actual_sensor_identities=actual, expected_reshape_identities=expected))
    if offset != nonfoot.shape[2]:
        raise ValueError('Sensor metadata does not cover all saved body columns')
    contacts = []
    event_lanes = []
    for event in events:
        indices = np.flatnonzero(frames == event['physics_frame'])
        if len(indices) != 1:
            raise ValueError('First event has no unique actual physics sample')
        tick = int(indices[0]); event_lanes += event['first_event_env_ids']
        for group in event['sensors']:
            start, count = offsets[group['name']]
            matrix = np.array(group['force_matrix_world_n'])
            if not np.array_equal(matrix[:, :, 0], nonfoot[tick, :, start:start+count]):
                raise ValueError('First-event force record changed from continuous telemetry')
            for item in group['native_patch_contacts']:
                native = np.array(item['summed_normal_force_world_n'])
                aggregate = matrix.reshape(-1, matrix.shape[2], 3)[item['sensor_row'], item['filter_column']]
                contacts.append(dict(physics_frame=event['physics_frame'],
                    sensor_group=group['name'], env_id=item['env_id'], body=item['body'],
                    point_count=item['count'], recorded_matrix_force_world_n=aggregate.tolist(),
                    raw_patch_force_world_n=native.tolist(),
                    maximum_force_disagreement_n=float(abs(native-aggregate).max()),
                    contact_point_z_m=[x[2] for x in item['point_world_m']],
                    separation_m=item['separation_m']))
    expected_events = set(np.flatnonzero(((metrics[:, :, 7] > 1.) & (metrics[:, :, 10] > .5)).any(0)).tolist())
    if set(event_lanes) != expected_events or len(event_lanes) != len(set(event_lanes)):
        raise ValueError('First nonfoot events were omitted or duplicated')
    result['additional_telemetry'] = dict(physics_samples=len(frames),
        aggregate_maximum_error_n=float(abs(total-metrics[:, :, 7]).max()),
        original_outcome_scoring_unchanged=True, ordering=ordering,
        first_event_lanes=event_lanes, raw_patch_comparisons=contacts,
        physical_collision_cause_not_automatically_certified=True)
    for name in ['contact_telemetry_metadata.json', 'first_nonfoot_contact_details.json',
                 'native_contact_telemetry.npz']:
        result['evidence_sha256'][name] = sha(source/name)
    return result
