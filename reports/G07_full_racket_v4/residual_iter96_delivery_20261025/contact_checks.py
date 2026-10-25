"""Match actual complete episode rows to their native string-contact records."""
import json


def match_complete_contacts(rows, contacts, protocol, required_rows=()):
    import numpy as np
    from quadsmash.evaluation_v4 import independent_episode

    by_episode = {}
    for event in contacts:
        by_episode.setdefault((event['env_id'], event['episode_index']), []).append(event)
    indexed = {(row['env_id'], row['episode_index']): row for row in rows}
    assert len(indexed) == len(rows)
    for preserved in required_rows:
        key = preserved['env_id'], preserved['episode_index']
        assert key in indexed
        assert json.dumps(indexed[key], sort_keys=True) == json.dumps(preserved, sort_keys=True)
    checked = [independent_episode(row, protocol) for row in rows]
    counts = dict(completed_episodes=len(rows), air=sum(c['airborne_contact'] for c in checked),
        conditional_recovery_count=sum(c['conditional_recovery'] for c in checked),
        complete=sum(c['complete'] for c in checked),
        numerical_failures=sum(row['numerical_failures'] for row in rows))
    matched = []
    for row, result in zip(rows, checked):
        if not result['complete']:
            continue
        # A frame/string transition in one permitted stroke can have more
        # than one recorded event. Match the actual first string event.
        events = by_episode.get((row['env_id'], row['episode_index']), [])
        candidates = [event for event in events if event['string_contact']
            and abs(event['time_s'] - row['first_string_time_s']) <= 1e-7
            and np.max(abs(np.asarray(event['point_m']) - np.asarray(row['first_string_contact_point_m']))) <= 1e-7]
        assert len(candidates) == 1, 'Missing/ambiguous native string contact: ' + row['case_id']
        event = candidates[0]
        assert not event['fixture_contact'] and event['legal_initial_contact']
        assert not any(event['fault_flags'].values())
        poses = np.asarray(event['body_link_poses_local'])
        assert poses.shape == (28, 7) and np.isfinite(poses).all()
        expected = np.asarray(event['flange_linear_velocity_mps']) + np.cross(
            event['flange_angular_velocity_rad_s'],
            np.asarray(event['point_m']) - np.asarray(event['flange_position_m']))
        error = float(np.linalg.norm(expected - np.asarray(event['point_velocity_mps'])))
        assert np.isfinite(error) and error <= 1e-4
        matched.append(dict(case_id=row['case_id'], env_id=row['env_id'], episode_index=row['episode_index'],
            exact_contact_record_matches=True, native_events_for_episode=len(events),
            rigid_point_velocity_identity_error_mps=error, actual_native_contact=event))
    assert len(matched) == counts['complete']
    return dict(independently_recomputed_episode_counts=counts,
        matched_complete_episodes=len(matched), exact_contact_record_matches=True, matches=matched,
        velocity_scope='Recorded post-PhysX contact integration state; not independently measured incident impact velocity.',
        full_training_motion_trajectory_available=False, formal_acceptance=False)
