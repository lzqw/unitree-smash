"""Select failure families before optimization and transfer a reference proposal.

Only the reference is translated/yaw-aligned. Measured state, velocity,
contact and previous-command inputs are never transformed or replaced.
"""
import copy
import math

import numpy as np
import pinocchio as pin


def select_cases(transitions, episodes, first_events):
    failures = sorted(set(transitions['ball5_except_recovery']['retained'])
        - set(transitions['complete5']['retained']))
    if len(failures) != 26:
        raise ValueError('Require the declared original 26-case recovery cohort')
    first = {}
    for event in first_events:
        for lane in event['first_event_env_ids']:
            first.setdefault(lane, event['physics_frame'] * .0000625)
    rows = {r['env_id']: r for r in episodes if r['episode_index'] == 0}
    falling, no_contact = [], []
    for lane in failures:
        row = rows[lane]; landing = row['contact_flight_recovery']['contact_flight_landing_s']
        value = dict(case=lane, release_time_s=row['release_time_s'], landing_time_s=landing,
            first_scored_nonfoot_s=first.get(lane),
            delay_after_landing_s=first[lane]-landing if lane in first else None)
        (falling if lane in first else no_contact).append(value)
    # Exclude the already optimized TRAIN3, not cases that prove difficult.
    ranked = sorted([r for r in falling if r['case'] != 3], key=lambda r: (r['delay_after_landing_s'], r['case']))
    if len(falling) != 22 or len(no_contact) != 4:
        raise ValueError('Changed original failure classification')
    selected = [dict(ranked[i], selection_reason=label) for i, label in
        [(0, 'earliest_scored_nonfoot'), (len(ranked)//2, 'median_scored_nonfoot'),
         (len(ranked)-1, 'latest_scored_nonfoot')]]
    selected.append(dict(min(no_contact, key=lambda r:r['case']),
        selection_reason='lowest_id_without_scored_nonfoot_but_failed_recovery'))
    if len({r['case'] for r in selected}) != 4:
        raise ValueError('Distinct held-out-from-TRAIN3 cases required')
    if any(not 0 < r['release_time_s'] < 2.8 < r['landing_time_s'] for r in selected):
        raise ValueError('Fixed 2.8 second query must follow release and precede landing')
    return dict(cohort=failures, selected=selected, control_tick=280,
        selection_uses_candidate_outcomes=False, native_success_claim=False,
        note='Original scoring omits some head contacts; no-scored-nonfoot is not all-body safety.')


def transfer_reference(initargs, state):
    args = list(copy.deepcopy(initargs))
    original = np.asarray(args[1]['initial_q'], dtype=float)
    old_rotation = pin.Quaternion(original[3:7]).matrix()
    new_rotation = pin.Quaternion(state.q[3:7]).matrix()
    yaw = math.atan2(new_rotation[1,0], new_rotation[0,0]) - math.atan2(old_rotation[1,0], old_rotation[0,0])
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
    reference = np.asarray(args[4], dtype=float).copy()
    positions = (reference[:,:3] - original[:3]) @ rotation.T + original[:3]
    positions[:,:2] += state.q[:2] - original[:2]
    reference[:,:2] = positions[:,:2]
    if yaw != 0.:
        for row in reference:
            row[3:7] = pin.Quaternion(rotation @ pin.Quaternion(row[3:7]).matrix()).coeffs()
    # Free-flyer velocities are in the body frame: a global yaw transform
    # changes the reference pose, not its body-frame tangent components.
    args[4] = reference
    args[1].update(initial_q=state.q.copy(), initial_v=state.v.copy(),
        initial_contacts=state.contacts.copy(), actual_previous_command=state.previous_command_target.copy(), tick=280)
    return tuple(args), dict(reference_only_planar_rotation_rad=yaw,
        reference_only_translation_xy_m=(state.q[:2]-original[:2]).tolist(),
        reference_height_and_joint_trajectory_preserved=True, measured_state_transformed=False)
