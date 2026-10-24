"""TRAIN search priorities; never changes native rally or acceptance labels."""
import numpy as np


ROBOT_FAULTS = ('robot_net_fault', 'body_ball_fault', 'wrong_side_fault',
                'double_hit_fault', 'racket_invasion_fault', 'body_over_net_fault',
                'under_net_obstruction_fault', 'engineering_body_crossing')


def usable_air_contact(row):
    """Preserve an actual legal airborne strings event for further TRAIN work.

    A blocked shuttle or failed body recovery is still a failed full smash.
    Those failures supply learning feedback instead of discarding the contact
    candidate in favor of a policy which never touches the incoming shuttle.
    """
    return bool(row['legal_initial_contact'] and row['airborne_string_hit']
                and row['string_hit'] and row['numerical_failures'] == 0
                and not any(row[name] for name in ROBOT_FAULTS))


def contact_search_tiers(groups):
    if not groups or any(len(g) != len(groups[0]) for g in groups):
        raise ValueError('Require equal nonempty TRAIN case groups')
    case_count = len(groups[0])
    if not case_count:
        raise ValueError('At least one TRAIN case required')
    tiers = []
    for group in groups:
        if any(row['numerical_failures'] for row in group):
            tiers.append(-1)
            continue
        complete = sum(bool(row['complete_smash']) for row in group)
        contacts = sum(usable_air_contact(row) for row in group)
        tiers.append(complete * (case_count + 1) + contacts)
    return np.asarray(tiers, dtype=np.int64)


def actual_outgoing_shaping(row):
    """Bounded guidance toward positive forward and downward actual velocity.

    The original 8 m/s, vx>0, vz<=-1 attack gate and every full success flag
    remain native measurements. Commanded racket velocity earns no reward.
    """
    if not usable_air_contact(row) or not row['released']:
        return 0.
    outgoing = np.asarray(row['outgoing_velocity_mps'], dtype=float)
    if outgoing.shape != (3,) or not np.isfinite(outgoing).all():
        return 0.
    return float(10 * np.clip(outgoing[0] / 8., 0., 1.)
                 + 3 * np.clip(-outgoing[2], 0., 1.))


def arm_timing_velocity_grid(reference, lower, upper):
    """Change arm timing/speed only; keep the measured leg timing intact."""
    result = []
    for lead_delta in [-.06, -.03, 0., .03]:
        for speed_multiplier in [.7, 1., 1.3]:
            candidate = np.asarray(reference, dtype=float).copy()
            candidate[14] = np.clip(candidate[14] + lead_delta, lower[14], upper[14])
            candidate[7] = np.clip(candidate[7] * speed_multiplier, lower[7], upper[7])
            result.append(candidate)
    return np.asarray(result)


def measured_recovery_leg_crossovers(reference, recovered, lower, upper):
    """Native TRAIN proposals mixing measured recovery legs with contact arms.

    No outcome transfers to these new coupled trajectories. All native fitness
    must be measured afresh, with the same physical limits and success gates.
    """
    reference = np.asarray(reference, dtype=float)
    recovered = np.asarray(recovered, dtype=float)
    if reference.shape != (26,) or recovered.shape != (26,):
        raise ValueError('Require compatible 26-parameter controller families')
    body_indices = [0,1,2,3,4,5,6,17,20,21,22,23]
    result = []
    for fraction in [.25, .5, .75, 1.]:
        for lead_delta in [-.03, 0.]:
            candidate = reference.copy()
            candidate[body_indices] = np.clip(
                (1-fraction)*reference[body_indices]+fraction*recovered[body_indices],
                np.asarray(lower)[body_indices], np.asarray(upper)[body_indices])
            candidate[14] = np.clip(reference[14]+lead_delta, lower[14], upper[14])
            result.append(candidate)
    return np.asarray(result)


def pitch_compensation_grid(reference, lower, upper):
    """Probe existing public-gravity compensation, preserving other controls."""
    result = []
    for gain in [.4, .7, 1., 1.3]:
        for lead_delta in [-.03, 0., .03]:
            candidate = np.asarray(reference, dtype=float).copy()
            candidate[15] = np.clip(gain, lower[15], upper[15])
            candidate[14] = np.clip(reference[14]+lead_delta, lower[14], upper[14])
            result.append(candidate)
    return np.asarray(result)
