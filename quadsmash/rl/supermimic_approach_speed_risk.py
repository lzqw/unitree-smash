"""Shared Approach risk loss and frozen-policy comparisons."""

import math

import torch


GROUPS = {1: 'low_speed', 2: 'marginal_success'}


def risk_group(outcome, terminal):
    if outcome == 'low_speed':
        return 1
    speed = terminal.get('outgoing_speed_mps')
    return 2 if outcome == 'success' and speed is not None and 4.5 <= speed < 4.75 else 0


def release_risk(speed, speed_loss):
    """Retain a speed derivative when the calibrated threshold loss saturates."""
    return speed_loss / (1. + speed_loss) + 4.5 / (4.5 + speed)


def incomplete_risk(geometry_cost):
    """Keep misses and invalid releases above every completed release."""
    geometry = geometry_cost.clamp_min(0.)
    return 2. + geometry / (1. + geometry)


def group_mean(values, groups):
    masks = groups[:, None] == groups.new_tensor(tuple(GROUPS))
    counts = masks.sum(0)
    means = (values[:, None] * masks).sum(0) / counts.clamp_min(1)
    return means.sum() / counts.gt(0).sum().clamp_min(1)


def risk_statistics(rollout):
    values, groups, released = rollout['speed_risk_by_lane']
    return {name: dict(lanes=int(mask.sum()), mean=float(values[mask].mean()),
                       releases=int((mask & released).sum()))
            for code, name in GROUPS.items() if (mask := groups == code).any()}


def compare_risk(before, after, tolerance):
    old, new = risk_statistics(before), risk_statistics(after)
    _, old_groups, old_release = before['speed_risk_by_lane']
    _, new_groups, new_release = after['speed_risk_by_lane']
    if not torch.equal(old_groups, new_groups):
        raise ValueError('Speed risk comparison requires identical native start groups')
    result = {}
    for code, name in GROUPS.items():
        if name not in old:
            continue
        lost = int(((old_groups == code) & old_release & ~new_release).sum())
        initial, trial = old[name], new[name]
        ok = (math.isfinite(trial['mean']) and trial['mean'] <= initial['mean'] + tolerance
              and (code != 2 or not lost))
        result[name] = dict(before=initial, after=trial, lost_releases=lost, passed=ok)
    return all(row['passed'] for row in result.values()), result


def forward_tolerances(first, repeated):
    """Maximum observed same-state differences, rounded up by one float ULP."""
    old, new = (row['flight_cost_by_lane'] for row in (first, repeated))
    finite = old.isfinite()
    if not torch.equal(finite, new.isfinite()):
        raise ValueError('Repeated frozen forward changed flight support')
    a, groups, released = first['speed_risk_by_lane']
    b, groups_repeat, released_repeat = repeated['speed_risk_by_lane']
    if not torch.equal(groups, groups_repeat) or not torch.equal(released, released_repeat):
        raise ValueError('Repeated frozen forward changed risk-group support')
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise FloatingPointError('Nonfinite repeated speed risk')
    flight_difference = float((old[finite] - new[finite]).abs().max()) if finite.any() else 0.
    risk_difference = float((a - b).abs().max())
    eps = torch.finfo(a.dtype).eps
    flight_scale = max(1., float(old[finite].abs().max())) if finite.any() else 1.
    return dict(flight=flight_difference + eps * flight_scale, risk=risk_difference + eps * max(1., float(a.abs().max())),
                observed_flight_difference=flight_difference, observed_risk_difference=risk_difference)


def rejudge_history(history, tolerances):
    """Check the comparable portions of archived trials with the measured tolerance."""
    rows = []
    for entry in history['attempts']:
        if not entry['update_accepted'] or entry['flight_merit'] is None:
            continue
        old_rule = entry['flight_merit']
        flight_delta = old_rule['after'] - old_rule['before']
        groups = {name: [] for name in GROUPS.values()}
        incomplete = {name: 0 for name in GROUPS.values()}
        for pair in entry['accepted_guard_pairs']:
            before, after = pair['before'], pair['after']
            name = before['risk_group']
            if before['speed_loss'] is None or after['speed_loss'] is None:
                incomplete[name] += 1
                if before['speed_loss'] is not None:
                    groups[name].append(2. - float(release_risk(before['release_speed_mps'], before['speed_loss'])))
                continue
            groups[name].append(float(release_risk(after['release_speed_mps'], after['speed_loss']) -
                                      release_risk(before['release_speed_mps'], before['speed_loss'])))
        rows.append(dict(attempt=entry['attempt'], flight_delta=flight_delta,
                         flight_failed=flight_delta > tolerances['flight'],
                         risk_groups={name: dict(incomplete=incomplete[name],
                             comparable_risk_delta=sum(values) / len(values) if values else None,
                             comparable_failed=bool(values and sum(values) / len(values) > tolerances['risk']),
                             complete_geometry_evidence=incomplete[name] == 0)
                             for name, values in groups.items()}))
    return dict(total_logged=history['total_proposals'], measured_tolerances=tolerances, accepted_trials=rows,
                scope='Archived partial line-search reconstruction; missing geometry is unresolved, not safe.')
