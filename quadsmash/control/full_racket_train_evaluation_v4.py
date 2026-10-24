"""TRAIN distribution checks without loading formal cases or changing task gates."""
import hashlib
import json
from pathlib import Path

import numpy as np


NOISE_SHAPES = {'joint_position_rad': (18,), 'joint_velocity_rad_s': (18,),
                'root_xy_m': (2,), 'root_yaw_rad': ()}


def training_cases(root, protocol, indices=None):
    suite = protocol['training_suite']
    source = Path(root) / suite['path']
    payload = source.read_bytes()
    if hashlib.sha256(payload).hexdigest() != suite['sha256']:
        raise ValueError('Original frozen TRAIN cases changed')
    cases = json.loads(payload)['cases']
    if len(cases) != suite['count']:
        raise ValueError('TRAIN count differs from frozen suite')
    chosen = list(range(len(cases))) if indices is None else list(indices)
    if (not chosen or any(type(i) is not int for i in chosen) or
            len(set(chosen)) != len(chosen) or min(chosen) < 0 or max(chosen) >= len(cases)):
        raise ValueError('Require distinct in-range TRAIN indices')
    return [cases[i] for i in chosen], chosen


def training_noise(count, amplitudes, seed):
    if type(count) is not int or count < 1:
        raise ValueError('Require a positive case count')
    if set(amplitudes) != set(NOISE_SHAPES):
        raise ValueError('Require the original four ground perturbation amplitudes')
    rng = np.random.default_rng(seed)
    result = {}
    for name, shape in NOISE_SHAPES.items():
        amplitude = amplitudes[name]
        if not np.isscalar(amplitude) or not np.isfinite(amplitude) or amplitude < 0:
            raise ValueError('Require finite nonnegative perturbation amplitudes')
        result[name] = rng.uniform(-amplitude, amplitude, (count, *shape))
    return result


def summarize_training(rows, cases, indices, protocol):
    from quadsmash.evaluation_v4 import FAULTS, independent_episode
    if (len(rows) != len(cases) or len(cases) != len(indices) or not rows or
            len(set(indices)) != len(indices)):
        raise ValueError('Each selected TRAIN case must have exactly one complete first episode')
    for i, (row, case, index) in enumerate(zip(rows, cases, indices)):
        if (row['env_id'] != i or row['episode_index'] != 0 or row['case_id'] != case['case_id'] or
                row['source_training_index'] != index or not row['ground_init'] or
                row['active_rally_launch_count'] > 1):
            raise ValueError('TRAIN order, first-episode or ground/single-launch contract violated')
        if type(row['numerical_failures']) is not int or row['numerical_failures'] < 0:
            raise ValueError('Numerical failure count must be a nonnegative integer')
    checked = [independent_episode(row, protocol) for row in rows]
    attacking = []
    for row, outcome in zip(rows, checked):
        velocity = np.asarray(row['outgoing_velocity_mps'], dtype=float)
        attacking.append(bool(outcome['airborne_contact'] and row['released'] and
            np.isfinite(velocity).all() and np.linalg.norm(velocity) >= 8. and
            velocity[0] > 0 and velocity[2] <= -1.))
    air = sum(row['airborne_contact'] for row in checked)
    recovered = sum(row['conditional_recovery'] for row in checked)
    complete = sum(row['complete'] for row in checked)
    return dict(episodes=len(rows), distinct_TRAIN_indices=indices, distinct_cases=len(set(indices)),
        complete=complete, complete_success_rate=complete/len(rows),
        airborne_contacts=air, airborne_contact_rate=air/len(rows),
        conditional_recovery_count=recovered, conditional_recovery_rate=recovered/air if air else 0.,
        numerical_failures=sum(row['numerical_failures'] for row in rows),
        actual_racket_contacts=sum(row['had_contact'] for row in rows),
        attack_outgoing=sum(attacking),
        crossed_net=sum(row['crossed_net'] for row in rows),
        landed_legal_court=sum(row['landed_legal_court'] for row in rows),
        landed_target=sum(row['landed_target'] for row in rows),
        fault_counts={name: sum(row[name] for row in rows) for name in FAULTS},
        formal_eligible=False, formal_acceptance=False, formal_cases_read=False,
        scope='Frozen TRAIN cases only; diagnostic performance of one fixed policy, not an A/B panel score.')
