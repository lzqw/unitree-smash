"""Coupled physical search ranking; predictions never create success flags."""
from __future__ import annotations

import math
import numpy as np

from quadsmash.physics.smash_events import rank_candidate
from quadsmash.physics.smash_feasibility import predict_flights


def add_flight_search_diagnostics(records):
    """Predict all measured outgoing flights in one CPU batch, without bounces.

    A high upward lob cannot receive zero violation just because it has a
    positive net margin. These terms optimize the coupled necessary conditions.
    """
    measurable = []
    for index, row in enumerate(records):
        p = np.asarray(row.get('outgoing_position_m', row['contact_position_m']))
        v = np.asarray(row['outgoing_velocity_mps'])
        row['coupled_flight_violation'] = 10.
        row['predicted_legal_attack'] = False
        row['predicted_net_margin_m'] = None
        row['predicted_first_landing_position_m'] = None
        if p.shape == (3,) and v.shape == (3,) and np.isfinite(p).all() and np.isfinite(v).all():
            measurable.append((index, p, v))
    if not measurable:
        return
    positions = np.stack([r[1] for r in measurable])
    velocities = np.stack([r[2] for r in measurable])
    prediction = predict_flights(positions, velocities)
    for lane, (index, p, v) in enumerate(measurable):
        row = records[index]
        speed = float(np.linalg.norm(v))
        margin = float(prediction['conservative_net_exit_margin'][lane])
        net_position = prediction['net_position'][lane]
        landing = prediction['landing_position'][lane]
        attack_error = (max(0., 8.-speed)/8. + max(0., v[2]+1.)/4.
                        + max(0., -v[0])/8.)
        net_error = ((max(0., .01-margin)/.5
                      + max(0., abs(net_position[1])+.025-1.))
                     if math.isfinite(margin) and np.isfinite(net_position).all() else 6.)
        if np.isfinite(landing).all():
            landing_error = (max(0., 2.-landing[0], landing[0]-4.)
                             + max(0., abs(landing[1])-1.))/2.
        else:
            landing_error = 3.
        invalid = (not row.get('airborne_contact', False)
                   or not row.get('effective_face_contact', False)
                   or row.get('contact_count', 0) != 1
                   or row.get('numerical_failures', 0)
                   or row.get('sweep_misses', 0) or row.get('sweep_unresolved', 0))
        row['coupled_flight_violation'] = float(attack_error + net_error + landing_error + 10.*bool(invalid))
        row['predicted_legal_attack'] = bool(prediction['predicted_legal_attack'][lane] and not invalid)
        row['predicted_net_margin_m'] = margin if math.isfinite(margin) else None
        row['predicted_first_landing_position_m'] = landing.tolist() if np.isfinite(landing).all() else None


def search_candidate_key(row):
    """Actual ordered results first, coupled violation second, no raw speed bonus."""
    actual = rank_candidate(row)
    distance = float(row.get('interception_min_distance_m', float('inf')))
    distance = distance if math.isfinite(distance) else 3.
    return (*actual[:3], -float(row['coupled_flight_violation']), -distance)
