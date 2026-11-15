"""Read-only ball-side quality shadow diagnostics.

This module never changes rewards or accepts/rejects an action. It reports
which observed native fields support each ball-side condition and keeps
missing evidence explicit.
"""
from __future__ import annotations


BALL_FIELDS = (
    'qualified_airborne_string_contact', 'ball_side_complete_except_recovery',
    'outgoing_velocity', 'crossed_net', 'landed_legal_court', 'landed_target',
    'net_margin', 'rally_fault', 'contact_point', 'contact_time',
)


def ball_side_shadow_row(row):
    """Return observed evidence only; no predicted event is substituted."""
    evidence={name: row.get(name, None) for name in BALL_FIELDS}
    missing=[name for name,value in evidence.items() if value is None]
    return {
        'case_id': row.get('case_id', row.get('source_case_index')),
        'episode_id': row.get('episode_index'),
        'observed': evidence,
        'missing_fields': missing,
        'shadow_ball_side_ok': (None if missing else bool(row['ball_side_complete_except_recovery'])),
        'enforce': False,
    }


def summarize_ball_side_shadow(rows):
    reports=[ball_side_shadow_row(row) for row in rows]
    return {
        'mode': 'shadow', 'enforce': False, 'count': len(reports),
        'complete_observed': sum(r['shadow_ball_side_ok'] is True for r in reports),
        'missing_evidence_rows': sum(bool(r['missing_fields']) for r in reports),
        'rows': reports,
    }
