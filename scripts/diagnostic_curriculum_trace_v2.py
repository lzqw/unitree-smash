"""Reserve failure traces for faster valid returns in future native training.

Keep the v1 recorder, its quotas, and every original outcome unchanged. Two
additional storage bands prevent early low-speed failures from consuming the
entire diagnostic allowance. These are recording bands, not task thresholds.
"""
from collections import Counter
import hashlib
import math

from quadsmash.rl.episode_trace_v4 import EpisodeTraceRecorderV4, trace_settings
from scripts.diagnostic_curriculum_trace_v1 import (
    DiagnosticCurriculumTraceSelection, DiagnosticCurriculumEpisodeTraceRecorderV1,
    diagnostic_retention_settings, retention_settings, PROTOCOL, SCHEMA)

SPEED_SCHEMA = 'ball-positive-speed-banded-recovery-failure-retention-v2'
SPEED_QUOTAS = {
    'ball_positive_speed7_to8_recovery_failure': 'max_speed7_to8_failure_traces',
    'ball_positive_speed8_plus_recovery_failure': 'max_speed8_plus_failure_traces',
}


def speed_diagnostic_retention_settings(value):
    if not isinstance(value, dict) or set(value) != set(SPEED_QUOTAS.values()):
        raise ValueError('Declare both finite speed-band trace quotas')
    if any(type(v) is not int or v < 0 for v in value.values()):
        raise ValueError('Speed-band quotas must be nonnegative integers')
    return dict(value)


def speed_failure_band(speed):
    if speed is None or not math.isfinite(speed) or speed < 7.0:
        return None
    if speed < 8.0:
        return 'ball_positive_speed7_to8_recovery_failure'
    return 'ball_positive_speed8_plus_recovery_failure'


class DiagnosticCurriculumTraceSelectionV2(DiagnosticCurriculumTraceSelection):
    def __init__(self, settings, retention, diagnostic_retention,
                 speed_diagnostic_retention, protocol=None):
        super().__init__(settings, retention, diagnostic_retention, protocol)
        self.speed_diagnostic_retention = speed_diagnostic_retention_settings(
            speed_diagnostic_retention)
        self.speed_observed = Counter()
        self.speed_selected = Counter()
        self.speed_retained = Counter()
        self.speed_skipped = Counter()
        self.speed_saved_candidates = {}

    def choose(self, row, *, first_predeclared=False):
        label, kinds, reasons = super().choose(row, first_predeclared=first_predeclared)
        previously_selected = bool(reasons)
        speed = label['outgoing_speed_mps']
        band = speed_failure_band(speed)
        eligible = (band is not None and kinds['failure']
                    and speed >= self.retention['minimum_speed_mps']
                    and not label['original_outcomes']['conditional_recovery']
                    and all(value for name, value in label['criteria_except_speed'].items()
                            if name != 'same_flight_stable_recovery'))
        if not eligible:
            return label, kinds, reasons
        self.speed_observed[band] += 1
        if self.speed_selected[band] < self.speed_diagnostic_retention[SPEED_QUOTAS[band]]:
            self.speed_selected[band] += 1
            reasons.append(band + '_until_declared_limit')
        key = (row['env_id'], row['episode_index'])
        if reasons:
            self.speed_retained[band] += 1
            self.speed_saved_candidates[key] = band
        else:
            self.speed_skipped[band] += 1
        if reasons and not previously_selected:
            # Both parent selectors counted this still-failed episode as skipped.
            # Reconcile their accounting without spending either parent's quota.
            self.retained['episodes'] += 1
            for kind, value in kinds.items():
                self.retained[kind] += int(value)
                self.skipped[kind] -= int(value)
                if self.skipped[kind] < 0:
                    raise ValueError('Invalid original retention accounting')
            kind = ('ball_positive_nonfoot_recovery_failure'
                    if row['contact_flight_recovery']['body_landing']
                    else 'ball_positive_other_recovery_failure')
            self.diagnostic_retained[kind] += 1
            self.diagnostic_skipped[kind] -= 1
            if self.diagnostic_skipped[kind] < 0:
                raise ValueError('Invalid v1 diagnostic retention accounting')
            self.diagnostic_saved_candidates[key] = kind
        return label, kinds, reasons


class DiagnosticCurriculumEpisodeTraceRecorderV2(DiagnosticCurriculumEpisodeTraceRecorderV1):
    def __init__(self, output, count, max_steps, body_names, settings, provenance,
                 *, retention, diagnostic_retention, speed_diagnostic_retention):
        self.selection = DiagnosticCurriculumTraceSelectionV2(
            settings, retention, diagnostic_retention, speed_diagnostic_retention)
        self.protocol_sha256 = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
        if provenance.get('protocol_sha256') != self.protocol_sha256:
            raise ValueError('Trace source must use the unchanged original protocol')
        if provenance.get('asset_sha256') != self.selection.protocol['asset']['sha256']:
            raise ValueError('Trace source belongs to a different loaded asset')
        EpisodeTraceRecorderV4.__init__(
            self, output, count, max_steps, body_names, settings, provenance)

    def statistics(self):
        result = super().statistics()
        actual = Counter()
        for saved in self.saved:
            band = self.selection.speed_saved_candidates.get(
                (saved['env_id'], saved['episode_index']))
            if band is not None:
                actual[band] += 1
        result.update(
            speed_diagnostic_selection_schema=SPEED_SCHEMA,
            speed_diagnostic_retention_settings=self.selection.speed_diagnostic_retention,
            speed_diagnostic_observed=dict(self.selection.speed_observed),
            speed_diagnostic_quota_selected=dict(self.selection.speed_selected),
            speed_diagnostic_selection_retained=dict(self.selection.speed_retained),
            speed_diagnostic_actual_retained=dict(actual),
            speed_diagnostic_quota_skipped=dict(self.selection.speed_skipped),
            speed_bands_are_storage_categories_not_task_thresholds=True)
        return result
