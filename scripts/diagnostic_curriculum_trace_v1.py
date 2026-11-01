"""Retain key native recovery failures without changing the learner or labels.

The existing v1 trace data format and all original retention categories remain.
Two additional finite storage quotas preserve already valid curriculum returns
that fail recovery, separating nonfoot ground contact from other recovery
failures. Rule-faulted returns cannot enter these two diagnostic categories.
"""
from collections import Counter
import hashlib

from quadsmash.rl.episode_trace_v4 import EpisodeTraceRecorderV4, trace_settings
from scripts.curriculum_episode_trace_v1 import (
    CurriculumTraceSelection, CurriculumEpisodeTraceRecorderV1,
    retention_settings, PROTOCOL, SCHEMA)

DIAGNOSTIC_SCHEMA='ball-positive-recovery-failure-retention-v1'
QUOTAS=dict(ball_positive_nonfoot_recovery_failure='max_ball_positive_nonfoot_failure_traces',
            ball_positive_other_recovery_failure='max_ball_positive_other_failure_traces')


def diagnostic_retention_settings(value):
    if not isinstance(value,dict) or set(value)!=set(QUOTAS.values()):
        raise ValueError('Declare both finite diagnostic trace quotas')
    if any(type(v) is not int or v<0 for v in value.values()) or sum(value.values())<1:
        raise ValueError('Nonnegative integer quotas with at least one diagnostic slot required')
    return dict(value)


class DiagnosticCurriculumTraceSelection(CurriculumTraceSelection):
    def __init__(self,settings,retention,diagnostic_retention,protocol=None):
        super().__init__(settings,retention,protocol)
        self.diagnostic_retention=diagnostic_retention_settings(diagnostic_retention)
        self.diagnostic_observed=Counter();self.diagnostic_selected=Counter()
        self.diagnostic_retained=Counter();self.diagnostic_skipped=Counter()
        self.diagnostic_saved_candidates={}

    def choose(self,row,*,first_predeclared=False):
        label,kinds,reasons=super().choose(row,first_predeclared=first_predeclared)
        previously_selected=bool(reasons)
        speed=label['outgoing_speed_mps']
        eligible=(kinds['failure'] and speed is not None
            and speed>=self.retention['minimum_speed_mps']
            and all(value for name,value in label['criteria_except_speed'].items()
                    if name!='same_flight_stable_recovery')
            and not label['original_outcomes']['conditional_recovery'])
        if eligible:
            kind=('ball_positive_nonfoot_recovery_failure' if row['contact_flight_recovery']['body_landing']
                  else 'ball_positive_other_recovery_failure')
            self.diagnostic_observed[kind]+=1
            if self.diagnostic_selected[kind]<self.diagnostic_retention[QUOTAS[kind]]:
                self.diagnostic_selected[kind]+=1
                reasons.append(kind+'_until_declared_limit')
            if reasons:
                self.diagnostic_retained[kind]+=1
                self.diagnostic_saved_candidates[(row['env_id'],row['episode_index'])]=kind
            else:
                self.diagnostic_skipped[kind]+=1
        if reasons and not previously_selected:
            # The parent counted this as quota-skipped; the new diagnostic
            # category retains exactly the same raw, still-failed episode.
            self.retained['episodes']+=1
            for kind,value in kinds.items():
                self.retained[kind]+=int(value);self.skipped[kind]-=int(value)
                if self.skipped[kind]<0:raise ValueError('Invalid retention accounting')
        return label,kinds,reasons


class DiagnosticCurriculumEpisodeTraceRecorderV1(CurriculumEpisodeTraceRecorderV1):
    def __init__(self,output,count,max_steps,body_names,settings,provenance,*,retention,diagnostic_retention):
        self.selection=DiagnosticCurriculumTraceSelection(settings,retention,diagnostic_retention)
        self.protocol_sha256=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
        if provenance.get('protocol_sha256')!=self.protocol_sha256:
            raise ValueError('Trace source must use the unchanged original protocol')
        if provenance.get('asset_sha256')!=self.selection.protocol['asset']['sha256']:
            raise ValueError('Trace source belongs to a different loaded asset')
        # Preserve the original buffer shape, lossless packing and close/reset
        # implementation. Only the closed-episode storage decision is extended.
        EpisodeTraceRecorderV4.__init__(self,output,count,max_steps,body_names,settings,provenance)

    def statistics(self):
        result=super().statistics()
        actual=Counter()
        for saved in self.saved:
            kind=self.selection.diagnostic_saved_candidates.get((saved['env_id'],saved['episode_index']))
            if kind is not None:actual[kind]+=1
        result.update(diagnostic_selection_schema=DIAGNOSTIC_SCHEMA,
            diagnostic_retention_settings=self.selection.diagnostic_retention,
            diagnostic_observed=dict(self.selection.diagnostic_observed),
            diagnostic_quota_selected=dict(self.selection.diagnostic_selected),
            diagnostic_selection_retained=dict(self.selection.diagnostic_retained),
            diagnostic_actual_retained=dict(actual),
            diagnostic_quota_skipped=dict(self.selection.diagnostic_skipped),
            failed_diagnostics_are_not_complete_expert_demonstrations=True)
        return result
