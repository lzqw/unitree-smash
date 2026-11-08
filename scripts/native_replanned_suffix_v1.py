"""Ground-start replay of a preserved prefix followed by a query-matched plan.

The suffix was optimized offline from the recorded actual state. Native query
agreement is required before dispatch; this is not an online MPC claim.
"""
import copy
import json
from pathlib import Path

import numpy as np
import pinocchio as pin

from scripts.native_early_support_tail_v1 import NativeEarlySupportTailV1
from scripts.loaded_pinocchio_model_v1 import PublicState
from scripts.search_landing_support_tail_v3 import SupportTailReference
from scripts.landing_reference_tracking_v1 import LandingReferenceTrackingV1
from scripts.terminal_attitude_feedback_v1 import TerminalAttitudeFeedbackV1
from scripts.solve_recorded_recovery_suffix_v1 import SuffixCommandCorrectionV1


class NativeReplannedSuffixV1(NativeEarlySupportTailV1):
    def __init__(self,bundle,case,limits,policy_sha256,output):
        super().__init__(bundle,case,limits,policy_sha256,output)
        definition=self.manifest['replanned_suffix_definition']
        if definition['contract']!='preserved_native_prefix_actual_query_suffix_v1':
            raise ValueError('Explicit prefix/suffix contract required')
        self.suffix_start=definition['control_tick'];self.suffix_query=None
        if not 320<=self.suffix_start<=600:raise ValueError('Declared suffix query clock required')
        with np.load(Path(bundle)/self.manifest['bundle_name'],allow_pickle=False) as z:
            self.suffix_q=z['suffix_query_q'].copy();self.suffix_v=z['suffix_query_v'].copy()
            self.suffix_contacts=z['suffix_query_contacts'].copy();self.suffix_previous=z['suffix_query_previous'].copy()
            rq=z['suffix_tracking_reference_q'].copy();rv=z['suffix_tracking_reference_v'].copy()
            origin_q=z['suffix_origin_q'].copy();origin_v=z['suffix_origin_v'].copy()
            origin_contact=z['suffix_origin_contacts'].copy();origin_previous=z['suffix_origin_previous'].copy()
        origin=PublicState(origin_q,origin_v,pin.Quaternion(origin_q[3:7]).matrix(),origin_q[:3].copy(),
            origin_q[self.model.q_indices].copy(),origin_v[self.model.v_indices].copy(),origin_contact,0.)
        origin.previous_command_target=origin_previous
        knots=np.asarray(definition['original_knots_rad'])
        base=SupportTailReference(self.model,self.controller.qp.interface,origin,knots[3:],fixed_knots=knots[:3])
        tracking=LandingReferenceTrackingV1(base,rq,rv,**definition['tracking_gains'])
        feedback=TerminalAttitudeFeedbackV1(tracking,proportional=.3,damping_s=.04,maximum_joint_delta_rad=.08)
        self.suffix_feedback=SuffixCommandCorrectionV1(feedback,self.suffix_start,self.suffix_previous,
            definition['corrections_rad'])
        self.suffix_feedback.last_diagnostic=None
        self.prefix_feedback=self.feedback

    def action(self,tick,public,student,finished,native_state=None):
        if tick>=self.suffix_start and not bool(finished[self.case]):
            if tick==self.suffix_start:
                if native_state is None or self.previous is None:raise ValueError('Actual suffix query missing')
                error=dict(configuration_tangent=float(np.max(abs(pin.difference(self.model.model,self.suffix_q,native_state.q)))),
                    velocity=float(np.max(abs(native_state.v-self.suffix_v))),
                    previous_target_rad=float(np.max(abs(self.previous-self.suffix_previous))))
                tolerances=dict(configuration_tangent=2e-5,velocity=2e-4,previous_target_rad=5e-6)
                contacts_equal=bool(np.array_equal(native_state.contacts,self.suffix_contacts))
                passed=contacts_equal and all(error[k]<=v for k,v in tolerances.items())
                self.suffix_query=dict(control_tick=tick,passed=passed,errors=error,
                    tolerances=tolerances,contacts_equal=contacts_equal)
                (self.output/'suffix_query_match.json').write_text(json.dumps(self.suffix_query,indent=2)+'\n')
                if not passed:raise ValueError('Native suffix query differs; new candidate not dispatched')
                # Float32 command round-trip is admitted by the explicit query
                # tolerance, then the actual history anchors the first request.
                self.suffix_feedback.previous_at_start=self.previous.copy()
            if not self.suffix_query or not self.suffix_query['passed']:
                raise ValueError('Suffix cannot run before query admission')
            self.feedback=self.suffix_feedback
            self.feedback.last_diagnostic=dict(kind='query_matched_full_suffix',query_tick=self.suffix_start,
                original_reference_tick=tick,online_mpc=False)
        before=len(self.feedback_records)
        result=super().action(tick,public,student,finished,native_state)
        if len(self.feedback_records)>before:
            self.feedback_records[-1]['segment']='preserved_prefix' if tick<self.suffix_start else 'replanned_suffix'
            self.records[-1]['mode']=self.feedback_records[-1]['segment']
        return result

    def save(self):
        result=super().save()
        result.update(controller_kind='preserved_prefix_then_offline_actual_state_suffix',
            suffix_query_match=self.suffix_query,suffix_start_tick=self.suffix_start,
            prediction_segments='Original prediction before suffix query; new prediction anchored at actual suffix query.',
            online_mpc=False,valid_expert_labels=False)
        path=self.output/'sequence_native_result.json';tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');tmp.replace(path)
        return result
