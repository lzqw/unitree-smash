"""Native execution candidate for the paired-model reference tracking layer.

Inherits the original query, actual-command mapping, finite episode lifecycle
and negative-result handling. This does not turn reference feedback into MPC.
"""
import copy
import json
from scripts.native_early_support_tail_v1 import NativeEarlySupportTailV1
from scripts.landing_reference_tracking_v1 import LandingReferenceTrackingV1


class NativeReferenceTrackingV1(NativeEarlySupportTailV1):
    def __init__(self,bundle,case,limits,policy_sha256,output):
        super().__init__(bundle,case,limits,policy_sha256,output)
        definition=self.manifest.get('reference_tracking_definition')
        if not isinstance(definition,dict):raise ValueError('Pinned tracking definition required')
        self.reference_tracking=LandingReferenceTrackingV1(self.feedback.base,
            self.expected_q[:-1],self.expected_v[:-1],**definition['gains'])
        # Exactly the order evaluated in the fine-model paired experiment:
        # full-body reference -> tracking -> original attitude feedback.
        self.feedback.base=self.reference_tracking

    def action(self,tick,public,student,finished,native_state=None):
        before=len(self.feedback_records)
        result=super().action(tick,public,student,finished,native_state)
        if len(self.feedback_records)>before:
            self.feedback_records[-1]['reference_tracking']=copy.deepcopy(self.reference_tracking.last_diagnostic)
            self.records[-1]['mode']='measured_state_early_support_reference_tracking'
        return result

    def save(self):
        result=super().save()
        result.update(controller_kind='offline_optimized_reference_with_measured_tracking_and_attitude_feedback',
            reference_tracking_definition=self.manifest['reference_tracking_definition'],online_mpc=False)
        path=self.output/'sequence_native_result.json';tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');tmp.replace(path)
        return result
