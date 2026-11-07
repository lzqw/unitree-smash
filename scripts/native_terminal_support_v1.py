"""Native diagnostic of a planned recovery prefix and continuous terminal feedback.

This variant does not replan online; it tests the proposed terminal controller
for later integration into receding-horizon MPC. No early exit or success label.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.native_frictional_recovery_v1 import NativeFrictionalRecoveryV1
from scripts.native_recovery_sequence_replay_v2 import NativeRecoverySequenceReplayV2
from scripts.terminal_support_policy_v1 import TerminalSupportPolicyV1
from scripts.extend_frictional_recovery_v1 import set_state


def load_verified_policy(model, controller, query_public, warm_path, validation_path):
    validation_path=Path(validation_path);warm_path=Path(warm_path)
    validation=json.loads(validation_path.read_text());warm=json.loads(warm_path.read_text())
    if (not validation['model_candidate_accepted'] or validation['failure'] is not None or
            validation['executed_model_steps']!=340 or validation['start_tick']!=460 or
            validation['native_continuation_executed'] or validation['valid_expert_labels']):
        raise ValueError('Require complete model-only feedback validation')
    for name,digest in validation['source_sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Changed feedback validation input '+name)
    if validation['source_sha256'].get(str(warm_path.resolve()))!=hashlib.sha256(warm_path.read_bytes()).hexdigest():
        raise ValueError('Different nominal prefix')
    template=model.state_from_public(query_public,np.zeros(18))
    initial=set_state(template,model,validation['initial_q'],validation['initial_v'],
        validation['initial_contacts'],validation['actual_previous_command'])
    tail=dict(validation['selected_tail'],failure=None,valid_steps=260,model_window=validation['model_window'])
    return TerminalSupportPolicyV1(model,controller.qp.interface,initial,warm,tail),validation


class NativeTerminalSupportV1(NativeFrictionalRecoveryV1):
    def __init__(self,bundle,case,limits,policy_sha256,output,warm_plan,sphere_audit,validation_path):
        super().__init__(bundle,case,limits,policy_sha256,output,warm_plan,sphere_audit)
        self.terminal_policy,record=load_verified_policy(self.model,self.controller,self.query_public,warm_plan,validation_path)
        if record['case']!=case or self.start!=460 or self.steps!=340:
            raise ValueError('Different native case or episode interval')
        self.validation_path=str(Path(validation_path).resolve())
        self.validation_sha256=hashlib.sha256(Path(validation_path).read_bytes()).hexdigest()
        self.expected_q=np.vstack([record['initial_q'],[s['q'] for s in record['states']]])
        self.expected_v=np.vstack([record['initial_v'],[s['v'] for s in record['states']]])
        self.expected_contacts=np.vstack([record['initial_contacts'],[s['contacts'] for s in record['states']]])

    def feedback_target(self,tick,native_state,student,prior):
        target,mode=self.terminal_policy.action(tick-self.start,native_state,prior)
        self.teacher_steps+=1
        return target,mode

    def save(self):
        try:
            result=NativeRecoverySequenceReplayV2.save(self)
            result.update(teacher_model='planned_prefix_terminal_feedback_v1',
                sustained_mpc_feedback_implemented=False,sustained_terminal_feedback_implemented=True,
                online_replanning=False,teacher_steps=self.teacher_steps,
                unverified_student_fallback_steps=0,validation_path=self.validation_path,
                validation_sha256=self.validation_sha256,warm_plan_sha256=self.warm_plan_sha256,
                sphere_audit_sha256=self.sphere_audit_sha256,
                prediction_error_reference='Complete measured-query model trajectory with the same feedback policy')
            path=self.output/'sequence_native_result.json';tmp=path.with_suffix('.tmp')
            tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');tmp.replace(path)
            return result
        finally:self.planner.close()
