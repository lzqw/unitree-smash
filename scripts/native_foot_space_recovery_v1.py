"""Native feedback execution of an independently verified foot-space recovery.

The case/time query is an experiment selector, not a deployable risk trigger.
Original full-panel execution, limits, actual command history and strict query
admission are inherited. No predicted state replaces native feedback.
"""
import numpy as np
import pinocchio as pin

from scripts.native_recovery_sequence_replay_v2 import NativeRecoverySequenceReplayV2
from scripts.native_early_support_tail_v1 import NativeEarlySupportTailV1
from scripts.parallel_landing_solver_v1 import build_controller
from scripts.loaded_pinocchio_model_v1 import PublicState
from scripts.search_preserved_early_support_v1 import joint_factory
from scripts.terminal_attitude_feedback_v1 import TerminalAttitudeFeedbackV1
from scripts.search_braked_support_transition_v1 import SupportTransitionV1
from scripts.landing_joint_braking_reference_v1 import LandingJointBrakingReferenceV1


class LoggedFeedback:
    def __init__(self, policy):
        self.policy=policy;self.last_diagnostic=None

    def action(self,offset,state,previous):
        target,mode=self.policy.action(offset,state,previous)
        self.last_diagnostic=dict(original_reference_offset=offset,
            actual_contacts=state.contacts.tolist(),mode=mode)
        return target,mode


class NativeFootSpaceRecoveryV1(NativeEarlySupportTailV1):
    def __init__(self,bundle,case,limits,policy_sha256,output):
        NativeRecoverySequenceReplayV2.__init__(self,bundle,case,limits,policy_sha256,output)
        if self.start!=280 or self.steps!=520:
            raise ValueError('Full original post-release remainder required')
        cfg=self.manifest['foot_space_definition']
        if cfg['contract']!='verified_foot_space_transition_v1' or not cfg['independent_replay_exact']:
            raise ValueError('Verified model feedback definition required')
        self.controller=build_controller(self.manifest['configuration']);self.model=self.controller.qp.model
        q=self.expected_q[0].copy();v=self.expected_v[0].copy()
        initial=PublicState(q,v,pin.Quaternion(q[3:7]).matrix(),q[:3].copy(),
            q[self.model.q_indices].copy(),v[self.model.v_indices].copy(),self.expected_contacts[0].copy(),0.)
        initial.previous_command_target=self.decode(self.query_previous_action)
        with np.load(self.bundle/self.manifest['bundle_name'],allow_pickle=False) as z:
            rq=z['tracking_reference_q'].copy();rv=z['tracking_reference_v'].copy()
        base=joint_factory(self.model,self.controller.qp.interface,initial,np.asarray(cfg['original_knots']),
            reference_q=rq,reference_v=rv,gains=cfg['tracking_gains'])
        tracking=TerminalAttitudeFeedbackV1(base,proportional=.3,damping_s=.04,maximum_joint_delta_rad=.08)
        transition=SupportTransitionV1(tracking,cfg['transition_knots'])
        self.feedback=LoggedFeedback(LandingJointBrakingReferenceV1(transition,**cfg['braking']))
        self.feedback_records=[];self.early_termination_tick=None
