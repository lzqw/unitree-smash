"""Measured-state execution of an offline optimized complete landing reference.

This validates a potential MPC warm start. It is not online MPC or a deployable
risk trigger. Nominal commands remain immutable; each actual command is
recomputed with the original state feedback and actual dispatch history.
"""
import copy
import json
import numpy as np
from scripts.native_recovery_sequence_replay_v2 import NativeRecoverySequenceReplayV2
from scripts.parallel_landing_solver_v1 import build_controller
from scripts.loaded_pinocchio_model_v1 import PublicState
from scripts.search_landing_support_tail_v3 import SupportTailReference
from scripts.terminal_attitude_feedback_v1 import TerminalAttitudeFeedbackV1
import pinocchio as pin


class NativeEarlySupportTailV1(NativeRecoverySequenceReplayV2):
    def __init__(self, bundle, case, limits, policy_sha256, output):
        super().__init__(bundle, case, limits, policy_sha256, output)
        if self.case!=3 or self.start!=280 or self.steps!=520:
            raise ValueError('Declared full early recovery test required')
        self.controller=build_controller(self.manifest['configuration'])
        self.model=self.controller.qp.model
        q=self.expected_q[0].copy(); v=self.expected_v[0].copy()
        initial=PublicState(q,v,pin.Quaternion(q[3:7]).matrix(),q[:3].copy(),
            q[self.model.q_indices].copy(),v[self.model.v_indices].copy(),self.expected_contacts[0].copy(),0.)
        initial.previous_command_target=self.decode(self.query_previous_action)
        cfg=self.manifest['feedback_definition']
        base=SupportTailReference(self.model,self.controller.qp.interface,initial,
            cfg['tail_knots_rad'],fixed_knots=cfg['fixed_early_knots_rad'])
        self.feedback=TerminalAttitudeFeedbackV1(base,proportional=cfg['proportional'],
            damping_s=cfg['damping_s'],maximum_joint_delta_rad=cfg['maximum_joint_delta_rad'])
        self.feedback_records=[]
        self.early_termination_tick=None

    def action(self,tick,public,student,finished,native_state=None):
        offset=tick-self.start
        if tick>=self.start and bool(finished[self.case]):
            if tick!=self.last_tick+1:raise ValueError('Continuous actual clock required')
            self.last_tick=tick
            if self.applied<self.steps and self.early_termination_tick is None:
                self.early_termination_tick=tick
            # A genuine failed/terminated episode is evidence, not a solver
            # exception. Never apply the old teacher to its reset episode.
            return student.clone()
        if not 0<=offset<self.steps:
            return super().action(tick,public,student,finished,native_state)
        if native_state is None or self.previous is None:
            raise ValueError('Actual state and command history required')
        state=copy.deepcopy(native_state)
        state.previous_command_target=self.previous.copy()
        target,mode=self.feedback.action(80+offset,state,self.previous.copy())
        if (target.shape!=(18,) or not np.isfinite(target).all() or
                np.any(target<self.limits[:,0]-1e-7) or np.any(target>self.limits[:,1]+1e-7) or
                np.max(abs(target[12:]-self.previous[12:]))>.100001):
            raise ValueError('Feedback command violates original interface')
        # Reuse the exact final arbitration, float32 dispatch, query admission
        # and logging. Restore the immutable nominal row even on failure.
        nominal=self.commands[offset].copy()
        self.commands[offset]=target
        try:
            result=super().action(tick,public,student,finished,native_state)
        finally:
            self.commands[offset]=nominal
        self.records[-1]['mode']='measured_state_early_support_feedback'
        self.feedback_records.append(dict(control_tick=tick,mode=mode,
            nominal_target_difference_rad=float(np.max(abs(target-nominal))),
            feedback=copy.deepcopy(self.feedback.last_diagnostic)))
        return result

    def save(self):
        if len(self.feedback_records)!=self.applied:raise ValueError('Feedback coverage mismatch')
        if self.early_termination_tick is None:
            result=super().save()
        else:
            shapes=dict(q=(25,),v=(24,),contacts=(4,),foot_force_world_n=(4,3),
                student_action=(18,),final_action=(18,),previous_target_rad=(18,),
                requested_target_rad=(18,),final_target_rad=(18,))
            np.savez_compressed(self.output/'sequence_native_trace.npz',
                **{k:np.stack([r[k] for r in self.records]) if self.records else np.empty((0,)+shape)
                   for k,shape in shapes.items()})
            result=dict(case=self.case,start_tick=self.start,planned_steps=self.steps,applied_steps=self.applied,
                prediction_endpoint_observed=False,query_match=self.query_check or {'passed':False},
                records=[{k:v for k,v in row.items() if k not in shapes} for row in self.records],
                bundle_sha256=self.manifest['bundle_sha256'],native_execution=True,
                case_time_conditioned_diagnostic=True,deployable_risk_trigger=False,
                post_sequence='Selected first episode terminated; teacher disabled for its reset episode.',
                sustained_mpc_feedback_implemented=False,valid_expert_labels=False)
        result.update(controller_kind='offline_optimized_reference_with_measured_state_feedback',
            measured_state_feedback=True, online_mpc=False, teacher_steps=self.applied,
            early_termination_tick=self.early_termination_tick,
            full_feedback_sequence_executed=self.applied==self.steps,
            feedback_records=self.feedback_records, valid_expert_labels=False)
        path=self.output/'sequence_native_result.json';temporary=path.with_suffix('.tmp')
        temporary.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');temporary.replace(path)
        return result
