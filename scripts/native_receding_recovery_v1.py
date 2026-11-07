"""Query-matched native feedback diagnostic, not a deployable risk trigger.

Run the original student to a recorded query, then revalidate/replan a complete
support sequence every 100 ms or immediately on a contact-mode change. Failed
planning latches an explicitly unverified student fallback for this episode.
The surrounding native job must enforce an external solver wall-clock guard.
"""
import copy
import inspect
import json
from pathlib import Path
import textwrap
import numpy as np
import scripts.native_recovery_sequence_replay_v2 as replay
from scripts.parallel_landing_solver_v1 import build_controller
from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance
from scripts.receding_recovery_sequence_v1 import RecedingRecoverySequenceV1,shifted_seed
from scripts.audit_joint_sequence_proposal_v1 import plain


class NativeRecedingRecoveryV1(replay.NativeRecoverySequenceReplayV2):
    def __init__(self,bundle,case,limits,policy_sha256,output):
        super().__init__(bundle,case,limits,policy_sha256,output)
        self.controller=build_controller(self.manifest['configuration'])
        root=Path(__file__).resolve().parents[1]
        geometry=root/'configs/go2_piper_authored_ground_geometry_v1.json'
        if replay.hashlib.sha256(geometry.read_bytes()).hexdigest()!=self.manifest['ground_geometry_sha256']:
            raise ValueError('Different authored geometry')
        self.controller.sequence_geometry=AuthoredGroundClearance(geometry,self.controller.qp.model,margin_m=0.)
        self.planner=RecedingRecoverySequenceV1(self.controller)
        self.plan=None;self.plan_start=None;self.plan_contacts=None
        self.planning_failed=False;self.plans=[];self.teacher_steps=0;self.fallback_steps=0

    def feedback_target(self,tick,native_state,student,prior):
        state=copy.deepcopy(native_state);state.previous_command_target=prior.copy()
        reference=self.decode(student)
        elapsed=None if self.plan_start is None else tick-self.plan_start
        changed=self.plan_contacts is not None and not np.array_equal(state.contacts,self.plan_contacts)
        if not self.planning_failed and (self.plan is None or elapsed>=10 or changed):
            if self.plan is None:
                warm=shifted_seed(self.commands,tick-self.start,40,prior,self.controller.qp.interface,self.dt)
            else:
                warm=shifted_seed(self.plan,elapsed,40,prior,self.controller.qp.interface,self.dt)
            decision=self.planner.solve(state,reference,warm)
            self.plans.append(dict(tick=tick,contact_changed=bool(changed),observed_q=state.q.copy(),
                observed_v=state.v.copy(),observed_contacts=state.contacts.copy(),
                measured_foot_force_world_n=state.measured_force_world.copy(),
                actual_previous_target_rad=prior.copy(),decision=decision))
            if decision['accepted']:
                self.plan=decision['commands'].copy();self.plan_start=tick;elapsed=0
                self.plan_contacts=state.contacts.copy()
            else:self.planning_failed=True
        if self.planning_failed:
            target=reference.copy();target[12:]=np.clip(target[12:],prior[12:]-self.arm_rate*self.dt,
                prior[12:]+self.arm_rate*self.dt)
            self.fallback_steps+=1
            return target,'planning_failed_student_fallback_not_certified'
        self.teacher_steps+=1
        return self.plan[elapsed].copy(),'receding_support_sequence'

    def save(self):
        result=super().save()
        result.update(sustained_mpc_feedback_implemented=True,native_recovery_verified=False,
            prediction_error_reference='original offline nominal trajectory, not latest replanned trajectory',
            replan_period_control_ticks=10,immediate_replan_on_contact_change=True,
            teacher_steps=self.teacher_steps,unverified_student_fallback_steps=self.fallback_steps,
            planning_failed=self.planning_failed,planning_records=plain(self.plans),
            per_call_hard_deadline=False,external_process_guard_required=True)
        path=self.output/'sequence_native_result.json';tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');tmp.replace(path)
        return result


def feedback_action():
    # Keep exactly the pinned query validation, continuous tick checks, actual
    # float32 command admission and native telemetry of the existing adapter.
    source=textwrap.dedent(inspect.getsource(replay.NativeRecoverySequenceReplayV2.action))
    old="target=self.commands[offset].copy();mode='planned_sequence';self.applied+=1"
    if source.count(old)!=1:raise ValueError('Changed pinned command selection contract')
    source=source.replace(old,"target,mode=self.feedback_target(tick,native_state,base,prior);self.applied+=1")
    namespace=dict(vars(replay));exec(compile(source,'<native-receding-recovery>','exec'),namespace)
    return namespace['action']


NativeRecedingRecoveryV1.action=feedback_action()
