"""Measured-state native recovery adapter for the frictional sequence planner.

Query conditioning is diagnostic. This is not yet a general risk trigger.
Requires an independently model-accepted 80-step warm plan, and revalidates it
at the real observation before applying a command. Failure is never a label.
"""
import atexit
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pinocchio as pin
from scripts.native_receding_recovery_v1 import NativeRecedingRecoveryV1
from scripts.frictional_receding_recovery_v1 import FrictionalRecedingRecoveryV1


def shifted_frictional_seed(commands, executed, horizon, previous):
    commands=np.asarray(commands,float)
    if (commands.ndim!=2 or commands.shape[1]!=18 or not np.isfinite(commands).all()
            or type(executed) is not int or not 0<=executed<=len(commands) or horizon!=80):
        raise ValueError('Actual executed count and complete horizon required')
    remaining=commands[executed:]
    tail=remaining[-1] if len(remaining) else np.asarray(previous,float)
    return np.vstack([remaining,np.repeat(tail[None],horizon,axis=0)])[:horizon].copy()


class NativeFrictionalRecoveryV1(NativeRecedingRecoveryV1):
    def __init__(self,bundle,case,limits,policy_sha256,output,warm_plan,sphere_audit):
        super().__init__(bundle,case,limits,policy_sha256,output)
        warm_path=Path(warm_plan);sphere_path=Path(sphere_audit)
        warm=json.loads(warm_path.read_text());sphere=json.loads(sphere_path.read_text())
        if warm['case']!=case or not warm['final']['accepted'] or warm.get('horizon_steps')!=80:
            raise ValueError('Require accepted same-query 0.8 second frictional warm plan')
        for record in [warm,sphere]:
            for name,digest in record['source_sha256'].items():
                if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:
                    raise ValueError('Changed declared warm plan or geometry input '+name)
        if str((Path(bundle)/'manifest.json').resolve()) not in warm['source_sha256']:
            raise ValueError('Warm plan came from another query bundle')
        self.warm_commands=np.asarray(warm['final']['commands'],float)
        if self.warm_commands.shape!=(80,18):raise ValueError('Incomplete warm command sequence')
        self.planner=FrictionalRecedingRecoveryV1(self.manifest['configuration'],sphere['spheres'],horizon=80)
        atexit.register(self.planner.close)
        self.warm_plan_sha256=hashlib.sha256(warm_path.read_bytes()).hexdigest()
        self.sphere_audit_sha256=hashlib.sha256(sphere_path.read_bytes()).hexdigest()
        self.predicted_states=None
        self.executed_prediction_errors=[]

    def feedback_target(self,tick,native_state,student,prior):
        state=copy.deepcopy(native_state);state.previous_command_target=prior.copy()
        reference=self.decode(student)
        elapsed=None if self.plan_start is None else tick-self.plan_start
        if self.predicted_states is not None and elapsed is not None and 0<elapsed<=len(self.predicted_states):
            prediction=self.predicted_states[elapsed-1]
            dq=pin.difference(self.model.model,np.asarray(prediction['q']),state.q)
            self.executed_prediction_errors.append(dict(tick=tick,plan_tick=self.plan_start,offset=elapsed,
                joint_position_error_rad=float(np.max(abs(dq[self.model.v_indices]))),
                root_position_error_m=float(np.linalg.norm(state.q[:3]-np.asarray(prediction['q'])[:3])),
                attitude_error_rad=float(np.linalg.norm(dq[3:6])),
                velocity_error_max=float(np.max(abs(state.v-np.asarray(prediction['v']))))))
        changed=self.plan_contacts is not None and not np.array_equal(state.contacts,self.plan_contacts)
        if not self.planning_failed and (self.plan is None or elapsed>=10 or changed):
            warm=shifted_frictional_seed(self.warm_commands,0,80,prior) if self.plan is None else shifted_frictional_seed(self.plan,elapsed,80,prior)
            decision=self.planner.solve(state,reference,warm)
            self.plans.append(dict(tick=tick,contact_changed=bool(changed),observed_q=state.q.copy(),
                observed_v=state.v.copy(),observed_contacts=state.contacts.copy(),
                measured_foot_force_world_n=state.measured_force_world.copy(),actual_previous_target_rad=prior.copy(),decision=decision))
            progress=dict(control_tick=tick,replans=len(self.plans),
                accepted_replans=sum(bool(p['decision']['accepted']) for p in self.plans),
                latest_accepted=bool(decision['accepted']),latest_wall_seconds=decision['wall_seconds'],
                latest_reason=decision.get('reason'),teacher_steps=self.teacher_steps,
                fallback_steps=self.fallback_steps,teacher_model='frictional_spheres_v1',
                native_recovery_success_not_yet_known=True)
            path=self.output/'frictional_feedback_progress.json';tmp=path.with_suffix('.tmp')
            tmp.write_text(json.dumps(progress,indent=2)+'\n');tmp.replace(path)
            if decision['accepted']:
                self.plan=decision['commands'].copy();self.plan_start=tick;elapsed=0
                self.plan_contacts=state.contacts.copy();self.predicted_states=decision['predicted_states']
            else:
                self.planning_failed=True;self.predicted_states=None;self.planner.close()
        if self.planning_failed:
            target=reference.copy();target[12:]=np.clip(target[12:],prior[12:]-self.arm_rate*self.dt,prior[12:]+self.arm_rate*self.dt)
            self.fallback_steps+=1
            return target,'planning_failed_student_fallback_not_certified'
        self.teacher_steps+=1
        return self.plan[elapsed].copy(),'frictional_receding_support_sequence'

    def save(self):
        try:
            result=super().save()
            result.update(teacher_model='frictional_spheres_v1',horizon_steps=80,
                warm_plan_sha256=self.warm_plan_sha256,sphere_audit_sha256=self.sphere_audit_sha256,
                executed_feedback_prediction_errors=self.executed_prediction_errors,
                per_call_hard_deadline=False,external_process_guard_required=True,
                legacy_nominal_error_reference='Original old-model plan retained only for provenance; use executed_feedback_prediction_errors for new plans')
            path=self.output/'sequence_native_result.json';tmp=path.with_suffix('.tmp')
            tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');tmp.replace(path)
            return result
        finally:
            self.planner.close()
