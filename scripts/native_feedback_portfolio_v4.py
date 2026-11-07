"""Native measured-state adapter for independently verified feedback seeds.

Uses the existing final-command arbitration and continuous episode trace. Still
a query-conditioned diagnostic, not a general autonomous recovery teacher.
"""
import atexit
import hashlib
import json
from pathlib import Path
from scripts.native_frictional_recovery_v1 import NativeFrictionalRecoveryV1
from scripts.native_terminal_support_v1 import load_verified_policy
from scripts.search_terminal_prefix_arm_margin_v1 import ArmPrefixCandidate
from scripts.arm_speed_reserve_feedback_v1 import ArmSpeedReserveFeedbackV1
from scripts.frictional_feedback_portfolio_v4 import FrictionalFeedbackPortfolioV4


class ObservedPhasePlanner:
    """Supply only the control tick observed by the native dispatch call."""
    def __init__(self, planner):
        self.planner = planner
        self.tick = None

    def solve(self, state, reference, warm):
        if type(self.tick) is not int:
            raise ValueError('No current native control tick')
        tick = self.tick
        self.tick = None
        result = self.planner.solve(state,reference,warm,control_tick=tick)
        result['observed_control_tick'] = tick
        return result

    def close(self):
        self.planner.close()


class NativeFeedbackPortfolioV4(NativeFrictionalRecoveryV1):
    def __init__(self,bundle,case,limits,policy_sha256,output,warm_plan,sphere_audit,validation):
        super().__init__(bundle,case,limits,policy_sha256,output,warm_plan,sphere_audit)
        self.planner.close()
        base, record = load_verified_policy(self.model,self.controller,self.query_public,warm_plan,validation)
        if record['case'] != case or self.start != 460 or self.steps != 340:
            raise ValueError('Different phase-conditioned recovery task')
        original = ArmPrefixCandidate(base,self.controller.qp.interface,-.006)
        reserve = ArmSpeedReserveFeedbackV1(
            ArmPrefixCandidate(base,self.controller.qp.interface,-.006),base,self.controller.qp.interface)
        spheres = json.loads(Path(sphere_audit).read_text())['spheres']
        solver = FrictionalFeedbackPortfolioV4(self.manifest['configuration'],spheres,
            [('original_arm_margin',original),('arm_speed_reserve',reserve)],horizon=80,time_limit_s=60.)
        self.planner = ObservedPhasePlanner(solver)
        atexit.register(self.planner.close)
        self.feedback_validation = str(Path(validation).resolve())
        self.feedback_validation_sha256 = hashlib.sha256(Path(validation).read_bytes()).hexdigest()

    def feedback_target(self,tick,native_state,student,prior):
        self.planner.tick = tick
        try:
            return super().feedback_target(tick,native_state,student,prior)
        finally:
            self.planner.tick = None

    def save(self):
        result = super().save()
        selected = {}
        for plan in self.plans:
            name = plan['decision']['feedback_portfolio']['selected'] or 'shifted_plan'
            selected[name] = selected.get(name,0)+1
        result.update(planner_version='verified_feedback_portfolio_v4',
            feedback_validation=self.feedback_validation,
            feedback_validation_sha256=self.feedback_validation_sha256,
            feedback_candidate_order=['original_arm_margin','arm_speed_reserve'],
            feedback_seed_selection_counts=selected,per_query_soft_budget_s=60.)
        path = self.output/'sequence_native_result.json'; tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');tmp.replace(path)
        return result
