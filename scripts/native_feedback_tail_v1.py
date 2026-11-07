"""Native MPC adapter with explicit, independently checked feedback continuations.

Original measured-state queries, final-command arbitration, replanning triggers,
and unverified fallback labels are preserved. This is a case-conditioned test.
"""
import atexit
import hashlib
import json
from pathlib import Path

from scripts.native_frictional_recovery_v1 import NativeFrictionalRecoveryV1
from scripts.native_feedback_portfolio_v4 import ObservedPhasePlanner
from scripts.native_terminal_attitude_v1 import load_verified_policy
from scripts.frictional_feedback_tail_planner_v1 import FrictionalFeedbackTailPlannerV1


class NativeFeedbackTailV1(NativeFrictionalRecoveryV1):
    def __init__(self, bundle, case, limits, policy_sha256, output, warm_plan, sphere_audit, validation):
        super().__init__(bundle, case, limits, policy_sha256, output, warm_plan, sphere_audit)
        self.planner.close()
        if case != 3 or self.start != 460 or self.steps != 340:
            raise ValueError('Declared original ground-start diagnostic required')
        policy, _ = load_verified_policy(self.controller.qp.model, self.controller,
                                        self.query_public, warm_plan, validation)
        root = Path(__file__).resolve().parents[1]
        protocol = json.loads((root/'configs/evaluation_protocol_v2.json').read_text())['recovery']
        spheres = json.loads(Path(sphere_audit).read_text())['spheres']
        solver = FrictionalFeedbackTailPlannerV1(self.controller, policy, spheres,
            self.controller.sequence_geometry, protocol, time_limit_s=300.)
        self.planner = ObservedPhasePlanner(solver)
        atexit.register(self.planner.close)
        self.feedback_validation = str(Path(validation).resolve())
        self.feedback_validation_sha256 = hashlib.sha256(Path(validation).read_bytes()).hexdigest()

    def feedback_target(self, tick, native_state, student, prior):
        self.planner.tick = tick
        try:
            return super().feedback_target(tick, native_state, student, prior)
        finally:
            self.planner.tick = None

    def save(self):
        result = super().save()
        result.update(planner_version='feedback_tail_candidate_selection_v1',
            feedback_validation=self.feedback_validation,
            feedback_validation_sha256=self.feedback_validation_sha256,
            acceptance_rule='constrained_prefix_plus_explicit_stable_feedback_continuation',
            per_query_soft_budget_s=300., recursive_feasibility_proven=False,
            task_thresholds_changed=False)
        path = self.output/'sequence_native_result.json'; temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        temporary.replace(path)
        return result
