"""State-verified feedback candidates for the original constrained MPC.

Candidate ordering is fixed, never selected by case ID or future native outcome.
No acceptance threshold or contact/actuator model is changed.
"""
import copy
import time
from pathlib import Path
import numpy as np
from scripts.frictional_receding_recovery_v2 import FrictionalRecedingRecoveryV2
from scripts.predict_frictional_spheres_v1 import predict_frictional_spheres
from scripts import search_frictional_recovery_cem_v1 as core
from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance


class FrictionalFeedbackPortfolioV4(FrictionalRecedingRecoveryV2):
    def __init__(self, configuration, spheres, feedback_policies, **kwargs):
        super().__init__(configuration, spheres, **kwargs)
        if not 1 <= len(feedback_policies) <= 3:
            raise ValueError('Bounded ordered feedback candidate set required')
        self.feedback_policies = list(feedback_policies)
        self.spheres = spheres
        self.geometry = AuthoredGroundClearance(
            Path(__file__).resolve().parents[1]/'configs/go2_piper_authored_ground_geometry_v1.json',
            self.controller.qp.model, margin_m=0.)

    def solve(self, observed_state, student_reference, warm_commands, *, control_tick):
        if type(control_tick) is not int or not 460 <= control_tick < 800:
            raise ValueError('Original observed phase required')
        started = time.monotonic(); attempts = []; selected = None; chosen = warm_commands
        for name, policy in self.feedback_policies:
            if time.monotonic()-started >= self.time_limit_s: break
            state = copy.deepcopy(observed_state); commands = []; failure = None
            for step in range(80):
                if time.monotonic()-started >= self.time_limit_s:
                    failure = dict(step=step, reason='feedback_generation_budget'); break
                target, _ = policy.action(min(control_tick+step-460,339),state,state.previous_command_target)
                try:
                    item = predict_frictional_spheres(self.controller.qp.model,self.controller.qp.interface,
                        state,target,self.spheres,geometry=self.geometry)
                except (ValueError,np.linalg.LinAlgError) as error:
                    failure = dict(step=step,error=repr(error)); break
                bad = {k:v for k,v in item['diagnostics'].items() if
                    (k=='nonfoot_ground_violation_m' and v>1e-6) or
                    (k in ['joint_position_violation_rad','joint_velocity_violation_rad_s','motor_torque_violation_nm'] and v>3e-4)}
                if bad: failure = dict(step=step,violations=bad); break
                state=item['state']; state.previous_command_target=target.copy(); commands.append(target.copy())
            verified = None
            if failure is None:
                seed = np.asarray(commands)
                core.CONTEXT = (self.controller,copy.deepcopy(observed_state),
                                observed_state.previous_command_target.copy(),seed,self.spheres,self.geometry)
                verified = core.evaluate(np.zeros_like(seed),save_states=False)
            attempts.append(dict(name=name,generated_steps=len(commands),generation_failure=failure,verification=verified))
            if verified is not None and verified['accepted']:
                chosen = np.asarray(commands); selected = name; break
        old_limit = self.time_limit_s
        self.time_limit_s = max(1.,old_limit-(time.monotonic()-started))
        try: decision = super().solve(observed_state,student_reference,chosen)
        finally: self.time_limit_s = old_limit
        decision.update(planner_version='verified_feedback_portfolio_v4',
                        feedback_portfolio=dict(attempts=attempts,selected=selected,
                                                original_shifted_seed_used=selected is None),
                        wall_seconds=time.monotonic()-started)
        return decision
