"""Bounded sequence candidate selection with independently verified feedback tails.

This changes the controller's terminal construction, not task thresholds:
80 proposed commands must lead to an explicitly simulated stable continuation.
Finite candidates do not establish global optimality or recursive feasibility.
"""
import copy
import time
import numpy as np

from scripts.feedback_tail_rollout_v1 import evaluate_feedback_tail
from scripts.frictional_receding_recovery_v1 import admit_commands


class FrictionalFeedbackTailPlannerV1:
    def __init__(self, controller, feedback, spheres, geometry, protocol, *, time_limit_s=300.):
        if not 120 <= time_limit_s <= 600:
            raise ValueError('Explicit finite budget allowing fine verification required')
        if feedback.model is not controller.qp.model:
            raise ValueError('Feedback and rollout must use the same loaded model')
        self.controller = controller; self.feedback = feedback
        self.spheres = spheres; self.geometry = geometry; self.protocol = copy.deepcopy(protocol)
        self.time_limit_s = time_limit_s

    def close(self):
        pass

    def solve(self, observed_state, student_reference, warm_commands, *, control_tick):
        began = time.monotonic(); state = copy.deepcopy(observed_state)
        reference = np.asarray(student_reference, float)
        if reference.shape != (18,) or not np.isfinite(reference).all():
            raise ValueError('Actual student absolute target required')
        warm = admit_commands(warm_commands, state.previous_command_target, self.controller.qp.interface)
        if warm.shape != (80, 18):
            raise ValueError('Full shifted short horizon required')
        attempts = []; candidates = []
        def compact(result):
            return {k: v for k, v in result.items() if k not in ['commands', 'predicted_states', 'records']}
        def run(prefix, substeps):
            remaining = self.time_limit_s-(time.monotonic()-began)
            if remaining <= 0:
                return None
            return evaluate_feedback_tail(self.controller, state, self.feedback, self.spheres,
                self.geometry, self.protocol, control_tick=control_tick,
                prefix_commands=prefix, substeps=substeps, time_limit_s=remaining)
        for name, prefix in [('shifted_previous_plan', warm), ('state_feedback_seed', None)]:
            coarse = run(prefix, 16)
            if coarse is None:
                break
            attempts.append(dict(name=name, stage='coarse', verification=compact(coarse)))
            if coarse['accepted']:
                commands = coarse['commands'][:80].copy()
                # Penalize modifying the current student's first command;
                # future student actions are not available as a causal input.
                score = coarse['cost']+.1*float(np.sum((commands[0]-reference)**2))
                candidates.append((score, name, commands))
        for score, name, commands in sorted(candidates, key=lambda row: row[0]):
            fine = run(commands, 160)
            if fine is None:
                break
            attempts.append(dict(name=name, stage='independent_fixed_prefix_fine', verification=compact(fine)))
            if not fine['accepted']:
                continue
            if not np.array_equal(fine['commands'][:80], commands):
                raise ValueError('Verification modified the executable prefix')
            return dict(accepted=True, commands=commands, predicted_states=fine['predicted_states'][:80],
                verification=compact(fine), continuation_commands=fine['commands'][80:],
                continuation_states=fine['predicted_states'][80:],
                records=attempts, selected=name, selection_score=score,
                initial_q=state.q.copy(), initial_v=state.v.copy(), initial_contacts=state.contacts.copy(),
                previous_command=state.previous_command_target.copy(), used_observed_state=True,
                horizon_steps=80, verified_continuation_steps=fine['required_model_steps'],
                observed_control_tick=control_tick, wall_seconds=time.monotonic()-began,
                planner_version='feedback_tail_candidate_selection_v1',
                acceptance_rule='constrained_prefix_plus_explicit_stable_feedback_continuation',
                native_execution=False, valid_expert_labels=False,
                recursive_feasibility_proven=False, per_call_hard_deadline=False)
        return dict(accepted=False, commands=None, records=attempts, selected=None,
            initial_q=state.q.copy(), initial_v=state.v.copy(), initial_contacts=state.contacts.copy(),
            previous_command=state.previous_command_target.copy(), used_observed_state=True,
            horizon_steps=80, observed_control_tick=control_tick, wall_seconds=time.monotonic()-began,
            planner_version='feedback_tail_candidate_selection_v1',
            acceptance_rule='constrained_prefix_plus_explicit_stable_feedback_continuation',
            reason='No candidate has a verified stable continuation within the solve budget',
            native_execution=False, valid_expert_labels=False,
            recursive_feasibility_proven=False, per_call_hard_deadline=False)
