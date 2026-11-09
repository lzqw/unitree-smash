"""State-driven short-horizon landing MPC without a reference library.

The controller starts from the current measured full-body state on every
tick.  It builds a small set of state-derived joint targets, evaluates each
through the existing coupled position/PD interval QP and a two-step physical
prediction, and executes only the first target of the lowest-risk candidate.
The candidate set is an initialization aid; no TRAIN id, failure list or
historical trajectory is required.  All original limits and authored
collision checks remain active.  This is a rolling candidate MPC, not a
real-time nonlinear MPC certificate.
"""
import copy
import time
import numpy as np

from scripts.pure_control_court_recovery_v4 import PureControlCourtRecoveryV4
from scripts.predict_self_contact_spheres_v2 import predict_self_contact_spheres


class PureControlStateDrivenMPCV1(PureControlCourtRecoveryV4):
    """Generate and score recovery actions from the current state only."""

    def _base_reference(self, state):
        # Start from the measured full-body configuration.  Pulling an
        # airborne robot directly toward the default standing posture was the
        # main failure mode of the first native trial: the QP was feasible but
        # commanded a large, state-independent jump in posture.  A short
        # velocity-damping horizon is state-driven and keeps the current
        # reachable leg/arm geometry while the swing-foot task handles
        # touchdown placement.
        reference = state.joint_position.copy()
        reference -= .08 * np.clip(state.joint_velocity, -3., 3.)
        if state.contacts.any() and self.initial_arm is not None:
            settle = np.clip((self.elapsed - .12) / .8, 0., 1.)
            reference[12:] = ((1. - settle) * state.joint_position[12:]
                              + settle * self.nominal[12:])
        return reference

    def _candidate_references(self, state):
        low, high = self.interface.admitted_target_bounds(state.joint_position, .01)
        base = np.clip(self._base_reference(state), low, high)
        measured = np.clip(state.joint_position - .02 * np.clip(state.joint_velocity, -5., 5.), low, high)
        blended = np.clip(.5 * base + .5 * measured, low, high)
        # The arm keeps its measured braking state in all candidates; the
        # legs and body are still free to be solved jointly by the QP.
        for candidate in (measured, blended):
            candidate[12:] = np.clip(base[12:], low[12:], high[12:])
        # Keep the online search bounded: the controller remains state-driven
        # and library-independent, while avoiding three full CPU QPs per lane
        # at the same control tick.  The measured/damped target is folded into
        # the blended candidate rather than evaluated as a third branch.
        return [('state_nominal', base), ('state_blended', blended)]

    @staticmethod
    def _risk(predicted, geometry, target, previous):
        r = predicted.root_rotation_world
        tilt = float(np.arccos(np.clip(r[2, 2], -1., 1.)))
        horizontal_speed = float(np.linalg.norm((r @ predicted.v[:3])[:2]))
        angular_speed = float(np.linalg.norm(predicted.v[3:6]))
        gaps, _ = geometry.evaluate(predicted.q)
        min_gap = float(np.min(gaps))
        command_change = float(np.max(np.abs(target - previous)))
        # Hard collision violations are rejected before this score. The score
        # only ranks feasible candidates and prefers lower residual motion.
        return (80. * max(0., .01 - min_gap) ** 2
                + 8. * tilt ** 2 + 3. * horizontal_speed ** 2
                + 2. * angular_speed ** 2 + .02 * command_change ** 2)

    def _evaluate_candidate(self, state, target, plan):
        predicted = copy.deepcopy(state)
        previous = predicted.previous_command_target.copy()
        last = None
        # One physical prediction is sufficient for this 10 ms receding
        # horizon; the next tick re-reads the native state and replans.  This
        # preserves the full episode window while keeping the CPU solve
        # bounded for the 128-lane native evaluation.
        for _ in range(1):
            try:
                last = predict_self_contact_spheres(
                    self.model, self.interface, predicted, target,
                    self.geometry.spheres, substeps=8, geometry=self.geometry)
            except (ValueError, FloatingPointError) as exc:
                # A local contact fixed point that does not converge is a
                # rejected candidate, not a reason to abort the whole native
                # episode.  The caller will keep the last admitted command and
                # re-solve from the next measured state.
                return None, dict(accepted=False,
                                  rejection='contact_prediction_not_converged',
                                  error=str(exc))
            diagnostics = last['diagnostics']
            if (diagnostics.get('nonfoot_ground_violation_m', 0.) > 1e-6 or
                    diagnostics.get('joint_position_violation_rad', 0.) > 3e-4 or
                    diagnostics.get('joint_velocity_violation_rad_s', 0.) > 3e-4 or
                    diagnostics.get('motor_torque_violation_nm', 0.) > 3e-4):
                return None, dict(accepted=False, rejection='two-step physical prediction')
            predicted = last['state']
            predicted.previous_command_target = target.copy()
        return predicted, dict(accepted=True,
                               score=self._risk(predicted, self.geometry, target, previous),
                               min_gap_m=float(np.min(self.geometry.evaluate(predicted.q)[0])),
                               prediction_steps=1)

    def command(self, state, phase, time_s):
        if phase not in ['AIR_PREPARE', 'BUFFER_LOAD', 'SUSTAIN']:
            raise ValueError('Recovery ownership required')
        began = time.perf_counter()
        if self.started is None:
            self.started = float(time_s)
            self.initial_arm = state.joint_position[12:].copy()
            self.initial_preload = state.previous_command_target - state.joint_position
        self.phase = phase
        self.elapsed = float(time_s) - self.started
        if self.elapsed < 0.:
            raise ValueError('Reset controller on episode changes')
        candidates = []
        original_previous = state.previous_command_target.copy()
        # Full-authority position targets still require the actual command
        # history for interface bookkeeping.  Bind it before constructing
        # state-derived candidates; this is not a trajectory-library input.
        self.interface.bind_previous_command(original_previous)
        for name, reference in self._candidate_references(state):
            state.previous_command_target = original_previous.copy()
            plan = self.solve(state, reference)
            if not plan.get('accepted'):
                candidates.append(dict(name=name, accepted=False,
                                       rejection=plan.get('rejection'),
                                       solver_status=plan.get('solver_status')))
                continue
            target = np.asarray(plan['joint_position_target'], dtype=float)
            predicted, score = self._evaluate_candidate(state, target, plan)
            candidates.append(dict(name=name, **score,
                                   solver_status=plan.get('solver_status')))
            if score.get('accepted'):
                candidates[-1]['target'] = target.copy()
                candidates[-1]['plan'] = plan
        state.previous_command_target = original_previous
        feasible = [item for item in candidates if item.get('accepted') and 'target' in item]
        if feasible:
            chosen = min(feasible, key=lambda item: item['score'])
            target = chosen['target'].copy()
            source = 'state_driven_mpc:' + chosen['name']
            chosen_name = chosen['name']
        else:
            # No trajectory match is needed. Keep the last admitted command
            # continuous while the next tick attempts a fresh state solve.
            self.interface.bind_previous_command(original_previous)
            low, high = self.interface.admitted_target_bounds(state.joint_position, .01)
            target = np.clip(original_previous - .01 * np.clip(state.joint_velocity, -5., 5.), low, high)
            source = 'state_driven_command_hold'
            chosen_name = None
        surfaces = self.geometry.evaluate(state.q)[0]
        diagnostic = dict(
            controller='pure_control_state_driven_mpc_v1',
            solver_accepted=bool(feasible),
            source=source,
            selected_candidate=chosen_name,
            candidates=[{k: v for k, v in item.items() if k not in ('target', 'plan')} for item in candidates],
            reference_library_required=False,
            train_case_required=False,
            full18_state_driven=True,
            fallback_certified=False,
            minimum_current_nonfoot_gap_m=float(np.min(surfaces)),
            wall_seconds=time.perf_counter() - began,
            native_recovery_certified=False,
            valid_expert_labels=False,
        )
        return target, diagnostic


__all__ = ['PureControlStateDrivenMPCV1']
