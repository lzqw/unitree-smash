"""CPU full-body admission for contact-first arm prefixes.

This adapter replays each proposed six-joint prefix through the repository's
loaded Pinocchio model and original implicit position interface.  It holds the
12 leg targets at their measured values; it is therefore an Approach-prefix
admission check, not a flight controller and not a PhysX safety certificate.
No ground force is invented when the measured state is airborne.
"""
from copy import deepcopy
from pathlib import Path

import numpy as np

from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance
from scripts.implicit_support_prediction_v1 import predict_implicit_support
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1, PublicState
from scripts.whole_body_support_qp_v1 import PositionPDInterface


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GEOMETRY = ROOT / "configs/go2_piper_authored_ground_geometry_v1.json"


class ContactFirstFullBodyEvaluatorV1:
    """Causal, short-interval full-body candidate evaluator.

    ``public`` and ``default`` are the measured state at the planning tick.
    The evaluator is deterministic and replays every candidate from that same
    state, so candidates cannot borrow mutable state from one another.
    """

    def __init__(self, public=None, default=None, interface_config=None, *,
                 geometry_path=DEFAULT_GEOMETRY, substeps=4,
                 max_tracking_error_rad=0.25, initial_state=None, model=None,
                 enforce_target_admission_bounds=False):
        if initial_state is None:
            public = np.asarray(public, dtype=np.float64)
            default = np.asarray(default, dtype=np.float64)
            if public.shape != (794,) or default.shape != (18,):
                raise ValueError("Contact-first full-body evaluator requires public794 and default18")
            if not np.isfinite(public).all() or not np.isfinite(default).all():
                raise ValueError("Contact-first full-body evaluator requires finite state")
        elif not isinstance(initial_state, PublicState):
            raise ValueError("initial_state must be a LoadedPinocchio PublicState")
        if type(substeps) is not int or not 1 <= substeps <= 32:
            raise ValueError("Evaluator substeps must be in [1, 32]")
        if not np.isfinite(max_tracking_error_rad) or max_tracking_error_rad <= 0.:
            raise ValueError("Positive tracking admission margin required")
        if not isinstance(interface_config, dict):
            raise ValueError("Full-body evaluator requires an interface mapping")
        self.model = LoadedPinocchioModelV1() if model is None else model
        self.interface = PositionPDInterface(**interface_config)
        self.geometry = AuthoredGroundClearance(geometry_path, self.model, margin_m=0.)
        if initial_state is None:
            self.initial_state = self.model.state_from_public(public, default)
        else:
            if (initial_state.q.shape != (25,) or initial_state.v.shape != (24,) or
                    initial_state.joint_position.shape != (18,) or
                    initial_state.joint_velocity.shape != (18,) or
                    initial_state.contacts.shape != (4,) or
                    not all(np.isfinite(value).all() for value in (
                        initial_state.q, initial_state.v, initial_state.joint_position,
                        initial_state.joint_velocity, initial_state.root_position_world,
                        initial_state.root_rotation_world))):
                raise ValueError("initial_state has invalid full-body dimensions")
            self.initial_state = deepcopy(initial_state)
        self.initial_joint_position = self.initial_state.joint_position.copy()
        self.substeps = substeps
        self.max_tracking_error_rad = float(max_tracking_error_rad)
        # The one-tick target envelope is an application-level command slew,
        # not an original joint speed/torque/position limit.  Contact-first
        # search must be allowed to propose a fast arm prefix; the implicit
        # full-body replay below still rejects actual physical velocity,
        # torque, position, contact, and tracking violations.  Keep an opt-in
        # compatibility switch for callers auditing the historical envelope.
        self.enforce_target_admission_bounds = bool(enforce_target_admission_bounds)
        self.dt_s = 0.01

    @classmethod
    def from_native(cls, env, lane, interface_config, *, default=None,
                    contact_force_threshold_n=0.0, **kwargs):
        """Construct from the live native articulation tensors.

        ``LoadedPinocchioModelV1.state_from_native`` reads the root pose/twist,
        admitted 18-joint state and measured foot contacts.  The evaluator
        still performs a CPU replay and does not query or command PhysX.
        """
        model = LoadedPinocchioModelV1()
        state = model.state_from_native(
            env, lane, default=default,
            contact_force_threshold_n=contact_force_threshold_n)
        return cls(interface_config=interface_config, initial_state=state,
                   model=model, **kwargs)

    @staticmethod
    def _violations(diagnostics):
        limits = {
            "dynamics_contact_residual_max": 2e-5,
            "joint_velocity_violation_rad_s": 3e-4,
            "joint_position_violation_rad": 3e-4,
            "normal_force_violation_n": 3e-4,
            "friction_violation_n": 3e-4,
            "motor_torque_violation_nm": 3e-4,
        }
        return {
            name: float(diagnostics.get(name, 0.))
            for name, limit in limits.items()
            if float(diagnostics.get(name, 0.)) > limit
        }

    def __call__(self, q_rad, qd_rad_s, contact_index, metadata):
        q_rad = np.asarray(q_rad, dtype=np.float64)
        qd_rad_s = np.asarray(qd_rad_s, dtype=np.float64)
        if q_rad.ndim != 2 or q_rad.shape[1] != 6 or not np.isfinite(q_rad).all():
            return {"accepted": False, "rejection": "invalid_arm_position_sequence"}
        if qd_rad_s.shape != q_rad.shape or not np.isfinite(qd_rad_s).all():
            return {"accepted": False, "rejection": "invalid_arm_velocity_sequence"}
        contact_index = int(contact_index)
        if not 1 <= contact_index < len(q_rad):
            return {"accepted": False, "rejection": "invalid_contact_index"}
        arm_velocity_limit = self.interface.velocity[12:]
        velocity_violation = np.maximum(np.abs(qd_rad_s[:contact_index + 1]) - arm_velocity_limit, 0.)
        if np.any(velocity_violation > 3e-4):
            return {
                "accepted": False,
                "rejection": "physical_arm_velocity_limit",
                "node": int(np.argmax(np.max(velocity_violation, axis=1))),
                "maximum_velocity_violation_rad_s": float(np.max(velocity_violation)),
            }

        state = deepcopy(self.initial_state)
        maximum_tracking_error = 0.
        minimum_ground_gap = float("inf")
        maximum_diagnostics = {}
        for node in range(contact_index + 1):
            target = self.initial_joint_position.copy()
            target[12:] = q_rad[node]
            if self.enforce_target_admission_bounds:
                low, high = self.interface.admitted_target_bounds(
                    state.joint_position, self.dt_s)
                if np.any(target < low - 1e-6) or np.any(target > high + 1e-6):
                    return {
                        "accepted": False,
                        "rejection": "native_admitted_target_bounds",
                        "node": node,
                    }
            try:
                result = predict_implicit_support(
                    self.model, self.interface, state, target,
                    duration_s=self.dt_s, substeps=self.substeps)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                return {
                    "accepted": False,
                    "rejection": "implicit_position_interface",
                    "node": node,
                    "error": repr(error),
                }
            state = result["state"]
            diagnostics = result["diagnostics"]
            violations = self._violations(diagnostics)
            for name, value in diagnostics.items():
                if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
                    maximum_diagnostics[name] = max(maximum_diagnostics.get(name, 0.), float(value))
            gap = float(np.min(self.geometry.evaluate(state.q)[0]))
            minimum_ground_gap = min(minimum_ground_gap, gap)
            if gap < -1e-6:
                violations["nonfoot_ground_violation_m"] = -gap
            tracking_error = float(np.max(np.abs(state.joint_position[12:] - q_rad[node])))
            maximum_tracking_error = max(maximum_tracking_error, tracking_error)
            if tracking_error > self.max_tracking_error_rad:
                violations["arm_tracking_error_rad"] = tracking_error
            if violations:
                return {
                    "accepted": False,
                    "rejection": "full_body_prefix_violation",
                    "node": node,
                    "violations": violations,
                    "minimum_ground_gap_m": minimum_ground_gap,
                    "maximum_tracking_error_rad": maximum_tracking_error,
                    "diagnostics": maximum_diagnostics,
                }

        return {
            "accepted": True,
            "nodes_checked": contact_index + 1,
            "minimum_ground_gap_m": minimum_ground_gap,
            "maximum_tracking_error_rad": maximum_tracking_error,
            "diagnostics": maximum_diagnostics,
            "native_execution": False,
            "physical_model": "LoadedPinocchioModelV1+implicit_position_interface",
            "airborne_ground_force_policy": "measured_contacts_only",
        }


__all__ = ["ContactFirstFullBodyEvaluatorV1"]
