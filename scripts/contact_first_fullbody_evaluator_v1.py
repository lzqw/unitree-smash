"""CPU full-body admission for contact-first arm prefixes.

This adapter replays each proposed six-joint prefix through the repository's
loaded Pinocchio model and original implicit position interface.  It can replay an explicit 12-leg target sequence alongside the six-axis arm
prefix. Without one it retains the historical measured-leg hold. It is an
Approach-prefix admission check, not a flight controller or PhysX certificate.
No ground force is invented when the measured state is airborne.
"""
from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np

from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance
from scripts.implicit_support_prediction_v1 import predict_implicit_support
from scripts.unilateral_implicit_prediction_v1 import predict_unilateral_support
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1, PublicState
from scripts.whole_body_support_qp_v1 import PositionPDInterface


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GEOMETRY = ROOT / "configs/go2_piper_authored_ground_geometry_v1.json"


class ContactFirstFullBodyEvaluatorV1:
    """Causal, short-interval full-body candidate evaluator.

    ``public`` and ``default`` are the measured state at the planning tick.
    The evaluator is deterministic and replays every candidate from that same
    state, so candidates cannot borrow mutable state from one another.

    ``racket_measure_fn`` must return position, normal and inertial point
    velocity expressed in the planner's fixed current planning frame. When
    that frame is the initial body frame, callers must transform native world
    measurements explicitly; it is not the changing future body frame. The
    legacy metadata keys ending in ``world`` refer to this common fixed frame.
    """

    def __init__(self, public=None, default=None, interface_config=None, *,
                 geometry_path=DEFAULT_GEOMETRY, substeps=4,
                 max_tracking_error_rad=0.25, initial_state=None, model=None,
                 enforce_target_admission_bounds=False,
                 racket_measure_fn=None, leg_target_sequence_rad=None,
                 record_state_trace=False, contact_prediction="measured_fixed",
                 leg_reference_factory=None, impact_state_fn=None, measurement_uses_metadata=False, enforce_joint_stops=False):
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
        if racket_measure_fn is not None and not callable(racket_measure_fn):
            raise TypeError("racket_measure_fn must be callable or None")
        self.racket_measure_fn = racket_measure_fn
        self.measurement_uses_metadata=measurement_uses_metadata
        if leg_reference_factory is not None and (not callable(leg_reference_factory) or leg_target_sequence_rad is not None):
            raise ValueError('Use one causal leg factory or explicit leg sequence')
        self.leg_reference_factory=leg_reference_factory
        self.impact_state_fn=impact_state_fn
        self.dt_s = 0.01
        self.leg_target_sequence_rad = None
        if leg_target_sequence_rad is not None:
            legs = np.array(leg_target_sequence_rad, dtype=np.float64, copy=True)
            if (legs.ndim != 2 or legs.shape[1] != 12 or len(legs) < 2 or
                    not np.isfinite(legs).all()):
                raise ValueError("Leg target sequence must be finite [nodes, 12] radians")
            if (np.any(legs < self.interface.limits[:12, 0]-1e-6) or
                    np.any(legs > self.interface.limits[:12, 1]+1e-6)):
                raise ValueError("Leg target sequence outside original position limits")
            legs.flags.writeable = False
            self.leg_target_sequence_rad = legs
        self.record_state_trace = bool(record_state_trace)
        if contact_prediction not in ("measured_fixed", "unilateral_release"):
            raise ValueError("Unknown contact prediction mode")
        if enforce_joint_stops and contact_prediction != "unilateral_release":
            raise ValueError("Joint stops are experimental Approach unilateral prediction only")
        self.enforce_joint_stops=bool(enforce_joint_stops)
        self.contact_prediction = contact_prediction
        self.predict = (predict_unilateral_support if contact_prediction == "unilateral_release"
                        else predict_implicit_support)

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
            "dynamics_residual_max": 2e-5,
            "joint_velocity_violation_rad_s": 3e-4,
            "joint_position_violation_rad": 3e-4,
            "normal_force_violation_n": 3e-4,
            "friction_violation_n": 3e-4,
            "motor_torque_violation_nm": 3e-4,
        }
        if not isinstance(diagnostics,dict):return {'invalid_diagnostics':float('inf')}
        residuals=('dynamics_contact_residual_max','dynamics_residual_max')
        violations={}
        if not any(k in diagnostics for k in residuals):violations['missing_dynamics_residual']=float('inf')
        for name,limit in limits.items():
            if name in residuals and name not in diagnostics:continue
            value=diagnostics.get(name)
            try:value=float(value)
            except (TypeError,ValueError):value=float('nan')
            if not np.isfinite(value) or value<0:
                violations['invalid_'+name]=float('inf')
            elif value>limit:violations[name]=value
        return violations

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
        if (self.leg_target_sequence_rad is not None and
                len(self.leg_target_sequence_rad) < contact_index):
            return {"accepted": False, "rejection": "incomplete_leg_target_sequence"}
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
        initial_witness=self.geometry.minimum_witness(state.q)
        if initial_witness['gap_m'] < -1e-6:
            return dict(accepted=False,rejection='initial_nonfoot_ground_violation',
                node=0,geometry_witness_before=initial_witness,
                violations={'nonfoot_ground_violation_m':-initial_witness['gap_m']})
        maximum_tracking_error = 0.
        minimum_ground_gap = float("inf")
        maximum_diagnostics = {}
        targets = np.repeat(self.initial_joint_position[None], contact_index, axis=0)
        targets[:, 12:] = q_rad[:contact_index]
        if self.leg_target_sequence_rad is not None:
            targets[:, :12] = self.leg_target_sequence_rad[:contact_index]
        # Hash exactly the replayed absolute commands, not a clipped approximation.
        command_sha256 = hashlib.sha256(targets.astype('<f8').tobytes()).hexdigest()
        leg_reference = None if self.leg_reference_factory is None else self.leg_reference_factory()
        state_trace = []
        contact_modes_changed = False
        # q[0] is the boundary at the measured planning state. Command i is
        # held over [i*dt, (i+1)*dt); contact_index therefore requires exactly
        # contact_index physical intervals, not contact_index+1.
        for node in range(contact_index):
            target = targets[node].copy()
            if leg_reference is not None:
                legs=np.asarray(leg_reference(deepcopy(state),node),dtype=float)
                if legs.shape!=(12,) or not np.isfinite(legs).all():
                    return dict(accepted=False,rejection='invalid_causal_leg_reference',node=node)
                target[:12]=legs;targets[node]=target
            before_witness=self.geometry.minimum_witness(state.q)
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
                result = self.predict(
                    self.model, self.interface, state, target,
                    duration_s=self.dt_s, substeps=self.substeps,
                    **({"enforce_joint_stops": True} if self.enforce_joint_stops else {}))
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                return {
                    "accepted": False,
                    "rejection": "implicit_position_interface",
                    "node": node,
                    "error": repr(error),
                }
            if leg_reference is not None and hasattr(leg_reference, 'commit_final_command'):
                leg_reference.commit_final_command(target.copy(),node)
            contact_modes_changed |= not np.array_equal(state.contacts, result["state"].contacts)
            state = result["state"]
            if not all(np.isfinite(v).all() for v in (state.q,state.v,state.joint_position,state.joint_velocity)):
                return dict(accepted=False,rejection='nonfinite_predicted_state',node=node)
            diagnostics = result.get("diagnostics",{})
            invalid=self._violations(diagnostics)
            if any(k.startswith(("invalid_","missing_")) for k in invalid):
                return dict(accepted=False,rejection="invalid_physics_diagnostics",node=node,violations=invalid)
            if self.record_state_trace:
                state_trace.append(dict(
                    node=node, elapsed_s=(node+1)*self.dt_s,
                    final_target_rad=target.tolist(), q=state.q.tolist(), v=state.v.tolist(),
                    contacts=state.contacts.tolist(),
                    foot_forces_world_n=(result["foot_forces_world_n"].tolist()
                        if "foot_forces_world_n" in result else None)))
            violations = self._violations(diagnostics)
            for name, value in diagnostics.items():
                if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
                    maximum_diagnostics[name] = max(maximum_diagnostics.get(name, 0.), float(value))
            after_witness = self.geometry.minimum_witness(state.q)
            gap = after_witness["gap_m"]
            minimum_ground_gap = min(minimum_ground_gap, gap)
            if not np.isfinite(gap):
                return dict(accepted=False,rejection="nonfinite_geometry",node=node)
            if gap < -1e-6:
                violations["nonfoot_ground_violation_m"] = -gap
            tracking_error = float(np.max(np.abs(state.joint_position[12:] - q_rad[node+1])))
            maximum_tracking_error = max(maximum_tracking_error, tracking_error)
            if tracking_error > self.max_tracking_error_rad:
                violations["arm_tracking_error_rad"] = tracking_error
            if violations:
                return {
                    "accepted": False,
                    "rejection": "full_body_prefix_violation",
                    "joint_limit_witness": dict(
                        joint=self.model.joint_order[int(np.argmax(np.maximum(
                            self.interface.limits[:,0]-state.joint_position,
                            state.joint_position-self.interface.limits[:,1])))],
                        positions_rad=state.joint_position.tolist(),
                        limits_rad=self.interface.limits.tolist(),
                        final_target_rad=target.tolist()),
                    "state_trace":state_trace if self.record_state_trace else None,
                    "node": node,
                    "violations": violations,
                    "geometry_witness_before": before_witness,
                    "geometry_witness_initial": initial_witness,
                    "geometry_witness_after": after_witness,
                    "contact_mask": state.contacts.tolist(),
                    "minimum_ground_gap_m": minimum_ground_gap,
                    "maximum_tracking_error_rad": maximum_tracking_error,
                    "diagnostics": maximum_diagnostics,
                }

        result = {
            "geometry_witness_initial": initial_witness,
            "geometry_witness_terminal": after_witness,
            "accepted": True,
            "nodes_checked": contact_index + 1,
            "prediction_intervals": contact_index,
            "terminal_elapsed_s": contact_index*self.dt_s,
            "leg_reference": ("causal_frozen_base" if leg_reference is not None else "explicit_sequence" if self.leg_target_sequence_rad is not None
                              else "measured_position_hold"),
            "replayed_command_sha256": hashlib.sha256(targets.astype("<f8").tobytes()).hexdigest(),
            "replayed_target_sequence_rad": targets.tolist(),
            "contact_modes_changed": bool(contact_modes_changed),
            "minimum_ground_gap_m": minimum_ground_gap,
            "maximum_tracking_error_rad": maximum_tracking_error,
            "diagnostics": maximum_diagnostics,
            "native_execution": False,
            "physical_model": "LoadedPinocchioModelV1+implicit_position_interface",
            "contact_prediction": self.contact_prediction,
            "joint_stops_enabled": self.enforce_joint_stops,
            "new_contact_candidates_added": False,
            "airborne_ground_force_policy": "measured_contacts_only",
        }
        if self.record_state_trace:
            result["state_trace"] = state_trace
        if self.impact_state_fn is not None:
            result["impact_robot_state"]=self.impact_state_fn(state,metadata) if self.measurement_uses_metadata else self.impact_state_fn(state)
        if self.racket_measure_fn is not None:
            try:
                measured = self.racket_measure_fn(state,metadata) if self.measurement_uses_metadata else self.racket_measure_fn(state)
                if not isinstance(measured, (tuple, list)) or len(measured) != 3:
                    raise ValueError("racket_measure_fn must return position, normal, velocity")
                measured_position = np.asarray(measured[0], dtype=np.float64)
                measured_normal = np.asarray(measured[1], dtype=np.float64)
                measured_velocity = np.asarray(measured[2], dtype=np.float64)
                target_position = np.asarray(metadata["target_position_world_m"], dtype=np.float64)
                target_normal = np.asarray(metadata["target_normal_world"], dtype=np.float64)
                target_velocity = np.asarray(metadata["target_velocity_world_m_s"], dtype=np.float64)
                if any(x.shape != (3,) or not np.isfinite(x).all() for x in (
                        measured_position, measured_normal, measured_velocity,
                        target_position, target_normal, target_velocity)):
                    raise ValueError("racket measure and target vectors must be finite 3D")
                normal_norm = np.linalg.norm(measured_normal)
                if normal_norm < 1e-9:
                    raise ValueError("racket_measure_fn returned a zero normal")
                measured_normal /= normal_norm
                position_error = float(np.linalg.norm(measured_position - target_position))
                normal_error = float(np.linalg.norm(measured_normal - target_normal))
                velocity_error = float(np.linalg.norm(measured_velocity - target_velocity))
                result.update(dict(
                    actual_racket_position_world_m=measured_position,
                    actual_racket_normal_world=measured_normal,
                    actual_racket_velocity_world_m_s=measured_velocity,
                    actual_racket_position_error_m=position_error,
                    actual_racket_normal_error=normal_error,
                    actual_racket_velocity_error_m_s=velocity_error))
                if (position_error > float(metadata.get("position_tolerance_m", .035)) or
                        normal_error > float(metadata.get("normal_tolerance", .12)) or
                        velocity_error > float(metadata.get("velocity_tolerance_m_s", .75))):
                    result.update(accepted=False, rejection="actual_racket_terminal_residual")
            except (KeyError, TypeError, ValueError, FloatingPointError) as error:
                result.update(accepted=False, rejection="actual_racket_measurement", error=repr(error))
        return result


__all__ = ["ContactFirstFullBodyEvaluatorV1"]
