"""Use one q/v-derived state representation for planning and native feedback.

The measured q/v, contacts and command history are preserved. Only redundant
fields are reconstructed identically to the CPU planner. This avoids an extra
rotation-matrix/quaternion round trip in one side of the exact command check.
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pinocchio as pin

from scripts.native_current_state_recovery_v1 import NativeCurrentStateRecoveryV1
from scripts.audit_joint_sequence_proposal_v1 import plain


def canonical_state(state, model):
    result = copy.deepcopy(state)
    values = dict(root_rotation_world=pin.Quaternion(state.q[3:7]).matrix(),
        root_position_world=state.q[:3].copy(),
        joint_position=state.q[model.q_indices].copy(),
        joint_velocity=state.v[model.v_indices].copy())
    errors = {}
    for name, value in values.items():
        original = np.asarray(getattr(state, name))
        if original.shape != value.shape or not np.isfinite(original).all() or not np.isfinite(value).all():
            raise ValueError('Invalid redundant native state: ' + name)
        errors[name] = float(np.max(abs(original - value)))
        # This is a consistency guard for redundant representations, not a
        # relaxation of command equality or any physical/task constraint.
        if errors[name] > 1e-10:
            raise ValueError('Native q/v contradict derived field: ' + name)
        setattr(result, name, value)
    if not all(np.array_equal(getattr(result, name), getattr(state, name)) for name in ['q', 'v', 'contacts']):
        raise ValueError('Canonicalization changed measured state')
    if hasattr(state, 'previous_command_target') and not np.array_equal(result.previous_command_target, state.previous_command_target):
        raise ValueError('Canonicalization changed command history')
    return result, errors


class NativeCurrentStateRecoveryV2(NativeCurrentStateRecoveryV1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.representation_records = []

    def _save_representation(self):
        if not hasattr(self, 'representation_records'):
            return
        path = self.output / 'state_representation.json'
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(dict(scope=__doc__, records=self.representation_records,
            exact_command_comparison_preserved=True, measured_q_v_contacts_preserved=True,
            does_not_certify_native_completion=True, valid_expert_labels=False,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2) + '\n')
        tmp.replace(path)

    def action(self, tick, public, student, finished, native_state=None):
        if tick >= self.planning_ticks[0] and not bool(finished[self.case]):
            if native_state is None:
                raise ValueError('Current measured state required')
            native_state, errors = canonical_state(native_state, self.model)
            self.representation_records.append(dict(control_tick=tick, redundant_field_errors=errors))
            if tick in self.planning_ticks:
                self._save_representation()
        return super().action(tick, public, student, finished, native_state)

    def close(self):
        super().close()
        self._save_representation()

    def save(self):
        result = super().save()
        result['state_representation'] = 'q/v-derived fields shared with current-state planner'
        path = self.output / 'sequence_native_result.json'
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(plain(result), indent=2, allow_nan=False) + '\n')
        tmp.replace(path)
        return result
