"""Keep attained terminal footholds consistent with the terminal body target.

These are soft swing-foot position objectives, never latched contacts or
prescribed support forces. Each interval still solves for all 18 commands
with the actual contact observation and unchanged physical constraints.
"""
import numpy as np

from scripts.pure_control_settled_stance_v14 import PureControlSettledStanceV14


class PureControlStanceFootReferenceV15(PureControlSettledStanceV14):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.terminal_foot_positions = None

    def reference(self, state, snapshot, joint_reference):
        reference = super().reference(state, snapshot, joint_reference)
        if self.terminal_stance is None:
            return reference
        if self.terminal_foot_positions is None:
            ids = [self.model.names.index(name) for name in self.model.feet]
            self.terminal_foot_positions = snapshot['body_positions_world'][ids].copy()
        # With no actual support, retain the inherited ballistic/moving-foot
        # reference. A saved world foothold is not evidence of ground force.
        if state.contacts.any():
            old = reference['swing_foot_target_world'].copy()
            reference['swing_foot_target_world'] = self.terminal_foot_positions.copy()
            reference['swing_foot_velocity_world'] = np.zeros((4, 3))
            reference['swing_foot_acceleration_world'] = np.zeros((4, 3))
            self.reference_diagnostics.update(
                terminal_foot_soft_target=True,
                terminal_foot_target_world=self.terminal_foot_positions.copy(),
                replaced_foot_reference_delta_m=reference['swing_foot_target_world']-old,
                swing_feet=reference['swing_foot_target_world'].copy(),
                force_constraints_use_current_contacts=True)
        return reference

    def command(self, state, phase, time_s):
        target, diagnostic = super().command(state, phase, time_s)
        diagnostic.update(controller='pure_control_stance_foot_reference_v15',
            saved_footholds_are_soft_targets_not_contacts=True,
            terminal_foot_reference_available=self.terminal_foot_positions is not None)
        return target, diagnostic
