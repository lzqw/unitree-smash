"""Native adapter for the single future-sequence repair candidate."""
import json
import numpy as np

from scripts.native_pure_control_recovery_v1 import NativePureControlRecoveryV1
from scripts.pure_control_state_driven_long_mpc_v1_qp_continuity_repair import (
    PureControlStateDrivenLongMPCV1QPContinuityRepair,
)
from scripts.whole_body_support_qp_v1 import PositionPDInterface


class NativePureControlStateDrivenLongMPCV1QPContinuityRepair(
        NativePureControlRecoveryV1):
    def action(self, tick, observation, student, finished):
        start = len(self.rows)
        result = super().action(tick, observation, student, finished)
        for row in self.rows[start:]:
            preview = row['diagnostic']['execution_preview']
            error = float(np.max(abs(np.asarray(preview['predicted_final_target']) - row['final'])))
            row['diagnostic']['prediction_dispatch_max_delta_rad'] = error
            if error > 2e-6:
                raise ValueError('Prediction preview differs from unchanged native arbiter')
        return result

    def controller(self, lane):
        if lane not in self.controllers:
            self.controllers[lane] = PureControlStateDrivenLongMPCV1QPContinuityRepair(
                PositionPDInterface(**self.interface_config), self.nominal,
            )
        return self.controllers[lane]

    def save(self):
        result = super().save()
        result.update(
            controller='pure_control_state_driven_long_mpc_v1_qp_continuity_repair',
            reference_library_required=False,
            train_case_required=False,
            fallback_certified=False,
        )
        (self.output / 'pure_control_result.json').write_text(
            json.dumps(result, indent=2) + '\n',
        )
        return result


__all__ = ['NativePureControlStateDrivenLongMPCV1QPContinuityRepair']
