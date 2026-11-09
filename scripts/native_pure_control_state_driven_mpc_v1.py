"""Native single-owner adapter for the state-driven landing MPC."""
import json

from scripts.native_pure_control_recovery_v1 import NativePureControlRecoveryV1
from scripts.pure_control_state_driven_mpc_v1 import PureControlStateDrivenMPCV1
from scripts.whole_body_support_qp_v1 import PositionPDInterface


class NativePureControlStateDrivenMPCV1(NativePureControlRecoveryV1):
    def controller(self, lane):
        if lane not in self.controllers:
            self.controllers[lane] = PureControlStateDrivenMPCV1(
                PositionPDInterface(**self.interface_config), self.nominal)
        return self.controllers[lane]

    def save(self):
        result = super().save()
        result.update(controller='pure_control_state_driven_mpc_v1',
                      reference_library_required=False,
                      train_case_required=False,
                      fallback_certified=False)
        (self.output / 'pure_control_result.json').write_text(
            json.dumps(result, indent=2) + '\n')
        return result


__all__ = ['NativePureControlStateDrivenMPCV1']
