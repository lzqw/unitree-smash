"""Original native command owner with experimental state-qualified stance feedback."""
import json

from scripts.native_pure_control_reference_clock_v12 import NativePureControlReferenceClockV12
from scripts.pure_control_settled_stance_v14 import PureControlSettledStanceV14
from scripts.whole_body_support_qp_v1 import PositionPDInterface


class NativePureControlSettledStanceV14(NativePureControlReferenceClockV12):
    def controller(self, lane):
        if lane not in self.controllers:
            self.controllers[lane] = PureControlSettledStanceV14(
                PositionPDInterface(**self.interface_config), self.nominal, self.library)
        return self.controllers[lane]

    def save(self):
        result = super().save()
        result.update(controller='pure_control_settled_stance_v14',
            terminal_stance_controllers=sum(c.terminal_stance is not None
                for c in self.controllers.values()),
            terminal_stance_soft_target_not_command_hold=True,
            terminal_stance_certified=False)
        (self.output / 'pure_control_result.json').write_text(
            json.dumps(result, indent=2) + '\n')
        return result
