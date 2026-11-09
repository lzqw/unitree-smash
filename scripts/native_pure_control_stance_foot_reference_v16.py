"""Native adapter for the command-continuous V16 recovery candidate."""
import json

from scripts.native_pure_control_settled_stance_v14 import NativePureControlSettledStanceV14
from scripts.pure_control_stance_foot_reference_v16 import PureControlStanceFootReferenceV16
from scripts.whole_body_support_qp_v1 import PositionPDInterface


class NativePureControlStanceFootReferenceV16(NativePureControlSettledStanceV14):
    def controller(self, lane):
        if lane not in self.controllers:
            self.controllers[lane] = PureControlStanceFootReferenceV16(
                PositionPDInterface(**self.interface_config), self.nominal, self.library)
        return self.controllers[lane]

    def save(self):
        result = super().save()
        result.update(controller='pure_control_stance_foot_reference_v16',
                      fallback_mode='hold_last_admitted_target_with_velocity_damping',
                      fallback_certified=False)
        (self.output / 'pure_control_result.json').write_text(
            json.dumps(result, indent=2) + '\n')
        return result


__all__ = ['NativePureControlStanceFootReferenceV16']
