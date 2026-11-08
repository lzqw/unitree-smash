"""Native single-owner recovery with terminal foothold soft objectives."""
import json
from scripts.native_pure_control_settled_stance_v14 import NativePureControlSettledStanceV14
from scripts.pure_control_stance_foot_reference_v15 import PureControlStanceFootReferenceV15
from scripts.whole_body_support_qp_v1 import PositionPDInterface


class NativePureControlStanceFootReferenceV15(NativePureControlSettledStanceV14):
    def controller(self, lane):
        if lane not in self.controllers:
            self.controllers[lane] = PureControlStanceFootReferenceV15(
                PositionPDInterface(**self.interface_config), self.nominal, self.library)
        return self.controllers[lane]

    def save(self):
        result = super().save()
        result.update(controller='pure_control_stance_foot_reference_v15',
            saved_footholds_are_soft_targets_not_contacts=True)
        (self.output/'pure_control_result.json').write_text(json.dumps(result,indent=2)+'\n')
        return result
