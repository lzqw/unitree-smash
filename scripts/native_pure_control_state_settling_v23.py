"""Single native recovery owner with current-state-qualified support transition."""
import json
from scripts.native_pure_control_stance_foot_reference_v15 import NativePureControlStanceFootReferenceV15
from scripts.pure_control_state_settling_v23 import PureControlStateSettlingV23
from scripts.whole_body_support_qp_v1 import PositionPDInterface


class NativePureControlStateSettlingV23(NativePureControlStanceFootReferenceV15):
    def controller(self, lane):
        if lane not in self.controllers:
            self.controllers[lane] = PureControlStateSettlingV23(
                PositionPDInterface(**self.interface_config), self.nominal, self.library)
        return self.controllers[lane]

    def save(self):
        result = super().save()
        result.update(controller='pure_control_state_settling_v23',
            settling_requires_reference_end=False, earlier_stance_is_not_success=True)
        (self.output / 'pure_control_result.json').write_text(json.dumps(result, indent=2) + '\n')
        return result
