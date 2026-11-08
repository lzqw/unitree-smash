"""Native single-writer adapter retaining the complete V15 recovery baseline."""
import json
from scripts.native_pure_control_preserved_stance_replanning_v17 import NativePureControlPreservedStanceReplanningV17
from scripts.pure_control_preserved_foot_replanning_v20 import PureControlPreservedFootReplanningV20
from scripts.whole_body_support_qp_v1 import PositionPDInterface


class NativePureControlPreservedFootReplanningV20(NativePureControlPreservedStanceReplanningV17):
    def controller(self, lane):
        if lane not in self.controllers:
            self.controllers[lane] = PureControlPreservedFootReplanningV20(
                PositionPDInterface(**self.interface_config), self.nominal,
                self.library, self.replanning_library)
        return self.controllers[lane]

    def save(self):
        result = super().save()
        result.update(controller='pure_control_preserved_foot_replanning_v20',
            preserves_complete_v15_baseline=True)
        (self.output/'pure_control_result.json').write_text(json.dumps(result, indent=2)+'\n')
        return result
