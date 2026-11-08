"""One original native command owner with causal local-state replanning."""
import json
from pathlib import Path
from scripts.native_pure_control_reference_clock_v12 import NativePureControlReferenceClockV12
from scripts.pure_control_state_replanning_v16 import PureControlStateReplanningV16
from scripts.whole_body_support_qp_v1 import PositionPDInterface


class NativePureControlStateReplanningV16(NativePureControlReferenceClockV12):
    def __init__(self,env,limits,default,configuration,output,observer):
        super().__init__(env,limits,default,configuration,output,observer)
        self.replanning_library=json.loads(Path(configuration).read_text())['model_replanning_library']

    def controller(self,lane):
        if lane not in self.controllers:
            self.controllers[lane]=PureControlStateReplanningV16(PositionPDInterface(**self.interface_config),
                self.nominal,self.library,self.replanning_library)
        return self.controllers[lane]

    def save(self):
        result=super().save()
        result.update(controller='pure_control_state_replanning_v16',
            replanned_controllers=sum(c.replanned is not None for c in self.controllers.values()),
            actual_state_matching_not_train_id=True,native_success_not_inferred=True)
        (self.output/'pure_control_result.json').write_text(json.dumps(result,indent=2)+'\n')
        return result
