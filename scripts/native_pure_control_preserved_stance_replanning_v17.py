"""One original native command owner with causal local-state replanning."""
import json
from pathlib import Path
from scripts.native_pure_control_settled_stance_v14 import NativePureControlSettledStanceV14
from scripts.pure_control_preserved_stance_replanning_v17 import PureControlPreservedStanceReplanningV17
from scripts.whole_body_support_qp_v1 import PositionPDInterface


class NativePureControlPreservedStanceReplanningV17(NativePureControlSettledStanceV14):
    def __init__(self,env,limits,default,configuration,output,observer):
        super().__init__(env,limits,default,configuration,output,observer)
        self.replanning_library=json.loads(Path(configuration).read_text())['model_replanning_library']

    def controller(self,lane):
        if lane not in self.controllers:
            self.controllers[lane]=PureControlPreservedStanceReplanningV17(PositionPDInterface(**self.interface_config),
                self.nominal,self.library,self.replanning_library)
        return self.controllers[lane]

    def save(self):
        result=super().save()
        result.update(controller='pure_control_preserved_stance_replanning_v17',
            replanned_controllers=sum(c.replanned is not None for c in self.controllers.values()),
            actual_state_matching_not_train_id=True,native_success_not_inferred=True)
        (self.output/'pure_control_result.json').write_text(json.dumps(result,indent=2)+'\n')
        return result
