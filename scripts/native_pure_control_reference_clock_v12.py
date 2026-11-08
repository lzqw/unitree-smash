"""Causal full18 control with model-reference matching by actual release state."""
import json
from pathlib import Path
from scripts.native_pure_control_moving_foot_recovery_v7 import NativePureControlMovingFootRecoveryV7
from scripts.pure_control_reference_clock_v12 import PureControlReferenceClockV12
from scripts.whole_body_support_qp_v1 import PositionPDInterface


class NativePureControlReferenceClockV12(NativePureControlMovingFootRecoveryV7):
    def __init__(self,env,limits,default,configuration,output,observer):
        super().__init__(env,limits,default,configuration,output,observer)
        self.library=json.loads(Path(configuration).read_text())['model_reference_library']

    def controller(self,lane):
        if lane not in self.controllers:
            self.controllers[lane]=PureControlReferenceClockV12(PositionPDInterface(**self.interface_config),self.nominal,self.library)
        return self.controllers[lane]

    def save(self):
        result=super().save();result.update(controller='pure_control_reference_clock_v12',
            model_reference_matched_controllers=sum(c.chosen is not None for c in self.controllers.values()),
            online_sequence_mpc=False,offline_model_reference_with_feedback=True)
        (self.output/'pure_control_result.json').write_text(json.dumps(result,indent=2)+'\n');return result
