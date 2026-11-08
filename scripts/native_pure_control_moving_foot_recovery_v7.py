"""Same full18 owner with coherent moving-foot reference feedback."""
import json
from scripts.native_pure_control_self_collision_recovery_v5 import NativePureControlSelfCollisionRecoveryV5
from scripts.pure_control_moving_foot_recovery_v7 import PureControlMovingFootRecoveryV7
from scripts.whole_body_support_qp_v1 import PositionPDInterface


class NativePureControlMovingFootRecoveryV7(NativePureControlSelfCollisionRecoveryV5):
    def controller(self,lane):
        if lane not in self.controllers:
            self.controllers[lane]=PureControlMovingFootRecoveryV7(PositionPDInterface(**self.interface_config),self.nominal)
        return self.controllers[lane]

    def save(self):
        result=super().save();result.update(controller='pure_control_moving_foot_recovery_v7',moving_foot_reference=True)
        (self.output/'pure_control_result.json').write_text(json.dumps(result,indent=2)+'\n');return result
