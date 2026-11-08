"""Same causal ownership, with hip/mount-aware full18 interval planning."""
import json
from scripts.native_pure_control_court_recovery_v4 import NativePureControlCourtRecoveryV4
from scripts.pure_control_self_collision_recovery_v5 import PureControlSelfCollisionRecoveryV5
from scripts.whole_body_support_qp_v1 import PositionPDInterface


class NativePureControlSelfCollisionRecoveryV5(NativePureControlCourtRecoveryV4):
    def controller(self,lane):
        if lane not in self.controllers:
            self.controllers[lane]=PureControlSelfCollisionRecoveryV5(PositionPDInterface(**self.interface_config),self.nominal)
        return self.controllers[lane]

    def save(self):
        result=super().save()
        result.update(controller='pure_control_self_collision_recovery_v5',hip_mount_avoidance_in_planning=True)
        (self.output/'pure_control_result.json').write_text(json.dumps(result,indent=2)+'\n')
        return result
