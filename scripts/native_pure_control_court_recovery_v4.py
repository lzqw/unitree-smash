"""Native full-command V4 owner; identical causal lifecycle and logging."""
import json
from scripts.native_pure_control_recovery_v1 import NativePureControlRecoveryV1
from scripts.pure_control_court_recovery_v4 import PureControlCourtRecoveryV4
from scripts.full_authority_position_interface_v1 import FullAuthorityCommandArbiterV1
from scripts.whole_body_support_qp_v1 import PositionPDInterface


class NativePureControlCourtRecoveryV4(NativePureControlRecoveryV1):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.arbiter=FullAuthorityCommandArbiterV1(self.limits)

    def controller(self,lane):
        if lane not in self.controllers:
            self.controllers[lane]=PureControlCourtRecoveryV4(PositionPDInterface(**self.interface_config),self.nominal)
        return self.controllers[lane]

    def save(self):
        result=super().save()
        result.update(controller='pure_control_court_recovery_v4',software_arm_command_slew_removed=True,
            physical_actuator_parameters_unchanged=True,sequence_mpc=False)
        (self.output/'pure_control_result.json').write_text(json.dumps(result,indent=2)+'\n')
        return result
