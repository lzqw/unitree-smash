"""Model-only separated-foot variant; native v1 trial remains unchanged."""
from scripts.pure_control_recovery_v1 import PureControlRecoveryV1
from scripts.pure_control_separated_qp_v3 import PureControlSeparatedQPBaseV3

class PureControlSeparatedRecoveryV3(PureControlRecoveryV1):
    solve=PureControlSeparatedQPBaseV3.solve
