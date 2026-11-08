"""Experimental head stopping-envelope variant; native v1 is unchanged."""
from scripts.pure_control_recovery_v1 import PureControlRecoveryV1
from scripts.pure_control_stopping_qp_v2 import PureControlStoppingQPBaseV2

class PureControlStoppingRecoveryV2(PureControlRecoveryV1):
    solve=PureControlStoppingQPBaseV2.solve
