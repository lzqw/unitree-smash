import numpy as np
import pytest
from scripts.landing_full_command_interval_mpc_v1 import LandingFullCommandIntervalMPCV1
from scripts.landing_free_arm_reference_shooting_v1 import LandingFreeArmReferenceShootingMPCV1
from scripts.whole_body_support_qp_v1 import SupportSettings
from test_admitted_arm_support_qp_v1 import fixture


def test_native_candidate_matches_zero_search_full_command_model():
    _,interface,state=fixture()
    state.previous_command_target=state.joint_position.copy()
    old=LandingFreeArmReferenceShootingMPCV1(interface,qp_settings=SupportSettings(),arm_target_rate_rad_s=10.)
    new=LandingFullCommandIntervalMPCV1(interface,qp_settings=SupportSettings(),arm_target_rate_rad_s=10.)
    old.qp.shooting_knots=np.zeros((3,3))
    a=old.qp.solve(state,state.joint_position);b=new.qp.solve(state,state.joint_position)
    assert a['accepted'] and b['accepted']
    np.testing.assert_array_equal(a['joint_position_target'],b['joint_position_target'])
    assert not new.qp.preserve_arm_command
    assert not b['arm_feasibility_fallback_attempted']
    np.testing.assert_array_equal(new.qp.interface.velocity[12:],np.full(6,5.))
    assert new.qp.interface.arm_rate==10.
    assert not new.contract()['reference_space_search_enabled']
    with pytest.raises(ValueError):
        LandingFullCommandIntervalMPCV1(interface,preserve_arm_command=True)
