import numpy as np
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.full_authority_position_interface_v1 import FullAuthorityPositionPDInterfaceV1,FullAuthorityCommandArbiterV1


def test_target_authority_is_distinct_from_actual_speed_and_effort():
    source=PositionPDInterface(np.tile([-2.,2.],(18,1)),np.full(18,80.),np.full(18,4.),
        np.full(18,100.),np.full(18,5.),.65)
    interface=FullAuthorityPositionPDInterfaceV1(source)
    interface.bind_previous_command(np.zeros(18))
    low,high=interface.admitted_target_bounds(np.zeros(18),.01)
    assert np.all(low==-2.) and np.all(high==2.)
    assert np.array_equal(interface.velocity,source.velocity)
    assert np.array_equal(interface.effort,source.effort)
    torque_low,torque_high=interface.torque_bounds(np.zeros(18),np.zeros(18),.01)
    assert np.all(torque_low==-100.) and np.all(torque_high==100.)
    target,record=FullAuthorityCommandArbiterV1(source.limits).select(None,np.ones(18),np.zeros(18),owns_recovery=True)
    assert np.all(target==1.) and record['software_command_slew_limit'] is None


def test_arbitration_preserves_physical_position_bounds_and_one_owner():
    limits=np.tile([-1.,1.],(18,1));arbiter=FullAuthorityCommandArbiterV1(limits)
    target,record=arbiter.select(np.full(18,-.5),np.full(18,3.),np.zeros(18),owns_recovery=True)
    assert np.all(target==1.) and record['owner']=='controller'
    target,record=arbiter.select(np.full(18,-.5),np.full(18,3.),np.zeros(18),owns_recovery=False)
    assert np.all(target==-.5) and record['owner']=='frozen_prefix'
