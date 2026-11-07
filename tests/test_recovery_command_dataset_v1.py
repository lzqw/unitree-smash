import numpy as np
import pytest
from quadsmash.rl.recovery_command_dataset_v1 import recovery_eligibility,native_commands,previous_commands,compose_native_recovery_delta


def episode(n):
    public=np.zeros((n,794),dtype=np.float32);public[:,775]=np.arange(n)*.01
    return public


def test_prehit_contact_does_not_start_recovery_and_contact_loss_does_not_end_it():
    public=episode(12);public[1:3,774]=1.;public[5,773]=1.;public[7,774]=1.
    active,elapsed=recovery_eligibility(public)
    np.testing.assert_array_equal(active,np.arange(12)>=7)
    np.testing.assert_allclose(elapsed[7:],np.arange(5)*.01,atol=1e-8)


def test_future_hit_cannot_change_previous_eligibility():
    a=episode(12);a[3,774]=1.;a[10,773]=1.;a[11,774]=1.
    b=a[:8].copy()
    np.testing.assert_array_equal(recovery_eligibility(a)[0][:8],recovery_eligibility(b)[0])
    assert not recovery_eligibility(b)[0].any()


def test_reset_requires_a_new_episode_instead_of_inheriting_old_latch():
    public=episode(10);public[6:,775]-=.06
    with pytest.raises(ValueError,match='contiguous'):
        recovery_eligibility(public)


def test_previous_command_is_actual_history_not_measured_joint_position():
    limits=np.tile([-2.,3.],(18,1));actions=np.stack([np.full(18,-1.),np.full(18,.2),np.ones(18)])
    normalized,target=native_commands(actions,limits)
    previous,known=previous_commands(target)
    np.testing.assert_array_equal(previous[1:],target[:-1])
    assert np.isnan(previous[0]).all() and not known[0] and known[1:].all()
    np.testing.assert_allclose(target[:,0],[-2.,1.,3.])
    np.testing.assert_allclose(normalized,actions,atol=1e-7)


def test_bad_action_is_rejected_not_clipped_into_an_expert_label():
    with pytest.raises(ValueError,match='recorded'):
        native_commands(np.ones((2,18))*1.2,np.tile([-2.,3.],(18,1)))


def test_full_range_coordinate_preserves_prefix_and_limits_large_requested_change():
    limits=np.tile([-2.,3.],(18,1));base=np.full((2,18),-.8);previous=base.copy()
    delta=np.ones((2,18));active=np.array([False,True])
    command=compose_native_recovery_delta(base,delta,previous,active,limits)
    np.testing.assert_array_equal(command[0],base[0])
    np.testing.assert_allclose(command[1,:12],1.)
    np.testing.assert_allclose(command[1,12:]-previous[1,12:],.04)


def test_native_target_roundtrip_is_not_limited_to_old_point_four_rad_residual():
    limits=np.tile([-2.,3.],(18,1));base=np.zeros((1,18));previous=np.zeros((1,18))
    target=np.full((1,18),.9);target[:,12:]=.02
    delta=(target-base)/2.
    actual=compose_native_recovery_delta(base,delta,previous,np.ones(1,dtype=bool),limits)
    np.testing.assert_allclose(actual,target,atol=1e-12)
    assert ((actual-base)*2.5)[0,0]>.4
