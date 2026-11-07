"""Compatibility and actual command-interface checks for the extended search."""
from types import SimpleNamespace
import numpy as np
import pytest
from scripts.search_early_landing_trajectory_v1 import EarlyReference
from scripts.search_whole_body_early_landing_v2 import WholeBodyEarlyReference


def fixture():
    interface=SimpleNamespace(limits=np.tile([-2.,2.],(18,1)),arm_rate=10.)
    initial=SimpleNamespace(previous_command_target=np.linspace(-.9,.9,18),
                            joint_position=np.linspace(.7,-.7,18))
    return interface,initial


def test_zero_arm_offsets_preserve_legacy_commands_across_all_phases():
    interface,initial=fixture();rng=np.random.default_rng(41)
    legs=rng.uniform(-1.8,1.8,(3,12));full=np.c_[legs,np.zeros((3,6))]
    old=EarlyReference(None,interface,initial,legs)
    new=WholeBodyEarlyReference(None,interface,initial,full)
    previous=initial.previous_command_target.copy()
    for step in range(500):
        before,_=old.action(80+step,initial,previous)
        after,_=new.action(80+step,initial,previous)
        np.testing.assert_array_equal(before,after)
        previous=after


def test_large_joint_coupled_candidates_keep_first_dispatch_limits_and_arm_slew():
    interface,initial=fixture()
    knots=np.array([[8.]*18,[-8.]*18,[8.]*18])
    policy=WholeBodyEarlyReference(None,interface,initial,knots)
    previous=initial.previous_command_target.copy()
    first,_=policy.action(80,initial,previous)
    np.testing.assert_array_equal(first,previous)
    for step in range(520):
        action,_=policy.action(80+step,initial,previous)
        assert np.max(abs(action[12:]-previous[12:]))<=.1+1e-12
        assert np.all(action>=interface.limits[:,0])
        assert np.all(action<=interface.limits[:,1])
        previous=action
    with pytest.raises(ValueError):policy.action(600,initial,previous)


def test_nonfinite_knots_rejected_before_any_command():
    interface,initial=fixture();knots=np.zeros((3,18));knots[1,16]=np.nan
    with pytest.raises(ValueError):WholeBodyEarlyReference(None,interface,initial,knots)
