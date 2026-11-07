"""Protect the landed prefix and original command map in tail search."""
from types import SimpleNamespace
import numpy as np
from scripts.search_whole_body_early_landing_v2 import WholeBodyEarlyReference
from scripts.search_landing_support_tail_v3 import SupportTailReference


def setup():
    interface=SimpleNamespace(limits=np.tile([-2.,2.],(18,1)),arm_rate=10.)
    initial=SimpleNamespace(previous_command_target=np.linspace(-.6,.6,18),joint_position=np.linspace(.5,-.5,18))
    fixed=np.random.default_rng(30).uniform(-.4,.4,(3,18))
    return interface,initial,fixed


def test_zero_tail_preserves_all_520_original_commands():
    interface,initial,fixed=setup()
    base=WholeBodyEarlyReference(None,interface,initial,fixed)
    tail=SupportTailReference(None,interface,initial,np.zeros((3,18)),fixed_knots=fixed)
    previous=initial.previous_command_target.copy()
    for step in range(520):
        expected,_=base.action(80+step,initial,previous);actual,_=tail.action(80+step,initial,previous)
        np.testing.assert_array_equal(actual,expected);previous=actual


def test_nonzero_tail_preserves_prefix_and_never_bypasses_limits():
    interface,initial,fixed=setup()
    corrections=np.array([[8.]*18,[-8.]*18,[8.]*18])
    base=WholeBodyEarlyReference(None,interface,initial,fixed)
    tail=SupportTailReference(None,interface,initial,corrections,fixed_knots=fixed)
    previous=initial.previous_command_target.copy()
    for step in range(520):
        actual,_=tail.action(80+step,initial,previous)
        if step<=80:
            expected,_=base.action(80+step,initial,previous)
            np.testing.assert_array_equal(actual,expected)
        assert np.all(actual>=interface.limits[:,0]) and np.all(actual<=interface.limits[:,1])
        assert np.max(abs(actual[12:]-previous[12:]))<=.1+1e-12
        previous=actual
    np.testing.assert_array_equal(tail.correction(.8),np.zeros(18))
    np.testing.assert_array_equal(tail.correction(1.3),corrections[0])
    np.testing.assert_array_equal(tail.correction(3.5),corrections[-1])
    assert np.max(abs(tail.correction(.8+1e-8)))<1e-12
