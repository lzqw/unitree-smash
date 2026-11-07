"""Equation-level comparisons; not native-control validation."""
import numpy as np
import pytest
from scripts.frictional_impulse_step_v1 import metric_disk_minimum, coulomb_impulses, frictional_velocity_step
from scripts.cached_frictional_impulse_v1 import disk_minimum_cached, coulomb_impulses_cached, frictional_velocity_step_cached


def test_anisotropic_disk_stick_slip_and_zero_radius():
    rng=np.random.default_rng(882108)
    for _ in range(100):
        angle=rng.uniform(-np.pi,np.pi)
        R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        matrix=R@np.diag(10.**rng.uniform(-4,3,2))@R.T
        bias=rng.normal(size=2);values,vectors=np.linalg.eigh(matrix)
        for radius in [0.,1e-5,.1,100.]:
            np.testing.assert_allclose(disk_minimum_cached(values,vectors,bias,radius),
                                       metric_disk_minimum(matrix,bias,radius),rtol=1e-12,atol=1e-12)


def test_four_contact_stick_slip_release_and_warm_start():
    rng=np.random.default_rng(882109)
    for _ in range(12):
        a=rng.normal(size=(12,12))*.1;matrix=a@a.T+np.eye(12)
        bias=rng.normal(size=12);warm=rng.normal(size=12)
        for mu in [0.,.65]:
            expected,info=coulomb_impulses(matrix,bias,mu,warm=warm)
            actual,details=coulomb_impulses_cached(matrix,bias,mu,warm=warm)
            np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
            assert info['iterations']==details['iterations']
            assert max(details['residual'].values())<=2e-7


def test_motor_saturation_still_solves_full_coupled_momentum():
    mass=np.array([[2.,.2],[.2,1.]])
    arguments=(mass,np.array([-.2,.1]),np.array([1.,.5]),np.array([[0.],[1.]]),
        np.array([25.]),np.array([.5]),np.array([1.]),np.array([.2]),.001,
        np.array([[0.,1.],[.1,0.],[1.,0.]]),np.array([0.]),.65)
    expected=frictional_velocity_step(*arguments)
    actual=frictional_velocity_step_cached(*arguments)
    for key in ['velocity','impulse','torque']:
        np.testing.assert_allclose(actual[key],expected[key],atol=1e-12,rtol=1e-12)
    assert actual['motor_iterations']==expected['motor_iterations']
    assert actual['momentum_residual']<=1e-8


def test_invalid_or_nonconverged_contact_not_accepted():
    for function in [coulomb_impulses,coulomb_impulses_cached]:
        with pytest.raises(ValueError):function(-np.eye(3),np.zeros(3),.65)
        with pytest.raises(ValueError):function(np.eye(3),np.zeros(3),.65,tolerance=.1)
    # An unused tangent block is not evaluated by the reference implementation.
    matrix=np.array([[1.,2.,0.],[2.,1.,0.],[0.,0.,1.]])
    expected,_=coulomb_impulses(matrix,np.array([0.,0.,1.]),.65)
    actual,_=coulomb_impulses_cached(matrix,np.array([0.,0.,1.]),.65)
    np.testing.assert_array_equal(actual,expected)
