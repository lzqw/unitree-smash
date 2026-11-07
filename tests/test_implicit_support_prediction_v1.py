import numpy as np
from scripts.implicit_support_prediction_v1 import implicit_velocity_step


def scalar(m,v,kp,kd,target,effort,dt):
    return implicit_velocity_step(np.array([[m]]),np.array([v]),np.zeros(1),np.ones((1,1)),
        np.array([kp]),np.array([kd]),np.array([target]),np.array([effort]),dt,
        np.zeros((0,1)),np.zeros(0))


def test_implicit_scalar_closed_form():
    r=scalar(.2,.3,80.,4.,.1,100.,.01)
    expected=(.2*.3+.01*80.*.1)/(.2+.01*4.+.01**2*80.)
    np.testing.assert_allclose(r['velocity'],[expected],atol=1e-12)


def test_saturated_motor_uses_bounded_force():
    r=scalar(1.,0.,80.,4.,10.,2.,.01)
    np.testing.assert_allclose(r['torque'],[2.],atol=1e-12)
    np.testing.assert_allclose(r['velocity'],[.02],atol=1e-12)


def test_small_inertia_damping_does_not_flip_to_large_velocity():
    r=scalar(1e-5,1.,80.,4.,0.,100.,.01)
    assert 0.<r['velocity'][0]<1.
    assert r['residual']<1e-9


def test_contact_reaction_balances_drive_without_motion():
    r=implicit_velocity_step(np.eye(2),np.zeros(2),np.array([0.,9.81]),
        np.array([[1.],[0.]]),np.array([80.]),np.array([4.]),np.array([.1]),
        np.array([100.]),.01,np.array([[0.,1.]]),np.zeros(1))
    assert abs(r['velocity'][1])<1e-12
    np.testing.assert_allclose(r['force'],[9.81],atol=1e-12)
