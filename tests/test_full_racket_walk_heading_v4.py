import math

import numpy as np
import pytest
import torch

from quadsmash.control.full_racket_walk_heading_v4 import (
    validate_walk_heading_gain, walk_heading_arguments)


def public_orientation(roll, pitch, yaw):
    cr,sr,cp,sp,cy,sy = [f(v) for f,v in [(math.cos,roll),(math.sin,roll),
        (math.cos,pitch),(math.sin,pitch),(math.cos,yaw),(math.sin,yaw)]]
    rx=np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    ry=np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    rz=np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    rotation=rz@ry@rx
    obs=torch.zeros(794,dtype=torch.float64)
    obs[6:9]=torch.tensor(rotation.T@np.array([0.,0.,-1.]))
    obs[763:766]=torch.tensor(rotation.T@np.array([1.,0.,0.]))
    return obs


def test_heading_correction_has_the_right_sign_even_when_tilted():
    angles=[(.0,.0,.0),(.12,-.2,.35),(-.3,.23,-.8),(.4,.5,3.12),(-.4,-.5,-3.12)]
    public=torch.stack([public_orientation(*a) for a in angles])
    arguments=walk_heading_arguments(public,1.5)
    np.testing.assert_allclose(arguments['heading_error'].numpy(),[-a[2] for a in angles],atol=1e-12)
    assert torch.equal(arguments['heading_kp'],torch.full((5,),1.5,dtype=torch.float64))


def test_zero_gain_does_not_read_the_orientation_or_add_walker_arguments():
    assert walk_heading_arguments(torch.full((2,794),float('nan')),0.) == {}
    assert validate_walk_heading_gain() == 0.


def test_feedback_uses_only_current_public_gravity_and_command_direction():
    public=torch.stack([public_orientation(.1,.2,.3),public_orientation(-.2,-.1,-.4)])
    expected=walk_heading_arguments(public,2.)
    used=torch.zeros(794,dtype=torch.bool);used[6:9]=True;used[763:766]=True
    public[:,~used]=float('nan');before=public.clone()
    actual=walk_heading_arguments(public,2.)
    for key in expected:
        assert torch.equal(actual[key],expected[key])
    torch.testing.assert_close(public,before,equal_nan=True,rtol=0,atol=0)


@pytest.mark.parametrize('value',[True,False,'1.5',None,float('nan'),float('inf'),-.1,4.01])
def test_invalid_heading_gain_is_rejected(value):
    with pytest.raises(ValueError):
        validate_walk_heading_gain(value)


def test_gain_accepts_finite_bounds_without_changing_the_scalar():
    for value in [0,0.,1.5,4]:
        assert validate_walk_heading_gain(value) == float(value)
