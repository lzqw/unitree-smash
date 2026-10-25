import numpy as np
import pytest

from quadsmash.control.full_racket_contact_uv_retarget_v4 import retarget_contact_uv


Q=np.array([[-.615,1.77,-2.288,.871,.883,2.09],[-.605,1.76,-2.28,.86,.88,2.08]])


def test_unchanged_control_point_preserves_the_exact_zero_offset():
    before=Q.copy();r=retarget_contact_uv(Q,[0,.1],[0,.1],np.full(6,-.15),np.full(6,.15))
    np.testing.assert_array_equal(r['delta_rad'],np.zeros(6))
    np.testing.assert_array_equal(Q,before)


def test_inward_retarget_reduces_position_error_with_bounded_offsets():
    before=Q.copy();r=retarget_contact_uv(Q,[0,.1],[0,.07],np.full(6,-.15),np.full(6,.15))
    np.testing.assert_allclose(r['position_error_before_m'],.03,atol=1e-12)
    assert r['position_error_after_m'].mean()<.03
    assert (abs(r['delta_rad'])<=.15).all()
    assert np.isfinite(r['normal_error_after_rad']).all()
    np.testing.assert_array_equal(Q,before)


@pytest.mark.parametrize('uv',[[0,.2],[float('nan'),0],[0]])
def test_retarget_rejects_points_outside_the_real_string_region(uv):
    with pytest.raises(ValueError):
        retarget_contact_uv(Q,[0,.1],uv,np.full(6,-.15),np.full(6,.15))


def test_malformed_configurations_and_bounds_are_rejected():
    for q in [np.zeros((2,5)),np.full((2,6),float('nan')),np.zeros((0,6))]:
        with pytest.raises(ValueError):
            retarget_contact_uv(q,[0,.1],[0,.07],np.full(6,-.15),np.full(6,.15))
    with pytest.raises(ValueError):
        retarget_contact_uv(Q,[0,.1],[0,.07],np.ones(6),np.zeros(6))
