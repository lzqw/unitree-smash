import numpy as np
import pytest

from scripts.touchdown_contact_state_v1 import TouchdownContactSettingsV1,TouchdownContactStateV1


def test_load_dip_is_latched_then_released_after_persistent_unload():
    state = TouchdownContactStateV1(TouchdownContactSettingsV1(5., 1., 3))
    geometry = np.array([True, False, False, False])
    assert state.update(geometry, np.array([[8., 0., 6.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]))[0]
    for _ in range(2):
        assert state.update(geometry, np.array([[0., 0., .5], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]))[0]
    assert not state.update(geometry, np.zeros((4, 3)))[0]


def test_geometry_loss_releases_immediately():
    state = TouchdownContactStateV1()
    assert state.update([True, False, False, False], np.array([[0., 0., 8.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]))[0]
    assert not state.update([False, False, False, False], np.zeros((4, 3)))[0]


def test_invalid_release_threshold_is_rejected():
    with pytest.raises(ValueError):
        TouchdownContactSettingsV1(1., 2., 3)
