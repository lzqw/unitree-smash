import numpy as np
import pytest

from quadsmash.physics import RacketGeometry, resolve_racket_collision, swept_sphere_plane_contact


def test_collision_reverses_incoming_normal_velocity():
    racket = RacketGeometry(np.zeros(3), np.array([1.0, 0.0, 0.0]))
    post, event = resolve_racket_collision(np.array([-4.0, 1.0, 0.0]), racket, restitution=0.75)
    assert post[0] > 0
    assert event.pre_velocity[0] == -4.0
    assert event.post_velocity[0] == post[0]


def test_swept_contact_and_boundary_checks():
    racket = RacketGeometry(np.zeros(3), np.array([1.0, 0.0, 0.0]))
    assert swept_sphere_plane_contact(np.array([-0.2, 0.0, 0.0]), np.array([0.2, 0.0, 0.0]), racket) is not None
    assert swept_sphere_plane_contact(np.array([-0.2, 0.4, 0.0]), np.array([0.2, 0.4, 0.0]), racket) is None
    with pytest.raises(ValueError):
        RacketGeometry(np.zeros(3), np.zeros(3))

