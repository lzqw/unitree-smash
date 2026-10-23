import itertools

import numpy as np
from scipy.spatial.transform import Rotation

from quadsmash.control.piper_geometry import PiperGeometry


def test_urdf_fk_matches_preexisting_native_nominal_racket_position():
    model = PiperGeometry()
    position, _ = model.forward([0, 1, -1, 0, 0, 0])
    # Measured ground reset pose in root_probe_all01/trajectory.json.
    assert np.allclose(position, [.284133315, -.00000090035, .646758080], atol=1e-6)


def test_oriented_box_support_matches_all_eight_vertices():
    model = PiperGeometry()
    rotation = Rotation.from_euler('xyz', [.7, .8, .9]).as_matrix()
    local = np.array(list(itertools.product((-1, 1), repeat=3))) * model.half_extents
    assert abs(model.vertical_support(rotation) - (local @ rotation.T)[:, 2].max()) < 1e-12
    diagonal = np.linalg.norm(model.half_extents)
    assert diagonal > .155 and diagonal > .11


def test_effective_face_contact_is_not_extreme_box_corner():
    model = PiperGeometry()
    q = [0, 1, -1, 0, 0, 0]
    center, rotation = model.forward(q)
    ball, normal = model.contact_center(q, uv=(.05, -.07), face=-1)
    local = rotation.T @ (ball - center)
    assert np.allclose(local, [-.0325, .05, -.07])
    assert np.allclose(normal, -rotation[:, 0])


def test_root_rotation_and_translation_apply_to_entire_chain():
    model = PiperGeometry()
    q = [0, 1.2, -2., .1, -.3, .4]
    p, r = model.forward(q, base_position=(0, 0, 0))
    rotation = Rotation.from_euler('xyz', [.2, -.4, .7]).as_matrix()
    actual, actual_r = model.forward(q, base_position=(1, 2, .5), base_rpy=(.2, -.4, .7))
    assert np.allclose(actual, [1, 2, .5] + rotation @ p)
    assert np.allclose(actual_r, rotation @ r)
