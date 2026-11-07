import numpy as np
import pytest
from scripts.condensed_implicit_drive_v1 import interval_constraints as original
from scripts.batched_interval_constraints_v1 import interval_constraints as batched


@pytest.mark.parametrize('feet', range(5))
@pytest.mark.parametrize('substeps', [1, 16])
def test_every_original_inequality_and_label_preserved(feet, substeps):
    rng = np.random.default_rng(841 + feet)
    selection = np.r_[np.zeros((6, 18)), np.eye(18)]
    stages = []
    for _ in range(substeps):
        s = {}
        for name, size in [('torque', 18), ('velocity', 24),
                           ('joint_position', 18), ('force', 3*feet)]:
            s[name+'_map'] = rng.normal(size=(size, 18))
            s[name+'_bias'] = rng.normal(size=size)
        stages.append(s)
    kwargs = dict(effort=np.arange(18)+10., velocity_limit=np.arange(18)+1.,
                  joint_limits=np.c_[-np.arange(18)-1., np.arange(18)+2.], friction=.65)
    a, b = original(stages, selection, **kwargs), batched(stages, selection, **kwargs)
    assert a['labels'] == b['labels']
    for field in ('matrix', 'low', 'high'):
        np.testing.assert_allclose(a[field], b[field], atol=1e-12, rtol=0.)
    # A held command must have the same residual against every physical bound.
    command = rng.normal(size=18)
    for side in ('low', 'high'):
        np.testing.assert_allclose(a['matrix']@command-a[side],
                                   b['matrix']@command-b[side], atol=1e-12, rtol=0.)


def test_rejects_contact_dimension_changes_inside_interval():
    stages = []
    for feet in (1, 2):
        s = {}
        for name, size in [('torque', 18), ('velocity', 24),
                           ('joint_position', 18), ('force', 3*feet)]:
            s[name+'_map'] = np.zeros((size, 18))
            s[name+'_bias'] = np.zeros(size)
        stages.append(s)
    with pytest.raises(ValueError, match='Contact dimension'):
        batched(stages, np.r_[np.zeros((6, 18)), np.eye(18)], effort=np.ones(18),
                velocity_limit=np.ones(18), joint_limits=np.c_[-np.ones(18), np.ones(18)], friction=.65)
