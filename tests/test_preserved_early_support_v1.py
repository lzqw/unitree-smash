import numpy as np
from scripts.search_preserved_early_support_v1 import preservation_passes, proposal_deltas


def test_regression_gate_requires_every_previously_successful_scenario():
    names = ['nominal', 'a', 'b', 'c', 'd']
    rows = [dict(probe=dict(name=n), result=dict(accepted=True)) for n in names]
    assert preservation_passes(rows, names)
    assert not preservation_passes(rows[:-1], names)
    rows[-1]['result']['accepted'] = False
    assert not preservation_passes(rows, names)
    rows[-1] = rows[0]
    assert not preservation_passes(rows, names)


def test_braking_proposals_follow_named_joint_not_assumed_column():
    order = ['other_' + str(i) for i in range(18)]
    order[2] = 'FL_calf_joint'; order[9] = 'FR_calf_joint'
    proposals = proposal_deltas(order, 123)
    for _, delta in proposals[:3]:
        assert delta.shape == (6, 18)
        assert np.all(delta[:3, 2] < 0)
        assert np.count_nonzero(delta) == 3
        np.testing.assert_array_equal(delta[3:], 0)
    assert np.all(proposals[3][1][:3, [2, 9]] < 0)
    for _, delta in proposals[4:]:
        assert np.any(delta[:3, 12:]) and np.any(delta[3:, 12:])
