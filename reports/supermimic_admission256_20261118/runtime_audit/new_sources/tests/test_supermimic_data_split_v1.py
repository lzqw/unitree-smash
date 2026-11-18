import pytest
from quadsmash.rl.supermimic_data_split_v1 import explicit_lane_split


def test_complete_split_preserves_whole_lane_identity():
    train, val = explicit_lane_split(dict(train_lanes=[3, 1], validation_lanes=[2, 0]), 4)
    assert train.tolist() == [1, 3] and val.tolist() == [0, 2]


@pytest.mark.parametrize('train,val', [([0, 1], [1, 2]), ([0], [1]),
    ([0, 0], [1, 2]), ([0, 1], [3]), ([True, 1], [2]), ([0.0, 1], [2]), ([], [0, 1, 2])])
def test_incomplete_ambiguous_or_leaking_split_rejected(train, val):
    with pytest.raises(ValueError):
        explicit_lane_split(dict(train_lanes=train, validation_lanes=val), 3)
