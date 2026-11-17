import torch

from scripts.train_supermimic_physical_block import select_launch_for_start


def test_compact_lane_start_selects_matching_native_launch():
    launch = torch.arange(128 * 6, dtype=torch.float32).reshape(128, 6)
    row = {'_lane_id': 37, 'before': {'physical': {'joint_position': torch.zeros(1, 18)}}}
    selected = select_launch_for_start(row, launch)
    assert selected.shape == (1, 6)
    torch.testing.assert_close(selected, launch[37:38])


def test_full_batch_start_keeps_full_launch():
    launch = torch.zeros(128, 6)
    row = {'before': {'physical': {'joint_position': torch.zeros(128, 18)}}}
    assert select_launch_for_start(row, launch) is launch
