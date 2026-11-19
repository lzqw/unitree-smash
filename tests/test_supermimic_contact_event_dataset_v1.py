import torch

from quadsmash.rl.supermimic_contact_event_dataset_v1 import (
    _event_edges,
    _episode_safe_indices,
    recommended_window,
)


def test_event_edge_is_transition_start_not_latched_next_frame():
    event = torch.zeros(5, 2, 8)
    event[2, 1, 1] = 1.0
    assert _event_edges(event, 1) == [(1, 1)]


def test_episode_safe_window_masks_reset_boundary():
    episode = torch.zeros(8, 1, dtype=torch.long)
    episode[5:, 0] = 1
    ticks, valid = _episode_safe_indices(episode, 0, 5, 2, 2)
    assert ticks.tolist() == [3, 4, 5, 6, 7]
    assert valid.tolist() == [False, False, True, True, True]


def test_recommended_window_matches_control_and_imagination_horizons():
    window = recommended_window(control_dt_s=0.01)
    assert window["pre_control_ticks"] == 4
    assert window["post_control_ticks"] == 64
    assert window["pre_window_s"] == 0.04
    assert window["post_window_s"] == 0.64
