import copy

import pytest
import torch

from quadsmash.control.measured_joint_placement_reward_v4 import (
    MeasuredJointPlacementRewardV4, placement_settings)


COURT = dict(target_x_limits_m=[2., 4.], target_y_half_width_m=1.)
SETTINGS = dict(weight=20., distance_scale_m=.25)


def inputs(count):
    return dict(ground_seen=torch.ones(count, dtype=torch.bool),
        legal_air=torch.ones(count, dtype=torch.bool), attack=torch.ones(count, dtype=torch.bool),
        crossed=torch.ones(count, dtype=torch.bool), faulted=torch.zeros(count, dtype=torch.bool),
        first_landing=torch.tensor([[3., 0., .025]]).repeat(count, 1))


def test_actual_attack_required_and_all_original_faults_block_shaping():
    data = inputs(7)
    data['legal_air'][1] = False
    data['attack'][2] = False  # A slow hit in the target earns no joint bonus.
    data['crossed'][3] = False
    data['faulted'][4] = True
    data['ground_seen'][5] = False
    data['first_landing'][6] = torch.nan
    reward = MeasuredJointPlacementRewardV4(7, 'cpu', COURT, SETTINGS).step(**data)
    torch.testing.assert_close(reward, torch.tensor([20., 0., 0., 0., 0., 0., 0.]))


def test_first_landing_once_and_post_rally_changes_cannot_requalify():
    r = MeasuredJointPlacementRewardV4(2, 'cpu', COURT, SETTINGS)
    data = inputs(2)
    data['attack'][1] = False
    assert r.step(**data).tolist() == [20., 0.]
    data['attack'][1] = True
    assert r.step(**data).eq(0).all()
    r.reset(torch.tensor([1]))
    assert r.step(**data).tolist() == [0., 20.]
    r.reset()
    assert r.step(**data).tolist() == [20., 20.]


def test_measured_distance_is_symmetric_and_does_not_change_target_flags():
    court = copy.deepcopy(COURT)
    data = inputs(6)
    data['first_landing'][:, :2] = torch.tensor([
        [4., 0.], [4.25, 0.], [1.75, 0.], [3., 1.25], [3., -1.25], [4.25, 1.25]])
    before = {k: v.clone() for k, v in data.items()}
    value = MeasuredJointPlacementRewardV4(6, 'cpu', court, SETTINGS).step(**data)
    assert value[0] == 20
    torch.testing.assert_close(value[1:5], torch.full((4,), 20 / torch.e))
    torch.testing.assert_close(value[5], torch.tensor(20 / torch.e**2))
    assert court == COURT
    assert all(torch.equal(v, before[k]) for k, v in data.items())


@pytest.mark.parametrize('bad', [dict(weight=0, distance_scale_m=.25),
    dict(weight=True, distance_scale_m=.25), dict(weight=float('nan'), distance_scale_m=.25),
    dict(weight=20, distance_scale_m=0), dict(weight=20, distance_scale_m=1e-300),
    dict(weight=20, distance_scale_m=float('inf')), dict(weight=20),
    dict(weight=20, distance_scale_m=.25, target_x_high=5)])
def test_invalid_or_geometry_overriding_settings_are_rejected(bad):
    with pytest.raises(ValueError):
        placement_settings(bad)
