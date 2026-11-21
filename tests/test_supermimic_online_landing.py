import torch

from scripts.audit_supermimic_online_landing import errors, landing_starts


def test_first_released_landing_is_unique_per_excluded_lane_and_episode():
    feet = torch.zeros(6, 2, 4)
    next_feet = feet.clone()
    next_feet[:, :, 0] = 1
    release = torch.ones(6, 2, dtype=torch.bool)
    release[0, 0] = False
    feet[1, 0, 1] = 1
    episode = torch.zeros(6, 2, dtype=torch.long)
    episode[4:, 0] = 1
    data = dict(feet=feet, next_feet=next_feet, released=release, episode=episode)
    assert landing_starts(data, {0}) == {(0, 0): 3, (0, 1): 5}


def test_landing_pose_handles_quaternion_sign_and_counts_individual_support_errors():
    truth = torch.zeros(2, 55)
    truth[:, 3] = 1
    predicted = truth.clone()
    predicted[0, 3] = -1
    predicted[1, 2] = .25
    feet = torch.zeros(2, 4, dtype=torch.bool)
    probabilities = feet.float()
    probabilities[1, 2:] = .5
    result = errors(predicted, probabilities, truth, feet)
    assert result['root_attitude_p90_rad'] == 0
    assert result['root_height_mae_m'] == .125
    assert result['foot_bit_error'] == .25 and result['support_count_mae'] == 1
