import torch

from scripts.fit_supermimic_body_pose_integration import consistent_pose


def test_zero_weights_preserve_each_learned_channel_and_gradient():
    state = torch.randn(3, 55)
    prediction = torch.randn_like(state, requires_grad=True)
    pose = consistent_pose(state, prediction, .01, 0., 0., 0.)
    torch.testing.assert_close(pose, prediction, atol=0, rtol=0)
    pose.sum().backward()
    torch.testing.assert_close(prediction.grad, torch.ones_like(prediction))


def test_constant_velocity_integration_uses_seconds_and_only_selected_coordinates():
    state = torch.zeros(1, 55)
    state[:, 3] = 1
    state[:, 7:10] = 2
    state[:, 43:49] = -3
    target = state.clone()
    target[:, 13:25] = 11
    target[:, 3:7] *= -1
    for dt in (.01, .02):
        got = consistent_pose(state, target, dt, 1., 1., 1.)
        expected = target.clone()
        expected[:, :3] += dt*2
        expected[:, 25:31] -= dt*3
        torch.testing.assert_close(got, expected)
        arm_only = consistent_pose(state, target, dt, 0., 0., 1.)
        torch.testing.assert_close(arm_only[:, :25], target[:, :25])
        torch.testing.assert_close(arm_only[:, 25:], expected[:, 25:])
