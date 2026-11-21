import torch

from quadsmash.rl.supermimic_ball_impulse_prior_v1 import BallImpulsePriorV1
from quadsmash.rl.supermimic_action_ball_v1 import TemporalCausalRacketModeImpulseResidualBallV13 as V13
from scripts.fit_supermimic_native_racket_toi_head import prepare, predicted_state


def test_zero_v13_matches_analytic_and_cannot_change_a_free_flight():
    prior=BallImpulsePriorV1().eval()
    ball=torch.tensor([[0.,0.,2.,0.,0.,-1.]])
    position=torch.tensor([[[0.,0.,0.]],[[0.,0.,0.]]])
    quaternion=torch.tensor([[[1.,0.,0.,0.]],[[1.,0.,0.,0.]]])
    stream=[position,quaternion,torch.zeros_like(position),torch.zeros_like(position)]
    data=prepare(prior,ball,ball[:,3:],stream)
    assert not(data['active'] & (data['object_id']==0)).any()
    model=V13()
    with torch.no_grad():
        model.impulse_residual_head.net[-1].bias.fill_(10.)
    predicted=predicted_state(model,prior,data)
    torch.testing.assert_close(predicted,torch.cat((data['analytic_position'],data['analytic']),1),rtol=0,atol=0)


def test_toi_slip_opposes_surface_slip_with_coulomb_bound_on_racket_only():
    from scripts.calibrate_supermimic_toi_slip import TOISlip
    normal = torch.tensor([[0., 0., 1.]]).expand(5, -1)
    row = dict(normal=normal, relative_velocity=torch.tensor([[2., 0., -3.]]).expand(5, -1),
               impulse_dv=normal * 2., object_id=torch.tensor([0, 1, 2, 0, 0]),
               active=torch.tensor([True, True, True, False, True]))
    present = torch.tensor([True, True, True, True, False])
    impulse = TOISlip(.1, .25).toi_response(row, present)['impulse_residual_dv']
    torch.testing.assert_close(impulse[0], torch.tensor([-.2, 0., 0.]))
    assert torch.count_nonzero(impulse[1:]) == 0
