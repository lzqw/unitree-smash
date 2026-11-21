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
