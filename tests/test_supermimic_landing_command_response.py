import pytest
import torch

from scripts.audit_supermimic_landing_command_response import response


def test_landing_response_keeps_noise_and_zero_predictions_visible():
    truth = torch.tensor([[1.,0.], [0.,0.], [0.,2.], [1.,0.]])
    model = torch.tensor([[-1.,0.], [0.,1.], [0.,0.], [9.,0.]])
    noise = torch.tensor([[0.,0.], [.1,0.], [0.,0.], [0.,0.]])
    result = response(model, truth, noise, torch.tensor([True,True,True,False]))
    assert result['eligible']==3 and result['above_repeat_noise']==2 and result['zero_predictions']==1
    assert result['gain_median']==.5 and result['cosine_median']==-.5
    assert result['absolute_component_rmse']==pytest.approx((9/6)**.5)
    assert result['zero_component_rmse']==pytest.approx((5/6)**.5)
    assert response(model, truth, noise, torch.zeros(4,dtype=torch.bool))==dict(eligible=0,above_repeat_noise=0)


def test_native_first_support_censors_and_keeps_event_misses():
    from scripts.audit_supermimic_landing_command_response import first_support
    initial=torch.tensor([[1],[0],[0],[0],[0]])
    predicted=torch.tensor([[1,0,0,1,0],[1,1,0,1,0]])[:,:,None].float()
    observed=torch.tensor([[0,1,1,0,0],[0,1,1,0,0]])[:,:,None]
    result=first_support(initial,predicted,observed)
    assert result==dict(initially_airborne=4,native_observed=2,predicted_observed=2,both_observed=1,
                        missed=1,spurious=1,jointly_censored=1,signed_tick_errors=[1],mean_abs_ticks=1.)
