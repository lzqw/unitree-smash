import torch

from scripts.audit_supermimic_recovery_critic import ranking, select_landing_rows


def test_post_landing_input_is_strictly_after_the_native_support_transition():
    assert select_landing_rows([100, 101], 100, 32) == {'before_support':100, 'after_support':101}
    assert select_landing_rows([67, 133], 100, 32) == {'before_support':None, 'after_support':None}
    assert select_landing_rows([68, 132], 100, 32) == {'before_support':68, 'after_support':132}


def test_equal_value_scores_count_as_half_and_single_class_is_unscored():
    scores = torch.tensor([1., 2., 2., 4.])
    success = torch.tensor([False, True, False, True])
    assert ranking(scores, success) == .875
    assert ranking(scores, ~success) == .125
    assert ranking(scores, torch.ones_like(success)) is None


def test_incident_twist_never_reads_a_velocity_after_the_first_native_force():
    from scripts.audit_supermimic_measured_impact_twist import incident_index
    frame, predicted = incident_index(torch.tensor([.0041,.0091]), 1, .001, 10, torch.tensor([20,17]))
    assert predicted.tolist() == [14,19]
    assert frame.tolist() == [14,17]


def test_partial_slip_is_bounded_between_zero_and_the_hard_friction_response():
    from scripts.fit_supermimic_partial_slip import PartialSlip
    row = dict(normal=torch.tensor([[0., 0., 1.]]).expand(3, -1),
               relative_velocity=torch.tensor([[1., 0., -1.]]).expand(3, -1),
               impulse_dv=torch.tensor([[0., 0., 1.],[0., 0., 0.],[0., 0., 1.]]),
               active=torch.tensor([True, True, True]), object_id=torch.tensor([0,0,2]))
    present = torch.ones(3,dtype=torch.bool)
    hard = PartialSlip(.5,.5,None).toi_response(row,present)['impulse_residual_dv']
    smooth = PartialSlip(.5,.5,2).toi_response(row,present)['impulse_residual_dv']
    torch.testing.assert_close(hard[0],torch.tensor([-.5,0.,0.]))
    torch.testing.assert_close(smooth[0],hard[0]/2**.5)
    torch.testing.assert_close(smooth[1:],torch.zeros(2,3))
