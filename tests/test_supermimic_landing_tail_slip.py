import torch

from scripts.audit_supermimic_landing_tail_value import relative_order_loss
from scripts.audit_supermimic_friction_reversal_support import compare
from scripts.fit_supermimic_contact_separation_slip import SeparationSlip, onset_separation


def test_rtg_order_ignores_ties_and_rewards_correct_ranking():
    values = torch.tensor([0., 2., 2.], requires_grad=True)
    assert relative_order_loss(values, values.detach()) < relative_order_loss(-values, values.detach())
    relative_order_loss(values, torch.ones_like(values)).backward()
    assert torch.equal(values.grad, torch.zeros_like(values))


def test_predicted_separation_has_physical_units_and_filters_nonclosing_contacts():
    row = dict(time=torch.tensor([.0004, .0004]), normal_speed=torch.tensor([-2., 1.]))
    assert torch.allclose(onset_separation(row, .0005, .0001), torch.tensor([.0004, 0.]), atol=1.e-8)
    row.update(normal=torch.tensor([[1.,0.,0.]]*2), relative_velocity=torch.tensor([[-2.,1.,0.]]*2),
               impulse_dv=torch.tensor([[3.,0.,0.]]*2), active=torch.ones(2,dtype=torch.bool), object_id=torch.tensor([0,1]))
    out = SeparationSlip(.32,2/7,0.,.0005,.0001).toi_response(row, torch.ones(2))
    torch.testing.assert_close(out['impulse_residual_dv'],torch.tensor([[0.,-2/7,0.],[0.,0.,0.]]))


def test_fixture_switch_must_exceed_noise_on_both_nominal_repeats():
    def branch(v):
        return dict(every_anchor_reversed=v<0,any_anchor_reversed=v<0,min_signed_mps=v,max_signed_mps=v)
    row=dict(seed=1,lane=2,kind='racket_stable',above_repeat_noise=True,minus=branch(-.1),plus=branch(.2),
             nominal_a=branch(.2),nominal_b=branch(.14))
    assert 'switch_beyond_nominal_slip_jitter' not in compare([row])['counts']
    row['nominal_b']=branch(.19)
    assert compare([row])['counts']['switch_both_branches_over_0p05_mps']==1


def test_training_launches_never_enter_original_policy_panel():
    from scripts.declare_supermimic_contact_training_cases import case_digest, resolve_probe_cases
    import pytest
    def case(i):
        return dict(case_id=str(i),launch=dict(position_m=[float(i),0,1],velocity_mps=[0,1,0]))
    cases = [case(i) for i in range(128)]
    original = [case(300)]
    spec = dict(contact_training_panel=dict(cases=cases,case_sha256=case_digest(cases)),
                data_role='contact_model_training', capture_substeps=True,capture_wm_window=False)
    assert resolve_probe_cases({},original)==(original,None)
    assert resolve_probe_cases(spec,original)==(cases,case_digest(cases))
    for bad in (dict(spec,capture_wm_window=True),dict(spec,data_role='training_action_identification')):
        with pytest.raises(ValueError):
            resolve_probe_cases(bad,original)
    with pytest.raises(ValueError):
        resolve_probe_cases(spec,[case(7)])
