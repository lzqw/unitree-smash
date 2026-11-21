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


def test_per_control_contact_support_counts_only_loaded_intervals():
    from scripts.fit_supermimic_native_racket_toi_head import control_event_groups
    from quadsmash.rl.supermimic_native_pair_capture_v1 import BRANCHES

    free = (False, False, False, (), ())
    hit = (False, False, True, (4,), ((1, 1),))
    late = (False, False, True, (5,), ((1, 1),))
    runs = {b: [[free, free], [hit, hit], [free, free]] for b in BRANCHES}
    runs['plus'][1][1] = late
    assert control_event_groups(runs, BRANCHES) == [
        'no_contact', 'no_contact', 'racket_stable', 'toi_switch', 'no_contact', 'no_contact']


def test_zero_residual_reproduces_calibrated_slip_and_gradient_only_on_racket():
    from scripts.calibrate_supermimic_toi_slip import TOISlip

    class ProbeHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.coefficients = torch.nn.Parameter(torch.zeros(3, 3))

        def toi_response(self, row, present):
            return dict(impulse_residual_dv=self.coefficients)

    normal = torch.tensor([[0., 0., 1.]]).expand(3, -1)
    row = dict(normal=normal, relative_velocity=torch.ones(3, 3), impulse_dv=normal,
               active=torch.tensor([True, True, False]), object_id=torch.tensor([0, 1, 0]))
    present = torch.ones(3, dtype=torch.bool)
    base = TOISlip(.24, .25)
    head = ProbeHead()
    result = TOISlip(.24, .25, residual=head).toi_response(row, present)['impulse_residual_dv']
    torch.testing.assert_close(result, base.toi_response(row, present)['impulse_residual_dv'], rtol=0, atol=0)
    result.sum().backward()
    torch.testing.assert_close(head.coefficients.grad, torch.tensor([[1., 1., 1.], [0., 0., 0.], [0., 0., 0.]]))


def test_window_force_balance_uses_filtered_friction_and_excludes_ground():
    from scripts.audit_supermimic_nominal_contact_repeat import window_force_balance

    dt = .01
    physics = dict(native_physics_dt_s=dt, mass_kg=.01, wind_m_s=[0, 0, 0],
                   quadratic_drag_k_per_m=0., contact_force_threshold_n=.001)
    state = torch.zeros(2, 2, 13, dtype=torch.float64)
    state[1, 0, 7:10] = torch.tensor([1., -.2, -9.81*dt], dtype=state.dtype)
    state[1, 1, 7:10] = 100.
    normal = torch.zeros(2, 2, 3, dtype=state.dtype)
    normal[1, :, 0] = 1.
    friction = torch.zeros_like(normal)
    friction[1, :, 1] = -.2
    ground = torch.zeros_like(normal)
    ground[1, 1, 2] = 1.
    trace = dict(physics_dt_s=dt, contact_metadata=dict(friction_source='physx_filtered_racket_pair'),
                 tensors=dict(shuttle_state_w=state, racket_force_w=normal, racket_friction_force_w=friction,
                              ground_force_w=ground, net_force_matrix_w=torch.zeros(2, 2, 1, 3)))
    result = window_force_balance([trace], physics)
    assert result['isolated_loaded_lane_ticks'] == 1
    assert result['normal_and_friction_rmse_mps'] < 1e-12 < result['normal_only_rmse_mps']
