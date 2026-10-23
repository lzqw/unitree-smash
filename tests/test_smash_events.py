import numpy as np
import pytest
import torch

from quadsmash.physics.smash_events import OrderedSmashEvents, rank_candidate
from quadsmash.physics.smash_feasibility import predict_flights
from quadsmash.control.smash_search import add_flight_search_diagnostics, search_candidate_key


def tick(events, old, new, time, *, contact=False, touching=False, face=True, airborne=True,
         separated=None, velocity=(8., 0., -3.)):
    flag = lambda x: torch.tensor([x])
    events.advance(previous_position=torch.tensor([old], dtype=torch.float32),
                   position=torch.tensor([new], dtype=torch.float32),
                   velocity=torch.tensor([velocity]),
                   contact_event=flag(contact), touching=flag(touching),
                   airborne=flag(airborne), effective_face=flag(face),
                   ground_touch=flag(False), time=torch.tensor([time]), dt=.01,
                   separated=None if separated is None else flag(separated))


def launch(events, *, face=True, airborne=True):
    tick(events, [1.2, 0, 1.9], [1.25, 0, 1.8], 1., contact=True,
         touching=True, face=face, airborne=airborne)
    tick(events, [1.25, 0, 1.8], [1.3, 0, 1.78], 1.01)
    tick(events, [1.3, 0, 1.78], [1.4, 0, 1.75], 1.02)


def test_ordered_airborne_face_contact_crossing_first_landing_and_recovery():
    events = OrderedSmashEvents(1, "cpu")
    launch(events)
    tick(events, [1.4, 0, 1.75], [1.6, 0, 1.70], 1.03)
    tick(events, [2.8, 0, .05], [2.9, 0, .01], 1.4)
    assert events.complete(torch.tensor([True])).item()
    assert events.contact_time < events.release_time < events.crossing_time < events.landing_time
    assert not events.complete(torch.tensor([False])).item()


def test_ground_bounce_before_net_cannot_become_valid_landing():
    events = OrderedSmashEvents(1, "cpu")
    launch(events)
    tick(events, [1.4, 0, .05], [1.45, 0, .01], 1.03)
    tick(events, [1.45, 0, 2.], [1.6, 0, 2.], 1.3)
    tick(events, [2.8, 0, .05], [2.9, 0, .01], 1.8)
    assert not events.crossed.item()
    assert not events.complete(torch.tensor([True])).item()


@pytest.mark.parametrize("face,airborne", [(False, True), (True, False)])
def test_rim_or_grounded_contact_is_not_airborne_attack(face, airborne):
    events = OrderedSmashEvents(1, "cpu")
    launch(events, face=face, airborne=airborne)
    assert not events.attack().item()


def test_recontact_invalidates_same_episode_and_reset_clears_all_state():
    events = OrderedSmashEvents(1, "cpu")
    launch(events)
    tick(events, [1.4, 0, 1.75], [1.45, 0, 1.72], 1.03, contact=True, touching=True)
    assert events.invalid_multiple_contact.item()
    assert not events.attack().item()
    events.reset(torch.tensor([0]))
    assert events.contact_count.item() == 0 and not events.contact.item()
    assert torch.isnan(events.contact_time).item()


def test_crossing_around_net_side_is_rejected():
    events = OrderedSmashEvents(1, "cpu")
    launch(events)
    tick(events, [1.4, 2., 2.], [1.6, 2., 2.], 1.03)
    assert not events.crossed.item()


def test_free_flight_prediction_matches_no_drag_analytic_and_rejects_low_smash():
    result = predict_flights([[1.35, 0, 1.8], [.4, 0, 1.1]],
                             [[7., 0, -4.], [.45, -1.7, -8.2]], drag=0., dt=.001)
    t = (.025 - 1.8)  # z(t)=1.8-4t-4.905t²
    landing_t = (-4 + np.sqrt(16 - 4*4.905*t)) / (2*4.905)
    assert abs(result['landing_time'][0] - landing_t) < 1e-6
    assert abs(result['landing_position'][0, 0] - (1.35 + 7*landing_t)) < 1e-5
    assert result['predicted_legal_attack'].tolist() == [True, False]


def test_ranking_prioritizes_ordered_landing_over_raw_speed():
    good = dict(ordered_landed_target=True, ordered_crossed_net=True,
                recovery_success=False, net_margin_m=.02, outgoing_velocity_mps=[7, 0, -4])
    ground_smash = dict(ordered_landed_target=False, ordered_crossed_net=False,
                        recovery_success=True, net_margin_m=-1, outgoing_velocity_mps=[1, 0, -20])
    assert rank_candidate(good) > rank_candidate(ground_smash)


def test_numerical_collision_failure_cannot_be_ranked_as_legal_success():
    good = dict(ordered_landed_target=True, ordered_crossed_net=True,
                recovery_success=True, net_margin_m=.02, outgoing_velocity_mps=[7, 0, -4])
    numerical = dict(good, numerical_failures=1)
    assert rank_candidate(good)[:3] == (1, 1, 1)
    assert rank_candidate(numerical)[:3] == (0, 0, 0)


def test_recontact_after_landing_cannot_survive_as_complete_search_candidate():
    events = OrderedSmashEvents(1, 'cpu')
    launch(events)
    tick(events, [1.4, 0, 1.75], [1.6, 0, 1.70], 1.03)
    tick(events, [2.8, 0, .05], [2.9, 0, .01], 1.4)
    tick(events, [2.8, 0, .5], [2.9, 0, .4], 1.8, contact=True, touching=True)
    assert events.landed_target.item() and not events.complete(torch.tensor([True])).item()
    row = dict(ordered_landed_target=True, ordered_crossed_net=True,
               contact_count=2, recovery_success=True, outgoing_velocity_mps=[8, 0, -3],
               complete_smash=False)
    assert rank_candidate(row)[:3] == (0, 0, 0)


def test_outgoing_state_uses_same_last_force_frame_and_requires_geometric_separation():
    events = OrderedSmashEvents(1, 'cpu')
    tick(events, [1.2, 0, 1.9], [1.25, 0, 1.8], 1., contact=True, touching=True)
    for time in (1.01, 1.02):
        tick(events, [1.25, 0, 1.8], [1.3, 0, 1.78], time, separated=False)
    assert not events.released.item()
    tick(events, [1.3, 0, 1.78], [1.32, 0, 1.77], 1.03,
         touching=True, velocity=(9., 0., -2.))
    tick(events, [1.32, 0, 1.77], [1.34, 0, 1.76], 1.04, separated=True)
    tick(events, [1.34, 0, 1.76], [1.36, 0, 1.75], 1.05, separated=True)
    assert events.released.item()
    assert torch.allclose(events.outgoing_position, torch.tensor([[1.32, 0, 1.77]]))
    assert torch.allclose(events.outgoing_velocity, torch.tensor([[9., 0, -2.]]))
    assert events.outgoing_time.item() == pytest.approx(1.03)


def test_coupled_search_prefers_valid_downward_flight_over_higher_upward_net_margin():
    common = dict(contact_position_m=[1.4, 0, 1.65], airborne_contact=True,
                  effective_face_contact=True, contact_count=1,
                  ordered_crossed_net=False, ordered_landed_target=False,
                  recovery_success=False, interception_min_distance_m=.1)
    downward = dict(common, outgoing_velocity_mps=[7.5, 0, -3.])
    upward = dict(common, outgoing_velocity_mps=[7.5, 0, 3.])
    add_flight_search_diagnostics([downward, upward])
    assert upward['predicted_net_margin_m'] > downward['predicted_net_margin_m']
    assert downward['predicted_legal_attack'] and not upward['predicted_legal_attack']
    assert search_candidate_key(downward) > search_candidate_key(upward)
    assert not downward['ordered_landed_target']  # prediction is not a real success flag


def test_net_clearance_without_first_target_landing_is_not_zero_violation():
    row = dict(contact_position_m=[1.4, 0, 1.8], outgoing_velocity_mps=[14., 0, -1.],
               airborne_contact=True, effective_face_contact=True, contact_count=1)
    add_flight_search_diagnostics([row])
    assert row['predicted_net_margin_m'] > 0
    assert row['predicted_first_landing_position_m'][0] > 4
    assert row['coupled_flight_violation'] > 0 and not row['predicted_legal_attack']


def test_native_inference_rollout_buffers_can_be_reset_outside_inference_mode():
    events = OrderedSmashEvents(1, 'cpu')
    with torch.inference_mode():
        launch(events)
    events.reset(torch.tensor([0]))
    assert not events.contact.item() and events.quiet_frames.item() == 0
