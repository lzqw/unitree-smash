import copy

import pytest
import torch

from quadsmash.control.full_racket_residual_reward_v4 import EVENT_WEIGHTS, ResidualEventRewardV4
from quadsmash.control.measured_string_alignment_reward_v4 import (
    MeasuredStringAlignmentRewardV4, alignment_settings, attack_aligned_target)

SETTINGS = dict(contact_potential_weight=5., normal_distance_scale_m=.2,
    attack_progress_weight=15., frame_penalty=5., discount=.997)


def reward(count):
    return MeasuredStringAlignmentRewardV4(count, 'cpu', SETTINGS,
        semiaxes_yz_m=[.0915, .1165], contact_plane_offset_m=.0256)


def inputs(count):
    zeros = lambda: torch.zeros(count, dtype=torch.bool)
    return dict(clock_s=torch.full((count,), 2.6),
        ball_relative_tool_m=torch.tensor([[.1256, 0., 0.]]).repeat(count, 1),
        plane_distance_rate_mps=torch.full((count,), -8.),
        ball_in_play=torch.ones(count, dtype=torch.bool), had_contact=zeros(), ground_seen=zeros(),
        body_contact=zeros(), faulted=zeros(), legal_air=zeros(), released=zeros(), frame_hit=zeros(),
        outgoing_velocity_mps=torch.full((count, 3), torch.nan), terminated=zeros())


def test_potential_telescopes_at_real_terminal_and_does_not_pay_stalling():
    r = reward(1); data = inputs(1); values = []
    for gap in [.3, .15, .05]:
        data['ball_relative_tool_m'][0, 0] = .0256+gap
        values.append(r.step(**data).item())
    assert values[0] > 0. and values[2] > 0.
    stalled = r.step(**data).item()
    assert stalled < 0.
    values.append(stalled)
    data['terminated'][0] = True
    values.append(r.step(**data).item())
    assert values[-1] < 0. and r.previous_potential.eq(0).all()
    assert sum(SETTINGS['discount']**i*value for i,value in enumerate(values)) == pytest.approx(0., abs=2e-6)


@pytest.mark.parametrize('flag,value', [('ball_in_play', False), ('had_contact', True),
    ('ground_seen', True), ('body_contact', True), ('faulted', True), ('terminated', True)])
def test_no_positive_precontact_potential_after_closed_or_invalid_rally(flag, value):
    data = inputs(1); data[flag][0] = value
    assert reward(1).step(**data).eq(0).all()


def test_receding_ball_and_outside_legal_clock_do_not_create_potential():
    data = inputs(3); data['plane_distance_rate_mps'][0] = 1.
    data['clock_s'][1:] = torch.tensor([2., 3.5])
    assert reward(3).step(**data).eq(0).all()


def test_center_guidance_uses_real_semiaxes_and_never_modifies_observed_flags():
    data = inputs(3)
    data['ball_relative_tool_m'][:, 1] = torch.tensor([0., .0915, .183])
    before = {key:value.clone() for key,value in data.items()}
    r = reward(3); result = r.step(**data)
    assert result[0] > result[1] > result[2] >= 0.
    assert result.max() <= SETTINGS['contact_potential_weight']
    for key,value in data.items():
        torch.testing.assert_close(value, before[key], equal_nan=True)
    assert not data['legal_air'].any() and not data['had_contact'].any()


def test_actual_release_progress_is_bounded_legal_and_once_only():
    data = inputs(7)
    data['had_contact'][:] = True; data['released'][:] = True; data['legal_air'][:] = True
    data['outgoing_velocity_mps'][:] = torch.tensor([8., 0., -1.])
    data['outgoing_velocity_mps'][1] = torch.tensor([6., 0., -1.])
    data['outgoing_velocity_mps'][2] = torch.tensor([20., 0., 1.])
    data['outgoing_velocity_mps'][3] = torch.tensor([-8., 0., -1.])
    data['legal_air'][4] = False; data['faulted'][5] = True; data['outgoing_velocity_mps'][6] = torch.nan
    r = reward(7); result = r.step(**data)
    assert result[0] == SETTINGS['attack_progress_weight'] and 0 < result[1] < result[0]
    assert result[2:].eq(0).all()
    assert r.step(**data).eq(0).all()
    # A release that was ineligible may not be requalified by later flags.
    data['legal_air'][4] = True; data['faulted'][5] = False
    data['outgoing_velocity_mps'][6] = torch.tensor([8., 0., -1.])
    assert r.step(**data).eq(0).all()


def test_selective_reset_and_real_job_accounting_do_not_reset_other_lanes():
    data = inputs(2); data['had_contact'][:] = True; data['frame_hit'][:] = True
    r = reward(2)
    assert r.step(**data).tolist() == [-5., -5.]
    assert r.step(**data).eq(0).all()
    r.reset(torch.tensor([0])); assert r.step(**data).tolist() == [-5., 0.]
    stats = r.statistics()
    assert stats['component_sums']['actual_frame_penalty'] == -15.
    assert stats['actual_first_frame_events'] == 3
    r.reset()
    assert r.statistics() == stats
    assert not r.seen_release.any() and not r.seen_frame.any()


def test_soft_target_hits_get_no_target_event_but_original_binary_weights_stay():
    air = torch.ones(4, dtype=torch.bool); target = air.clone()
    attack = torch.tensor([False, True, True, True]); faulted = torch.tensor([False, False, True, False])
    air[3] = False; before = [value.clone() for value in [air, target, attack, faulted]]
    event = attack_aligned_target(air, attack, target, faulted)
    assert event.tolist() == [False, True, False, False]
    flags = {name:torch.zeros(4, dtype=torch.bool) for name in EVENT_WEIGHTS}; flags['target'] = event
    paid = ResidualEventRewardV4(4, 'cpu').step(flags, torch.zeros(4), torch.zeros(4), .01)
    assert paid.tolist() == [0., 20., 0., 0.]
    assert EVENT_WEIGHTS['attack'] == 15. and EVENT_WEIGHTS['complete'] == 150.
    assert all(torch.equal(value, old) for value,old in zip([air, target, attack, faulted], before))


@pytest.mark.parametrize('change', [dict(contact_potential_weight=0), dict(discount=True),
    dict(frame_penalty=-1), dict(normal_distance_scale_m=float('nan')),
    dict(attack_progress_weight=float('inf')), dict(target_x_high=5.)])
def test_invalid_settings_and_scoring_geometry_overrides_are_rejected(change):
    value = dict(copy.deepcopy(SETTINGS), **change)
    with pytest.raises(ValueError):
        alignment_settings(value)
