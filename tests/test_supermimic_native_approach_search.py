import pytest
import torch

from quadsmash.rl.supermimic_native_approach_search import (
    ARM_BIAS, actor_with_bias, candidate_layout, native_fitness, rank_key, validate_offsets,
)


def test_native_search_changes_only_six_biases_and_pairs_the_same_cases():
    anchor = {ARM_BIAS: torch.ones(18), 'frozen': torch.arange(3)}
    offsets = validate_offsets([[0.] * 6, [.004] * 6], maximum=4)
    result = actor_with_bias(anchor, offsets[1])
    torch.testing.assert_close(result[ARM_BIAS][:12], anchor[ARM_BIAS][:12])
    torch.testing.assert_close(result[ARM_BIAS][-6:], torch.full((6,), 1.004))
    torch.testing.assert_close(anchor[ARM_BIAS], torch.ones(18))
    assert result['frozen'] is anchor['frozen']
    cases = list(range(128))
    odd = list(range(1, 64, 2))
    assert candidate_layout(cases, odd, offsets) == odd * 2
    with pytest.raises(ValueError, match='even TRAIN64'):
        candidate_layout(cases, list(range(32)), offsets)


def test_native_rank_prioritizes_actual_return_and_preserves_signed_margins():
    court = dict(target_x_limits_m=(2., 4.), target_y_half_width_m=1.)
    record = dict(case_id='A', numerical_failures=0, qualified_airborne_string_contact=True,
                  robot_net_fault=False, frame_hit=False, string_hit=True, landed_target=False,
                  shuttle_net_contact=False, released=True, crossed_net=True,
                  outgoing_velocity_mps=[4.4, 0., 0.], first_landing_m=[3., 1.04, .025],
                  net_margin_m=.01, first_approach_contact_motion={'edge': -.002},
                  ball_side_complete_except_recovery=False)
    margin = lambda motion: motion['edge']
    partial = native_fitness([record], court, margin)
    assert partial['rows'][0]['low_speed']
    assert partial['rows'][0]['landing_margin_m'] == pytest.approx(-.04)
    assert partial['rows'][0]['string_margin_m'] == pytest.approx(-.002)
    complete = native_fitness([dict(record, ball_side_complete_except_recovery=True)], court, margin)
    assert rank_key(complete) > rank_key(partial)
