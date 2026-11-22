import pytest

from scripts.audit_supermimic_online_release_speed import compare_refresh, impact_component_errors


def test_paired_errors_require_identical_native_labels_and_count_missed_contacts():
    def case(lane, predicted):
        return dict(lane=lane, native_tick=100, native_release_tick=101, native_latched_mps=4.8,
                    common_boundary_native_mps=4.9, sole_speed_loss=True, wm_first_outgoing_mps=predicted)
    prior = dict(cases=[case(1, 5.4), case(2, None)], world_updates=10, world_sha256='w10')
    rows = [case(2, 5.1), case(1, 4.9)]
    result = compare_refresh(prior, rows, {2})
    assert result['all']['matched_hits'] == 1
    assert result['all']['before']['common_hit_false_above'] == 1
    assert result['all']['after']['common_hit_false_above'] == 0
    assert result['excluded_lanes']['after']['predicted_hits'] == 1
    assert result['excluded_lanes']['after']['common_hit_mae_mps'] is None
    assert result['training_lanes']['after']['common_hit_mae_mps'] == pytest.approx(.1)
    rows[1]['native_latched_mps'] = 5.
    with pytest.raises(ValueError, match='Changed native label'):
        compare_refresh(prior, rows, {2})


def test_speed_clocks_use_the_same_released_predictions_and_include_observed_misses_in_support():
    from scripts.audit_supermimic_online_release_speed import release_clock_errors

    rows = [
        dict(native_latched_mps=4.9, wm_first_outgoing_mps=5.1, wm_impact_velocity_mps=[3., 4., 0.]),
        dict(native_latched_mps=5., wm_first_outgoing_mps=5.1, wm_impact_velocity_mps=[4.8, 0., 0.]),
        dict(native_latched_mps=5.2, wm_first_outgoing_mps=None, wm_impact_velocity_mps=None),
        dict(native_latched_mps=None, wm_first_outgoing_mps=8., wm_impact_velocity_mps=[8., 0., 0.]),
    ]
    values = release_clock_errors(rows)
    assert (values['first_contacts'], values['released'], values['common_predictions']) == (4, 3, 2)
    assert values['impact']['mae_mps'] == pytest.approx(.15)
    assert values['control_end']['false_below'] == 0
    assert (values['impact']['false_above'], values['impact']['false_below']) == (1, 1)
    assert release_clock_errors(rows[2:])['impact']['mae_mps'] is None


def test_impulse_terms_compare_vector_and_speed_on_identical_released_hits():
    row = dict(native_release_velocity_mps=[0., 5., 0.], wm_impact_velocity_mps=[0., 5., 0.],
               wm_impact_components_mps=dict(after_normal=[5., 0., 0.], tangent_dv=[-5., 4., 0.],
                                              residual_dv=[0., 1., 0.]))
    result = impact_component_errors([row, dict(native_release_velocity_mps=None),
                                      dict(native_release_velocity_mps=[1., 2., 3.])])
    assert (result['first_contacts'], result['released'], result['component_cases']) == (3, 2, 1)
    assert result['normal']['speed_mae_mps'] == 0.
    assert result['normal']['component_rmse_mps'] == pytest.approx((50/3)**.5)
    assert result['normal_plus_tangent']['false_below_5'] == 1
    assert result['full']['component_rmse_mps'] == 0.
    assert impact_component_errors([])['component_cases'] == 0
