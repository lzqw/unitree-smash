import pytest

from scripts.audit_supermimic_online_release_speed import compare_refresh


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
