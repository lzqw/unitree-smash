from scripts.watch_supermimic_native_return import ReturnTrend


def case(i, value, episode=0):
    return dict(case_id=str(i), env_id=i, episode_index=episode, native_episode_return=value,
                scoring_schema='v1', body_ground_seen=False, qualified_airborne_string_contact=True,
                ball_side_complete_except_recovery=True, complete5=True)


def test_completed_collections_require_worsening_relative_to_matched_cases():
    baseline = {(str(i),i):case(i,0 if i<4 else -10) for i in range(8)}
    trend=ReturnTrend(baseline)
    assert trend.observe([case(i,-1) for i in range(3)],100) is None
    assert trend.observe([case(3,-1)],200)['reason'] is None
    assert trend.observe([case(i,-10) for i in range(4,8)],300)['reason'] is None
    trend=ReturnTrend(baseline)
    assert trend.observe([case(i,-1) for i in range(4)],100)['reason'] is None
    assert trend.observe([case(i,-13) for i in range(4,8)],200)['reason']=='paired_first_episode_loss_worsened'


def test_later_episodes_require_three_distinct_declining_collections():
    trend=ReturnTrend({})
    for v in (0,-1):
        assert trend.observe([case(i,v,episode=1) for i in range(4)],100-v)['reason'] is None
    assert trend.observe([case(i,-2,episode=2) for i in range(4)],200)['reason']=='consecutive_mixed_episode_return_declines'


def test_pause_keeps_both_matched_cohorts_and_does_not_pair_resets():
    observed = [dict(case(i, -3 if i < 4 else -10), body_ground_seen=True,
                     native_flight={'outgoing_velocity_mps': [3., 4., 0.]}) for i in range(8)]
    baseline = {(str(i), i): case(i, 0) for i in range(8)}
    trend = ReturnTrend(baseline)
    trend.observe(observed[:4], 128)
    decision = trend.observe(observed[4:], 256)
    assert decision['reason'] == 'paired_first_episode_loss_worsened'
    for cohort in decision['cohorts']:
        assert len(cohort['cases']) == 4
        for detail in cohort['cases']:
            assert detail['paired_delta'] == cohort['paired_delta']
            assert detail['observed_scores']['body_ground'] and not detail['baseline_scores']['body_ground']
            assert detail['observed_scores']['release_speed_mps'] == 5
            assert detail['baseline_scores']['release_speed_mps'] is None
    reset = trend.observe([case(i, -12, episode=1) for i in range(4)], 384)
    assert all(c['paired_delta'] is None and c['baseline_scores'] is None for c in reset['cases'])


def test_pause_if_both_curves_keep_falling_despite_a_partial_cohort_rebound():
    baseline = {(str(i), i): case(i, int(i >= 8)) for i in range(12)}
    trend = ReturnTrend(baseline)
    for start, value in ((0, 0), (4, -9), (8, -6)):
        decision = trend.observe([case(i, value) for i in range(start, start+4)], start)
        if start < 8:
            assert decision['reason'] is None
    assert decision['reason'] == 'consecutive_native_and_paired_curve_declines'
    assert decision['cohorts'][-1]['mean'] > decision['cohorts'][-2]['mean']


def test_small_terminal_cohort_with_full_panel_evidence_triggers_at_collection():
    baseline = {(str(i), i): case(i, -100) for i in range(12)}
    for observed_body_count, expected in ((0, None), (2, 'declining_return_with_confirmed_net_new_body_contacts')):
        trend = ReturnTrend(baseline)
        assert trend.observe([case(i, -1) for i in range(4)], 512, 0)['reason'] is None
        assert trend.observe([case(i, -3) for i in range(4, 8)], 640, 0)['reason'] is None
        decision = trend.observe([case(8, -8), case(9, -9)], 768, observed_body_count)
        if expected:
            assert decision['reason'] == expected
            assert decision['cohorts'][-1]['first_episode_body_cases'] == 2
        else:
            assert decision is None
