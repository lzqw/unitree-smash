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
