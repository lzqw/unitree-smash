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
    final = [dict(case(i, -2, episode=2), body_ground_seen=True, complete5=False,
                  relaxed_recovery={'stable_s': .1, 'first_reset_reasons': ['body_ground']}) for i in range(4)]
    decision = trend.observe(final, 200)
    assert decision['reason'] == 'consecutive_mixed_episode_return_declines'
    cohort = decision['cohorts'][-1]
    assert cohort['paired_count'] == cohort['complete5_count'] == 0
    assert cohort['episode_indices'] == {2: 4}
    assert cohort['body_ground_outcomes']['contact'] == {'count': 4, 'mean_return': -2}
    assert decision['cases'][0]['observed_scores']['relaxed_recovery'] == final[0]['relaxed_recovery']


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


def test_native_collection_accumulates_distinct_body_lanes_before_first_reset():
    import torch
    from quadsmash.rl.supermimic_native_metrics import count_first_episode_body_contacts
    seen = torch.zeros(3, dtype=torch.bool)
    def collect(episode, edges):
        return count_first_episode_body_contacts(seen, torch.tensor(episode), torch.tensor(edges).bool())
    assert collect([[0, 0, 0], [0, 0, 1]], [[1, 0, 0], [1, 0, 1]]) == 1
    assert collect([[1, 0, 1], [1, 0, 1]], [[1, 0, 1], [0, 1, 0]]) == 2
    assert seen.tolist() == [True, True, False]


def test_registered_arm_and_external_watcher_only_observe_once(tmp_path, monkeypatch):
    import fcntl
    import psutil
    from scripts import watch_supermimic_native_return as watch
    calls = []
    monkeypatch.setattr(watch, 'monitor', lambda arm, process: calls.append((arm, process.pid)))
    trainer = psutil.Process()
    with (tmp_path/'native_return_watch.lock').open('a') as existing:
        fcntl.flock(existing, fcntl.LOCK_EX)
        watch.monitor_registered_arm(tmp_path, trainer, wait_for_lock=False)
        assert not calls
    watch.monitor_registered_arm(tmp_path, trainer, wait_for_lock=True)
    assert calls == [(tmp_path, trainer.pid)]


def test_pause_and_resume_with_buffered_collections_preserves_each_cohort(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from scripts import watch_supermimic_native_return as watch
    train = tmp_path/'train'
    (train/'native_transitions').mkdir(parents=True)
    (train/'native_transitions/manifest.json').write_text('{}')
    (tmp_path/'registration.json').write_text(json.dumps(dict(arm='G1', baseline_path=str(tmp_path/'baseline.json'))))
    baseline = {(str(i), i): case(i, 0) for i in range(16)}
    monkeypatch.setattr(watch, 'FirstEpisodeReturns', lambda *args: SimpleNamespace(baseline=baseline))
    episodes = [case(i, -(i//4+1)) for i in range(16)]
    history = [dict(completed_native_episodes=n, native_ticks=n*8, accepted_id=n//4) for n in (4,8,12,16)]
    for path, rows in ((train/'native_episodes.jsonl', episodes), (train/'native_collection_history.jsonl', history)):
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    trainer = SimpleNamespace(pid=1, state=watch.psutil.STATUS_RUNNING, alive=True)
    trainer.cmdline = lambda: ['scripts/train_supermimic_physical_block.py','--output',str(train),'--recovery-rollout-landing']
    trainer.is_running = lambda: trainer.alive
    trainer.status = lambda: trainer.state
    trainer.suspend = lambda: setattr(trainer, 'state', watch.psutil.STATUS_STOPPED)
    def resume_or_finish(_):
        if trainer.state == watch.psutil.STATUS_STOPPED:
            trainer.state = watch.psutil.STATUS_RUNNING
        else:
            trainer.alive = False
    monkeypatch.setattr(watch.time, 'sleep', resume_or_finish)
    watch.monitor(tmp_path, trainer)
    decisions, _ = watch.read_new(tmp_path/'native_return_pauses.jsonl', 0)
    assert [d['through_episode'] for d in decisions] == [8,16]
    assert [d['cohorts'][-1]['tick'] for d in decisions] == [64,128]
