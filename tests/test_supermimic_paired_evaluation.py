import torch

from quadsmash.rl.supermimic_paired_evaluation_v1 import paired_evaluation, parameter_delta


def row(case, contact, ball, recovery, complete, value):
    return dict(case_id=str(case), env_id=case, episode_index=0,
                qualified_airborne_string_contact=contact,
                ball_side_complete_except_recovery=ball,
                recovery_success=recovery, complete5=complete,
                body_ground_seen=not recovery, rally_fault=False,
                numerical_failures=0, native_episode_return=value,
                native_episode_discounted_return=value,
                native_return_definition='undiscounted_sum_of_native_control_step_rewards_v1',
                native_episode_length_steps=800)


def test_paired_evaluation_requires_same_case_identity_and_reports_deltas():
    before=[row(i, i<64, i<48, i<32, i<24, -float(i)) for i in range(128)]
    after=[row(i, i<66, i<50, i<34, i<26, -float(i)+.25) for i in range(128)]
    report=paired_evaluation(before,after,stage='post-Approach',
        actor_before={'head':torch.zeros(2)},actor_after={'head':torch.ones(2)})
    assert report['paired_case_identity_verified']
    conditional=report['paired_recovery']
    assert conditional['common_contacts']==64 and conditional['improved_cases']==['32','33']
    assert conditional['regressed_cases']==[] and conditional['gained_contacts']==['64','65']
    assert conditional['paired_two_sided_p']==.5
    assert report['after']['complete5']==26
    assert report['deltas']['complete5_rate'] == .015625
    assert report['actor_parameter_delta']['changed_elements']==2
    assert report['failure_taxonomy_before']['complete5_failure'] == 104

    report=paired_evaluation(before,after,stage='post-Approach',
        action_before={'action':torch.zeros(4,2),'log_std':torch.zeros(2)},
        action_after={'action':torch.ones(4,2),'log_std':torch.zeros(2)})
    assert report['action_rms'] == 1.0 and report['actor_kl'] > 0


def test_parameter_delta_is_explicit_when_unavailable():
    assert parameter_delta(None, {'x': torch.ones(1)}) == {'available': False}


def test_clean_resume_cannot_credit_completed_block_and_resume_still_restores_it():
    from quadsmash.rl.supermimic_sequential_progress_v1 import SequentialProgress
    old = SequentialProgress()
    old.block('pair0', 'recovery', 8, 16).update(status='done', accepted=8, attempts=9)
    for fresh, expected in ((False, 8), (True, 0)):
        restored = SequentialProgress.for_run(old.state_dict(), fresh=fresh)
        assert restored.block('pair0', 'recovery', 8, 16)['accepted'] == expected


def test_paired_baseline_rejects_extra_cases_and_other_policy():
    import pytest
    from quadsmash.rl.supermimic_paired_evaluation_v1 import verify_baseline
    before = [dict(row(i, True, True, True, True, 0.), actor_sha256='actor',
                   policy_execution={'gate': 'native'}) for i in range(128)]
    verify_baseline(before, 'actor', {'gate': 'native'})
    with pytest.raises(ValueError, match='loaded actor'):
        verify_baseline(before, 'another', {'gate': 'native'})
    with pytest.raises(ValueError, match='128'):
        paired_evaluation(before + before[:1], before, stage='repeat')
    report = paired_evaluation(before, before, stage='scale', action_before={
        'action': torch.zeros(2, 1), 'log_std': torch.zeros(1)}, action_after={
        'action': torch.zeros(2, 1), 'log_std': torch.ones(1)})
    expected = torch.distributions.kl_divergence(
        torch.distributions.Normal(0., 1.), torch.distributions.Normal(0., torch.exp(torch.tensor(1.))))
    assert abs(report['actor_kl'] - expected.item()) < 1e-6


def test_native_td_pairs_adjacent_ticks_in_same_lane_and_episode():
    from quadsmash.rl.supermimic_paired_evaluation_v1 import native_critic_td
    identities = [(0, 0, 1), (0, 0, 2), (1, 0, 3), (0, 1, 3), (0, 0, 6)]
    td = native_critic_td([2, 3, 0, 0, 0], [5, 4, 9, 9, 9], identities, .5)
    assert td['sample_count'] == 1 and td['mae'] == 2.5


def test_load_native_training_baseline_uses_completed_frozen_run_and_rejects_conflicts(tmp_path):
    import json
    import pytest
    from scripts.compare_supermimic_fixed_stages import load_training_baseline
    from quadsmash.rl.supermimic_native_metrics import summarize_native_episodes
    rows = [row(i,True,True,True,True,2.) for i in range(128)]
    execution, task = {'gate':'causal'}, {'suite':'fixed'}
    metrics = dict(actor_parameters_unchanged=True,num_envs=128,optimizer_updates={'actor':0},
                   scoring_schema='s',actor_state_sha256='a',task_contract=task,policy_execution=execution,
                   native_episode_metrics=summarize_native_episodes(rows))
    for name, data in dict(startup={'argv':['--seed','1']}, completed={'control_ticks':800,'episodes':128},
                           metrics=metrics,episodes_summary=rows,task_contract=task,initial_states={},
                           physics={},policy_execution={'contract':execution}).items():
        (tmp_path/(name+'.json')).write_text(json.dumps(data))
    path = tmp_path/'episodes_summary.json'
    assert all(r['actor_sha256']=='a' for r in load_training_baseline(path,'a',execution,task))
    with pytest.raises(ValueError,match='loaded actor'):
        load_training_baseline(path,'b',execution,task)
    rows[0]['actor_sha256']='b'
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError,match='Baseline differs'):
        load_training_baseline(path,'a',execution,task)
    rows[0].pop('actor_sha256')
    rows[0]['complete5']=False
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError,match='scores disagree'):
        load_training_baseline(path,'a',execution,task)


def test_mid_block_resume_keeps_baseline_and_previous_refresh_separate_from_live_actor(tmp_path):
    import json
    import pytest
    from scripts.compare_supermimic_fixed_stages import digest, restore_training_comparison, training_comparison_context
    from quadsmash.rl.supermimic_physical_loop_loading_v1 import tensor_digest

    pre, post, current = ({'head': torch.tensor([x])} for x in (1., 2., 3.))
    execution = {'recovery_gate':'posthit'}
    def records(actor):
        sha = tensor_digest(actor.items())
        return [dict(row(i,True,True,True,True,1.),actor_sha256=sha,policy_execution=execution) for i in range(128)]
    path=tmp_path/'initial.json'
    path.write_text(json.dumps(records(pre)))
    legacy=dict(config={'paired_evaluation_baseline':str(path)},actor_references={'block':pre})
    snapshots, actors, sha=restore_training_comparison(path,current,execution,{},continuation=legacy)
    assert sha == tensor_digest(pre.items()) and current['head'].item() == 3.
    snapshots['post-refresh_05'] = records(post)
    actors['post-refresh_05'] = post
    fresh=dict(legacy,actor_references={},paired_evaluation_context=training_comparison_context(snapshots,actors,digest(path)))
    restored, weights, _=restore_training_comparison(path,current,execution,{},continuation=fresh)
    assert list(restored)==['pre-update','post-refresh_05'] and weights['post-refresh_05']['head'].item()==2.
    with pytest.raises(ValueError,match='Baseline differs'):
        restore_training_comparison(path,current,execution,{})
    path.write_text(json.dumps(records(post)))
    with pytest.raises(ValueError,match='scores changed'):
        restore_training_comparison(path,current,execution,{},continuation=fresh)


def test_native_panel_reports_flips_and_return_interval_on_their_own_denominators():
    from scripts.compare_supermimic_fixed_stages import paired_task_outcomes
    from scripts.evaluate_supermimic_research_arm import paired_scalars

    before = [row(i, i < 2, i < 2, i == 0, i == 0, 2.) for i in range(128)]
    after = [row(i, i in (1, 2), i in (1, 2), i in (1, 2), i in (1, 2), 2.5)
             for i in reversed(range(128))]
    result = paired_evaluation(before, after, stage='native')
    outcomes = paired_task_outcomes(result)
    assert outcomes['binary']['complete5']['regressed_cases'] == ['0']
    assert outcomes['binary']['complete5']['improved_cases'] == ['1', '2']
    assert outcomes['native_return']['descriptive_paired_case_bootstrap_95'] == [.5, .5]
    cached = paired_scalars(dict(result, paired_task_outcomes=outcomes))
    assert cached == paired_scalars(result)
    assert cached['common_contact_recovery/common_contacts'] == 1
    assert cached['common_contact_recovery/gained_cases'] == 1
    assert cached['common_contact_recovery/lost_cases'] == 0
    assert cached['contact/gained_cases'] == cached['contact/lost_cases'] == 1


def test_failure_flips_require_a_frame_first_and_count_ground_on_all_cases():
    from scripts.compare_supermimic_fixed_stages import first_episode_failure_flips

    string = dict(qualified_airborne_string_contact=True, had_contact=True, body_ground_seen=False,
                  frame_hit=False, string_hit=True, first_string_time_s=1., contact_time_s=1.)
    frame = dict(string, qualified_airborne_string_contact=False, frame_hit=True,
                 first_string_time_s=1.0000625, body_ground_seen=True)
    same_substep = dict(frame, first_string_time_s=1.)
    miss = dict(string, qualified_airborne_string_contact=False, had_contact=False, string_hit=False)
    before = dict(a=string, b=frame, c=string, d=miss)
    after = dict(d=dict(miss, body_ground_seen=True), c=same_substep, b=string, a=frame)
    assert first_episode_failure_flips(before, after, .0000625) == dict(
        new_body_ground=['a', 'c', 'd'], avoided_body_ground=['b'],
        qualified_to_frame_first=['a'], frame_first_to_qualified=['b'])


def test_simultaneous_and_unmeasured_frame_order_are_separate():
    from scripts.compare_supermimic_fixed_stages import frame_before_string
    dt = .0000625
    hit = dict(had_contact=True, frame_hit=True, string_hit=True, contact_time_s=2.)
    assert frame_before_string(dict(hit, first_string_time_s=2.), dt) is False
    assert frame_before_string(dict(hit, first_string_time_s=2.+dt), dt) is True
    assert frame_before_string(hit, dt) is None
    assert frame_before_string(dict(hit, string_hit=False), dt) is True
