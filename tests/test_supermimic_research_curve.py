from scripts.report_supermimic_research_curve import axes, read_new, scalars


def test_native_axis_excludes_inherited_replay_and_repeated_proposals():
    row = dict(accepted_id=18, native_ticks=5118, native_transitions=5118*128, research_elapsed_s=321.9)
    assert axes(row,(17,4990),128) == dict(retained_updates=1,new_native_env_steps=16384,training_wall_seconds=321)
    row['accepted_id']+=1
    assert axes(row,(17,4990),128)['new_native_env_steps'] == 16384


def test_incomplete_json_line_is_consumed_on_next_poll(tmp_path):
    path = tmp_path/'history'
    path.write_bytes(b'{"a":1}\n{"a":')
    rows, cursor = read_new(path,0)
    assert rows == [{'a':1}]
    with path.open('ab') as f:
        f.write(b'2}\n')
    assert read_new(path,cursor) == ([{'a':2}],path.stat().st_size)


def test_supervision_batches_are_distinct_from_optimizer_steps():
    row = dict(block='recovery', update_accepted=True, peak_gpu_bytes=2**30,
        world_update_reports=[dict(contact_supervision=dict(updated=False, measured_events=0)),
                              dict(contact_supervision=dict(updated=True, measured_events=3))])
    result = scalars(row)
    assert result['world/contact_supervision_batches'] == 2
    assert result['world/contact_optimizer_steps'] == 1
    assert result['world/sampled_contact_events'] == 3
    assert 'world/collection_policy_transition_uses' not in result


def test_fit_coverage_tracks_the_policy_that_collected_before_a_retained_or_rejected_proposal():
    batch = dict(sampling=dict(online_fit_by_policy={
        '17': dict(transition_uses=5, racket_edge_uses=2),
        '25': dict(transition_uses=8, racket_edge_uses=3)}))
    for accepted in (True, False):
        result = scalars(dict(block='recovery', update_accepted=accepted, accepted_id=25+int(accepted),
            peak_gpu_bytes=0, world_update_reports=[batch, dict(batch, contact_supervision={
                'updated': True, 'measured_events': 5})]))
        assert result['world/online_transition_uses'] == 26
        assert result['world/collection_policy_transition_uses'] == 16
        assert result['world/collection_policy_racket_edge_uses'] == 6
        assert result['world/collection_policy_contact_fit_edge_uses'] == 3


def test_first_pair_budget_waits_for_both_heads_and_counts_rejections():
    from scripts.report_supermimic_first_pair_budget import first_pair_budget
    from quadsmash.rl.supermimic_alternating_schedule_v1 import make_plan
    plan = make_plan(128,128,256,256,refresh_every_pairs=8)
    rows = []
    for head in ('recovery','approach'):
        for attempt in range(9):
            rows.append(dict(block_key='cycle000_pair000',block=head,update_accepted=attempt>0,
                             block_accepted_updates=attempt,research_elapsed_s=10*(len(rows)+1),
                             bptt_seconds=5.))
    historical = dict(frozen_collection=dict(proposal_record_to_episodes_file_seconds=100.),
                      evaluation=dict(elapsed_s=60.))
    assert first_pair_budget(plan,rows[:-1],historical) is None
    result = first_pair_budget(plan,rows,historical)
    assert result['first_pair']['recovery']['bptt_seconds'] == 45.
    assert result['first_pair_wall_seconds'] == 180.
    assert result['estimated_remaining_hours_after_first_pair'] == (180*15+320)/3600
    first_arm=first_pair_budget(plan,rows,historical,shared_baseline_pending=True)
    assert first_arm['fixed_panels']==3
    assert first_arm['estimated_remaining_hours_after_first_pair']==(180*15+380)/3600


def test_extension_contact_totals_include_prior_stage_and_repair_missing_export(tmp_path):
    import json
    from scripts.report_supermimic_native_contact_counts import NativeContactCounts
    directory = tmp_path/'train/native_transitions'
    directory.mkdir(parents=True)
    prior = {'qualified_contact': 8, 'frame': 2, 'qualified_or_frame': 9}
    counts = {'qualified_contact': 2, 'frame': 0, 'qualified_or_frame': 2}
    (tmp_path/'registration.json').write_text(json.dumps({'previous_native_contacts': {'env_steps': 512, 'counts': prior}}))
    (directory/'manifest.json').write_text(json.dumps({'num_envs': 128, 'chunks': [
        {'file':'a','sha256':'example','ticks':2,'native_events':counts}]}))
    for _ in range(2):
        NativeContactCounts(tmp_path).update()
        report = json.loads((tmp_path/'native_contact_counts.json').read_text())
        assert report['durable_env_steps'] == 768 and report['total']['qualified_or_frame'] == 11


def test_checkpoint_and_stage_end_account_for_frozen_suffix_and_async_episode_lengths():
    import pytest
    from scripts.evaluate_supermimic_research_arm import training_interactions

    receipts = [dict(start_tick=110, end_tick=125, native_ticks=15, complete_lanes=2, lengths=[8, 7]),
                dict(start_tick=135, end_tick=147, native_ticks=12, complete_lanes=2, lengths=[8, 8])]
    assert training_interactions(100, 110, 2, receipts)['frozen_all'] == 0
    assert training_interactions(100, 135, 2, receipts) == dict(
        training_total=70, online_mixed=40, frozen_all=30, frozen_in_complete_rtg_episodes=15)
    assert training_interactions(100, 147, 2, receipts) == dict(
        training_total=94, online_mixed=40, frozen_all=54, frozen_in_complete_rtg_episodes=31)
    with pytest.raises(ValueError):
        training_interactions(100, 147, 2, receipts + receipts[:1])


def test_interaction_budget_uses_only_collections_completed_by_checkpoint(tmp_path):
    import json
    import pytest
    from scripts.evaluate_supermimic_research_arm import checkpoint_interactions

    manifest = tmp_path/'manifest.json'
    identity = dict(policy_id='combined_v128', actor_sha256='abc')
    manifest.write_text(json.dumps(dict(**identity, schedule_event_id='first_refresh', start_tick=100,
        end_tick=112, native_ticks=12, complete_lanes=2, lengths=[8, 7])))
    refresh = dict(event=dict(kind='refresh', id='first_refresh'),
                   result=dict(**identity, manifest_path=str(manifest)))
    ck = dict(native_ticks=120, alternating_schedule={'results':[refresh]})
    assert checkpoint_interactions(ck, 90, 2)['online_mixed'] == 36
    refresh['result']['policy_id'] = 'wrong'
    with pytest.raises(ValueError, match='identity'):
        checkpoint_interactions(ck, 90, 2)


def test_contact_contribution_separates_weighted_gradient_from_coverage():
    row = dict(block='approach', update_accepted=True, peak_gpu_bytes=0, toi_racket_weight=2.,
               task_gradient_norm=5., tracking_gradient_norm=.25, task_tracking_cosine=-.5)
    assert 'approach/outgoing_to_tracking_gradient_ratio' not in scalars(row)
    active = scalars(dict(row, contact_aux_weight=.1, outgoing_impact_lanes=2))
    assert active['approach/weighted_outgoing_gradient_norm'] == .5
    assert active['approach/outgoing_to_tracking_gradient_ratio'] == 1.
    assert active['approach/outgoing_impact_lanes'] == 2
    legacy = scalars(dict(row, contact_aux_weight=.1, tracking_gradient_norm=0))
    assert 'approach/outgoing_impact_lanes' not in legacy
    assert 'approach/outgoing_to_tracking_gradient_ratio' not in legacy


def test_mixed_case_returns_use_only_first_online_episode_and_matching_frozen_case(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from scripts import compare_supermimic_fixed_stages as fixed
    from scripts.report_supermimic_research_curve import FirstEpisodeReturns
    base = dict(case_id='a', env_id=2, scoring_schema='v1', native_episode_return=10.,
                native_episode_length_steps=800, episode_timeout=True,
                physical_recovery_success=False, complete5=False, ball_side_complete_except_recovery=True)
    identity = dict(seed=1, task='t', execution='e')
    monkeypatch.setattr(fixed, 'load_run', lambda folder: ([base], identity, 'h'))
    report = FirstEpisodeReturns(tmp_path, dict(baseline_path='baseline', initial={'actor_sha256':'h'}),
                                 dict(seed=1, task_contract='t', policy_execution='e'))
    (tmp_path/'train').mkdir()
    row = dict(base, scope='mixed-policy monitoring, not frozen score', episode_index=0,
               native_episode_return=6., ball_side_complete_except_recovery=False,
               native_episode_length_steps=505, episode_timeout=False)
    report.path.write_text(''.join(json.dumps(r)+'\n' for r in
                                   [dict(row,scope='frozen'), dict(row,episode_index=1), row, row]))
    events = []
    writer = SimpleNamespace(add_event=events.append, flush=lambda: None)
    report.update(writer)
    report.update(writer)
    assert len(events) == 1 and events[0].step == 1
    values = {s.tag:s.simple_value for s in events[0].summary.value}
    assert values['first_completed_mixed_cases/mean_difference'] == -4.
    assert values['first_completed_mixed_cases/mean_episode_length_delta_ticks'] == -295
    assert values['first_completed_mixed_cases/baseline_timeout_to_online_early'] == 1
    assert values['first_completed_mixed_cases/ball_side_complete_except_recovery_net_cases'] == -1


def test_geometric_cross_and_target_require_a_qualified_fast_release_for_native_rewards():
    from scripts.report_supermimic_online_flights import flight_flags
    shot=dict(released=True,had_contact=True,frame_hit=False,legal_initial_contact=True,
              string_hit=True,airborne_string_hit=True,ground_seen=True,
              crossed_net=True,landed_target=True,shuttle_net_contact=False)
    slow=flight_flags(dict(shot,outgoing_velocity_mps=[4.999,0.,0.]))
    assert slow['crossed_net'] and slow['landed_target'] and not slow['scoring_cross'] and not slow['scoring_target']
    assert flight_flags(dict(shot,outgoing_velocity_mps=[5.,0.,0.]))['scoring_target']
    assert not flight_flags(dict(shot,outgoing_velocity_mps=[-6.,0.,0.]))['scoring_cross']
    miss=flight_flags(dict(shot,released=False,outgoing_velocity_mps=[None]*3))
    assert not miss['released_at_least_5_mps'] and not miss['scoring_target']


def test_target_score_evidence_separates_speed_margin_from_misses_and_backward_releases():
    from scripts.report_supermimic_online_flights import flight_flags, target_evidence
    shot=dict(released=True,had_contact=True,frame_hit=False,legal_initial_contact=True,
              string_hit=True,airborne_string_hit=True,ground_seen=True,
              crossed_net=True,shuttle_net_contact=False,landed_target=True)
    for x,landed,expected in [(5., True, []), (4.999, True, ['release_speed']),
                              (-5., False, ['forward_release','landed_target'])]:
        case=dict(shot, outgoing_velocity_mps=[x,0.,0.], landed_target=landed)
        flags=flight_flags(case)
        evidence=target_evidence(case,flags)
        assert evidence['release_speed_mps'] == abs(x)
        assert evidence['failed_conditions'] == expected
        assert flags['scoring_target'] == (not expected)
    missing=dict(shot,outgoing_velocity_mps=[None]*3)
    assert target_evidence(missing,flight_flags(missing)) == dict(release_speed_mps=None,failed_conditions=['measured_release'])
