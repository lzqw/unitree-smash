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
