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
