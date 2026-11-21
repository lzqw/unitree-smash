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
