import torch

from quadsmash.rl.supermimic_physical_loop_replay_v1 import (
    PhysicalLoopReplay,
    _stage_ids_from_tensors,
    sample_stratified_windows,
)


def _data(time=8, lanes=2):
    zeros = lambda *shape: torch.zeros(*shape)
    data = dict(
        event_state=zeros(time, lanes, 8),
        next_event_state=zeros(time, lanes, 8),
        feet=torch.zeros(time, lanes, 4, dtype=torch.bool),
        next_feet=torch.zeros(time, lanes, 4, dtype=torch.bool),
        elapsed=torch.arange(time, dtype=torch.float32)[:, None].expand(time, lanes) * .9,
        complete_reward=torch.zeros(time, lanes, dtype=torch.bool),
    )
    return data


def test_stage_classification_is_per_lane_and_touchdown_is_event_aligned():
    data = _data()
    # Only lane 0 touches down at tick 4; a batch-wide any() would label lane 1.
    data['event_state'][2:, 0, 0] = 1.
    data['next_feet'][4, 0, 0] = True
    data['next_event_state'][6, 1, 5] = 1.
    ids = _stage_ids_from_tensors(data)
    assert int(ids[4, 0]) == 4
    assert int(ids[4, 1]) == 2
    assert int(ids[6, 1]) == 5


def test_stratified_sampler_reports_quota_shortfall_instead_of_faking_data():
    pools = {
        'pre_touchdown': torch.tensor([[0, 0],[0,1], [1, 0],[1,1]]),
        'touchdown': torch.tensor([[2, 0],[2,1],[3,0], [3, 1]]),
    }
    source_labels = torch.tensor([[0, 0], [0, 0], [1, 1], [1, 1]])
    chosen, report = sample_stratified_windows(
        pools, source_labels, 8, torch.Generator().manual_seed(11),
        source_quota={0: 4, 1: 4},
        stage_quota={'pre_touchdown': 2, 'touchdown': 2, 'recovery_failure': 4},
    )
    assert chosen.shape == (8, 2)
    assert report['stage_shortfall']['recovery_failure'] > 0
    assert report['source_shortfall'][0] >= 0
    assert report['fallback_count'] > 0


def test_replay_legacy_window_reports_source_without_stratification_bug():
    t, lanes = 24, 2
    tensors = {
        'state': torch.zeros(t, lanes, 55), 'next_state': torch.zeros(t, lanes, 55),
        'values': torch.zeros(t, lanes, 8), 'valid': torch.ones(t, lanes, 8),
        'next_values': torch.zeros(t, lanes, 8), 'next_valid': torch.ones(t, lanes, 8),
        'event_state': torch.zeros(t, lanes, 8), 'next_event_state': torch.zeros(t, lanes, 8),
        'targets': torch.zeros(t, lanes, 18), 'previous_action': torch.zeros(t, lanes, 18),
        'feet': torch.zeros(t, lanes, 4, dtype=torch.bool), 'next_feet': torch.zeros(t, lanes, 4, dtype=torch.bool),
        'present': torch.zeros(t, lanes, dtype=torch.bool), 'elapsed': torch.arange(t)[:, None].expand(t, lanes).float(),
        'episode': torch.zeros(t, lanes, dtype=torch.long), 'terminated': torch.zeros(t, lanes, dtype=torch.bool),
        'timeout': torch.zeros(t, lanes, dtype=torch.bool), 'torque': torch.zeros(t, lanes, 18),
        'complete_reward': torch.zeros(t, lanes, dtype=torch.bool),
    }
    replay = PhysicalLoopReplay({'tensors': tensors}, None, capacity_ticks=64)
    _, _, report = replay.window(steps=16, batch=4, device='cpu')
    assert report['window_count'] == 4
    assert report['source_counts'] == {'initial_archive': 4, 'recent_online': 0, 'older_online': 0}
    _,_,approach=replay.window(steps=16,batch=4,device='cpu',head='approach',
        source_quota={'initial_archive':4},stage_quota={'approach_active':4})
    assert approach['approach_missing_fields']['approach_gate']==64
    assert approach['stage_shortfall']['approach_active']==4
    assert approach['actual_event_coverage']['approach_active']==0
    assert approach['fallback_count']==4
