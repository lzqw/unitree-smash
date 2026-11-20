from scripts.build_supermimic_event_aligned_probe_specs import make_event_aligned_spec
from scripts.build_supermimic_probe_region import event_validation_lanes


def test_event_probe_spec_keeps_only_event_aligned_lanes():
    spec = make_event_aligned_spec(
        dict(tick=267, lanes=[96, 99, 102, 115], contact=[96, 99, 115], frame=[102]),
        'approach', 0.001,
    )
    assert spec['lanes'] == [96, 99, 102, 115]
    assert len(spec['request_directions']) == 4
    assert spec['event_alignment']['contact_validation_lanes'] == [96, 99, 115]


def test_approach_region_rejects_frame_only_lane():
    spec=make_event_aligned_spec(
        dict(tick=267, lanes=[1, 2], contact=[1], frame=[2]), 'approach', .001)
    assert event_validation_lanes(spec) == [1]
    import pytest
    with pytest.raises(ValueError, match='frame-only'):
        event_validation_lanes(spec, [2])


def test_event_probe_spec_can_request_substep_trace():
    spec = make_event_aligned_spec(
        dict(tick=12, lanes=[2], contact=[2], frame=[]),
        'approach', 0.01, capture_substeps=True)
    assert spec['capture_substeps'] is True


def test_event_probe_spec_can_apply_action_before_event_tick():
    spec = make_event_aligned_spec(
        dict(tick=12, lanes=[2], contact=[2], frame=[]),
        'approach', 0.01, action_lead_ticks=2)
    assert spec['action_tick'] == 10


def test_event_probe_spec_can_hold_action_window():
    spec = make_event_aligned_spec(
        dict(tick=12, lanes=[2], contact=[2], frame=[]),
        'approach', 0.01, action_lead_ticks=4, action_hold_ticks=4)
    assert spec['action_tick'] == 8
    assert spec['action_hold_ticks'] == 4
