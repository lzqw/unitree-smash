import torch
import pytest

from scripts.build_supermimic_training_excitation_specs import make_spec, training_lanes
from quadsmash.rl.supermimic_native_pair_capture_v1 import TrainingWindowRecorder, validate_spec


def test_excitation_uses_train_lanes_and_distinct_reproducible_arm_commands():
    lanes = training_lanes([range(96,128), [0,2,17]])
    assert set(lanes).isdisjoint(set(range(96,128)) | {0,2,17})
    spec = make_spec(lanes, 267, 8, 8, .02, 123)
    assert spec == make_spec(lanes, 267, 8, 8, .02, 123)
    assert (spec['action_tick'], spec['tick']) == (259,274)
    d = torch.tensor(spec['request_directions'])
    assert not d[:,:12].any() and d.unique(dim=0).size(0)==len(lanes)
    torch.testing.assert_close(d.norm(dim=1),torch.ones(len(lanes)))
    with pytest.raises(ValueError,match='16-step'):
        validate_spec(dict(spec, tick=267))


def test_probe_window_rejects_resets_anywhere_in_rollout():
    recorder = TrainingWindowRecorder([4,77])
    recorder.rows=[dict(episode=torch.zeros(2),terminated=torch.zeros(2,dtype=torch.bool),
                        timeout=torch.zeros(2,dtype=torch.bool)) for _ in range(16)]
    recorder.rows[2]['terminated'][1]=True
    with pytest.raises(ValueError,match='boundary'):
        recorder.payload()
