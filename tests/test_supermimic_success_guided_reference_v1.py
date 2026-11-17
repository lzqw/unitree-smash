import torch

from quadsmash.rl.supermimic_task_space_reference_v1 import TaskSpaceReferenceBankV1


class _Geometry:
    model_sha256 = 'geometry-test'


def test_reference_bank_selects_only_terminal_valid_success_lanes():
    states = torch.zeros(8, 3, 55)
    events = torch.zeros(8, 3, 8)
    # lane 0 is a clean approach/recovery success; lane 1 is faulted;
    # lane 2 has no terminal task event.
    events[:, 0, 1:5] = 1.
    events[:, 1, 1:5] = 1.
    events[-1, 1, 7] = 1.
    bank = TaskSpaceReferenceBankV1.from_packed(
        {'state': states, 'event_state': events,
         'elapsed': torch.arange(8, dtype=torch.float32)[:, None].expand(8, 3)},
        _Geometry(), source_sha256='source-test')
    assert bank.geometry_sha256 == 'geometry-test'
    assert bank.source_sha256 == 'source-test'
    assert bank.approach_sequences.shape[:2] == (1, 8)
    assert bank.recovery_sequences.shape[:2] == (1, 8)
    assert bank.manifest['schema'] == 'success-guided-task-space-reference-v1'
