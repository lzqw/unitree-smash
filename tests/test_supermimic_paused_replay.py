import json

import pytest
import torch

from quadsmash.rl.supermimic_physical_loop_loading_v1 import tensor_digest
from scripts.resume_supermimic_revised_research import recover_paused_block


def test_pause_preserves_scored_weights_and_only_imports_observed_replay(tmp_path):
    train = tmp_path / 'train'
    train.mkdir()
    archive = str(train / 'native_transitions')
    previous = {'ticks': 2, 'sha256': 'a'}
    collected = {'ticks': 2, 'sha256': 'b'}
    checkpoint = train / 'pre_block.pt'
    actor = {'recovery': torch.tensor([1.])}
    entry = dict(actor=actor, world={'weight': torch.tensor([2.])}, actor_optimizer={'step': 4},
                 native_ticks=12, global_counters={'accepted': 4}, config={'native_ticks_per_actor_update': 2},
                 replay_archives=[archive], replay_archive_receipts=[{'chunks': [previous]}])
    torch.save(entry, checkpoint)
    pause = dict(checkpoint=str(checkpoint), checkpoint_actor=tensor_digest(actor.items()),
                 body_audit={'source_chunks': ['a', 'b']})
    interrupted = dict(entry, actor={'recovery': torch.tensor([9.])},
                       actor_optimizer={'step': 5}, world={'weight': torch.tensor([9.])},
                       native_ticks=14, replay_state={'write_pos': 1},
                       replay_metadata={'policy_versions': torch.tensor([[4], [2], [4]])},
                       replay_archive_receipts=[{'chunks': [previous, collected]}],
                       actor_start_bank={}, dynamic_stage_starts={})
    (train/'failure.json').write_text(json.dumps({'message': 'SIGTERM'}))
    torch.save(interrupted, train/'interrupted.pt')
    result, source = recover_paused_block(tmp_path, pause)
    assert source == checkpoint
    for field in ('actor', 'actor_optimizer', 'world', 'global_counters'):
        assert result[field] == entry[field]
    assert result['native_ticks'] == 14
    assert result['replay_archive_receipts'] == interrupted['replay_archive_receipts']
    interrupted['replay_metadata']['policy_versions'][2, 0] = 5
    torch.save(interrupted, train/'interrupted.pt')
    with pytest.raises(ValueError, match='scored policy'):
        recover_paused_block(tmp_path, pause)
