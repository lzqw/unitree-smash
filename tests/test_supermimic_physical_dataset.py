import torch
import pytest

from quadsmash.rl.supermimic_physical_dataset_v1 import (
    STATE_FIELDS, STATE_DIM, encode_physical, decode_physical, valid_windows, split_lanes)


def test_physical_codec_roundtrip_and_differentiability():
    assert STATE_DIM == 55
    x = torch.randn(4, STATE_DIM, requires_grad=True)
    decoded = decode_physical(x)
    assert list(decoded) == [name for name, _ in STATE_FIELDS]
    torch.testing.assert_close(encode_physical(decoded), x, atol=0, rtol=0)
    encode_physical(decoded).sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))
    with pytest.raises(ValueError, match='field shape'):
        encode_physical(dict(decoded, joint_position=x[:, :17]))


def test_windows_allow_terminal_last_but_never_reset_inside():
    episode = torch.tensor([[0,0],[0,0],[0,0],[1,0],[1,0],[1,0]])
    terminal = torch.zeros_like(episode, dtype=torch.bool)
    timeout = terminal.clone(); timeout[2,0] = True
    mask = valid_windows(episode, terminal, timeout, 3)
    assert mask[:,0].tolist() == [True, False, False, True]
    assert mask[:,1].all()
    # Even an erroneous unchanged episode ID must not allow crossing done.
    assert not valid_windows(torch.zeros_like(episode),terminal,timeout,3)[1,0]


def test_validation_keeps_entire_cases_separate():
    train, val = split_lanes(128, 32, 971)
    assert len(train) == 96 and len(val) == 32
    assert not set(train.tolist()) & set(val.tolist())
    assert sorted(train.tolist()+val.tolist()) == list(range(128))
    assert torch.equal(val, split_lanes(128,32,971)[1])
    with pytest.raises(ValueError):
        split_lanes(128,128,971)


def test_packing_archive_preserves_terminal_state_and_final_command(tmp_path):
    from quadsmash.rl.supermimic_physical_capture_v1 import PhysicalTransitionArchive
    from quadsmash.rl.supermimic_physical_dataset_v1 import pack_archive
    def snapshot(tick):
        physical = {name:torch.full((2,width),float(tick)) for name,width in STATE_FIELDS}
        physical.update(elapsed_s=torch.full((2,),tick*.01),
                        foot_contact=torch.zeros(2,4,dtype=torch.bool),
                        ball_present=torch.ones(2,dtype=torch.bool))
        return dict(control_tick=tick, episode_index=torch.zeros(2,dtype=torch.long),
                    physical=physical,event_state=torch.zeros(2,8))
    archive = PhysicalTransitionArchive(tmp_path/'source',metadata={},chunk_ticks=1)
    for tick in (798,799):
        archive.begin_record(snapshot(tick),torch.ones(2,18),torch.zeros(2,794),{})
        archive.pending['base_action']=torch.zeros(2,18)
        archive.finish_record(snapshot(tick+1),torch.full((2,18),.3),torch.full((2,18),.7),
                              torch.zeros(2),torch.zeros(2,dtype=torch.bool),
                              torch.full((2,),tick==799),{})
    result=pack_archive(tmp_path/'source')['tensors']
    assert result['state'].shape==(2,2,55)
    assert result['next_state'][-1].eq(800).all()
    assert result['residual_request'].eq(1).all()
    assert result['final_joint_targets'].eq(.7).all()
    assert valid_windows(result['episode'],result['terminated'],result['timeout'],2).all()
