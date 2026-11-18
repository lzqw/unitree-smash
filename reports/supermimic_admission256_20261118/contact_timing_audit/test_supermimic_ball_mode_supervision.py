"""Observed mode supervision contracts; no native data fit or Actor update."""
import pytest
import torch
from quadsmash.rl.supermimic_ball_mode_supervision_v1 import observed_ball_mode_targets,observed_ball_mode_loss
from quadsmash.rl.supermimic_ball_residual_v2 import BallResidualV2


def test_unexplained_impulses_and_parked_ball_are_not_false_free_labels():
    residual=torch.zeros(5,6)
    residual[1,3]=10.;residual[2,3]=2.;residual[3,0]=.01
    labels=observed_ball_mode_targets(residual,torch.tensor([False,True,False,False,False]),
                                     torch.tensor([True,True,True,True,False]))
    assert labels.tolist()==[0,1,-1,-1,-1]


def test_native_contact_edge_takes_priority_over_small_observed_residual():
    labels=observed_ball_mode_targets(torch.zeros(2,6),torch.ones(2,dtype=torch.bool),
                                     torch.tensor([True,False]))
    assert labels.tolist()==[1,-1]


def test_unknown_rows_cannot_supply_gate_gradients_or_dilute_known_rows():
    logits=torch.zeros(3,5,requires_grad=True)
    labels=torch.tensor([0,1,-1])
    loss,counts=observed_ball_mode_loss(logits,labels)
    expected=torch.nn.functional.cross_entropy(logits[:2],labels[:2])
    assert torch.equal(loss,expected)
    loss.backward()
    assert counts==dict(free=1,racket=1,unknown=1,supervised=2)
    assert torch.equal(logits.grad[2],torch.zeros(5))
    assert logits.grad[0,0]<0 and logits.grad[1,1]<0


def test_no_known_labels_returns_zero_not_nan_or_fake_supervision():
    logits=torch.zeros(2,5,requires_grad=True)
    loss,counts=observed_ball_mode_loss(logits,torch.full((2,),-1,dtype=torch.long))
    loss.backward()
    assert loss.item()==0 and counts['supervised']==0
    assert not logits.grad.any()


def test_real_gate_can_update_while_response_and_normalization_remain_frozen():
    torch.manual_seed(19)
    model=BallResidualV2(gate_hidden=(8,),response_hidden=(8,))
    model.requires_grad_(False);model.gate.requires_grad_(True)
    before={k:v.clone() for k,v in model.state_dict().items()}
    features=torch.randn(4,40);labels=torch.tensor([0,0,1,-1])
    optimizer=torch.optim.Adam(model.gate.parameters(),lr=2e-4)
    loss,_=observed_ball_mode_loss(model.gate(features),labels)
    optimizer.zero_grad(set_to_none=True);loss.backward()
    assert any(p.grad is not None and p.grad.abs().max()>0 for p in model.gate.parameters())
    assert all(p.grad is None for p in model.response.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    after=model.state_dict()
    assert any(not torch.equal(after[k],v) for k,v in before.items() if k.startswith('gate.'))
    assert all(torch.equal(after[k],v) for k,v in before.items() if not k.startswith('gate.'))


@pytest.mark.parametrize('bad', [float('nan'),float('inf')])
def test_nonfinite_native_residual_is_rejected(bad):
    residual=torch.zeros(1,6);residual[0,0]=bad
    with pytest.raises(ValueError,match='Nonfinite'):
        observed_ball_mode_targets(residual,torch.zeros(1,dtype=torch.bool),torch.ones(1,dtype=torch.bool))


def test_shape_and_unknown_label_contracts_are_explicit():
    with pytest.raises(ValueError,match='batch mismatch'):
        observed_ball_mode_targets(torch.zeros(2,6),torch.zeros(1,dtype=torch.bool),torch.ones(2,dtype=torch.bool))
    with pytest.raises(ValueError,match='Only known'):
        observed_ball_mode_loss(torch.zeros(1,5),torch.tensor([4]))
    with pytest.raises(FloatingPointError):
        observed_ball_mode_loss(torch.full((1,5),float('nan')),torch.tensor([0]))
