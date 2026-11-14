import torch
from quadsmash.rl.supermimic_ball_residual_v2 import (
    approach_features, response_mode_labels, BallResidualV2)


def test_current_relative_motion_predicts_plane_approach():
    f=torch.zeros(2,25); f[:,6]=.04; f[:,9]=torch.tensor([-8.,8.])
    f.requires_grad_(); x=approach_features(f)
    assert x.shape==(2,40) and x.isfinite().all()
    torch.testing.assert_close(x[:,25],torch.tensor([.5,0.]))
    torch.testing.assert_close(x[:,32],torch.tensor([.5,-.5]))
    assert torch.autograd.grad(x[:,26:29].square().sum(),f)[0].isfinite().all()


def test_parallel_zero_velocity_is_finite():
    f=torch.zeros(4,25,requires_grad=True)
    x=approach_features(f); assert x.isfinite().all()
    assert torch.autograd.grad(x.sum(),f)[0].isfinite().all()


def test_labels_disjoint_priority_and_absent_ball():
    masks={k:torch.zeros(6,dtype=torch.bool) for k in ('new_net','after_net','new_ground','after_ground','new_racket')}
    masks['new_racket'][1]=True; masks['after_ground'][2]=True
    masks['new_net'][3]=True; masks['new_ground'][1]=True
    masks['new_racket'][5]=True
    r=torch.ones(6,6);r[0]=0
    labels=response_mode_labels(masks,r,torch.tensor([1,1,1,1,1,0],dtype=torch.bool))
    assert labels.tolist()==[0,1,2,3,4,0]


def test_frozen_model_retains_input_gradients_and_response_loss_is_isolated():
    m=BallResidualV2(gate_hidden=(8,),response_hidden=(8,))
    f=torch.randn(4,25)
    m.response(m.normalized(f)).square().mean().backward()
    assert all(p.grad is None for p in m.gate.parameters())
    for p in m.parameters():p.requires_grad_(False)
    f.requires_grad_();delta,_=m(f,torch.ones(4,dtype=torch.bool))
    g=torch.autograd.grad(delta.sum(),f)[0]
    assert g.isfinite().all() and g.abs().sum()>0
    absent,_=m(f,torch.zeros(4,dtype=torch.bool))
    assert absent.eq(0).all()


def test_free_mode_does_not_have_a_learned_response():
    m=BallResidualV2(gate_hidden=(8,),response_hidden=(8,))
    assert m.residual_scale.shape==(4,6)
    p,r=m.components(torch.zeros(2,25))
    torch.testing.assert_close(p.sum(-1),torch.ones(2))
    assert p.shape==(2,5) and r.shape==(2,4,6)
