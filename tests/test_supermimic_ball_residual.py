import json
from pathlib import Path

import torch

from quadsmash.rl.supermimic_ball_residual_v1 import BallResidualV1,ball_geometry_features
from quadsmash.rl.supermimic_full_body_geometry_v1 import FullBodyGeometryV1


def test_geometry_uses_current_physical_state_and_keeps_joint_gradient():
    asset=json.loads(Path('configs/go2_piper_racket_landing_model_v1.json').read_text())['asset_sha256']
    geometry=FullBodyGeometryV1(asset)
    state=torch.zeros(2,55);state[:,3]=1;state[:,2]=.3;state[:,49:52]=torch.tensor([1.,0.,2.])
    state.requires_grad_();features=ball_geometry_features(state,geometry)
    assert features.shape==(2,25) and features.isfinite().all()
    grad=torch.autograd.grad(features[:,6:9].square().sum(),state)[0]
    assert grad[:,13:31].abs().sum()>0
    torch.testing.assert_close(features[:,:6],state[:,49:55])


def test_response_and_gate_parameters_have_separate_supervision():
    model=BallResidualV1(gate_hidden=(8,),response_hidden=(8,))
    x=torch.randn(4,25)
    model.response(model.normalized(x)).square().mean().backward()
    assert all(p.grad is None for p in model.gate.parameters())
    assert any(p.grad is not None for p in model.response.parameters())


def test_ball_not_observed_cannot_receive_learned_correction():
    model=BallResidualV1(gate_hidden=(8,),response_hidden=(8,))
    delta,prob=model(torch.randn(4,25),torch.zeros(4,dtype=torch.bool))
    assert delta.eq(0).all() and ((prob>=0)&(prob<=1)).all()
