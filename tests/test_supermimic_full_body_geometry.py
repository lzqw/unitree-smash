import json
from pathlib import Path
import pytest
import torch
from quadsmash.rl.supermimic_full_body_geometry_v1 import FullBodyGeometryV1, MODEL, rotation, rotate
from quadsmash.control.reactive_landing_v1 import LoadedKinematics

ASSET=json.loads(MODEL.read_text())['asset_sha256']


def inputs(n=3):
    g=torch.Generator().manual_seed(17)
    q=torch.randn(n,18,generator=g,dtype=torch.float64)*.25
    qd=torch.randn(n,18,generator=g,dtype=torch.float64)*.5
    p=torch.zeros(n,3,dtype=torch.float64)
    quat=torch.tensor([[1.,0.,0.,0.]],dtype=torch.float64).expand(n,-1).clone()
    v=torch.randn(n,3,generator=g,dtype=torch.float64)
    w=torch.randn(n,3,generator=g,dtype=torch.float64)
    return q,qd,p,quat,v,w


def test_all_body_geometry_matches_pinned_loaded_model():
    model=FullBodyGeometryV1(ASSET,dtype=torch.float64)
    reference=LoadedKinematics(dtype=torch.float64)
    q,qd,p,quat,v,w=inputs()
    actual=model(q,qd,p,quat,v,w)
    old=reference.forward(q,qd,v,w)
    for name in ('position','rotation','feet','center'):
        torch.testing.assert_close(actual[name],old[name],atol=1e-10,rtol=1e-10)
    torch.testing.assert_close(actual['center_velocity'],old['center_v'],atol=1e-10,rtol=1e-10)
    assert actual['head_origins'].shape==(3,2,3)


def test_racket_only_chain_preserves_ball_features_and_input_gradients():
    from quadsmash.rl.supermimic_ball_residual_v1 import ball_geometry_features
    from quadsmash.rl.supermimic_racket_dynamics_v1 import racket_state
    model = FullBodyGeometryV1(ASSET, dtype=torch.float64)
    class FullGeometry:
        ids = model.ids
        def __call__(self, *args):
            return model(*args)
    generator = torch.Generator().manual_seed(81)
    state = torch.randn(9, 55, dtype=torch.float64, generator=generator, requires_grad=True)
    assert 0 < len(model.racket_chain) < len(model.edges)
    for feature in (racket_state, ball_geometry_features):
        expected = feature(state, FullGeometry())
        actual = feature(state, model)
        weights = torch.randn(actual.shape, dtype=actual.dtype, generator=generator)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        reference_grad, = torch.autograd.grad(expected, state, weights)
        actual_grad, = torch.autograd.grad(actual, state, weights)
        torch.testing.assert_close(actual_grad, reference_grad, rtol=1e-13, atol=1e-13)


def test_world_pose_and_velocity_transform_consistently():
    m=FullBodyGeometryV1(ASSET,dtype=torch.float64)
    q,qd,p,quat,v,w=inputs()
    a=m(q,qd,p,quat,v,w)
    quat2=torch.tensor([[.8,.2,-.1,.4]],dtype=q.dtype).expand(3,-1)
    r=rotation(quat2)
    p2=p+torch.tensor([1.,-2.,.5],dtype=q.dtype)
    b=m(q,qd,p2,quat2,rotate(r,v),rotate(r,w))
    for key in ('position','feet','head_origins'):
        torch.testing.assert_close(b[key],rotate(r[:,None],a[key])+p2[:,None],atol=1e-10,rtol=1e-10)
    torch.testing.assert_close(b['center_velocity'],rotate(r,a['center_velocity']),atol=1e-10,rtol=1e-10)


def test_com_velocity_matches_position_time_derivative():
    m=FullBodyGeometryV1(ASSET,dtype=torch.float64)
    q,qd,p,quat,v,w=inputs();w=w*0
    eps=1e-6
    a=m(q,qd,p,quat,v,w)
    before=m(q-eps*qd,qd,p-eps*v,quat,v,w)
    after=m(q+eps*qd,qd,p+eps*v,quat,v,w)
    torch.testing.assert_close(a['center_velocity'],(after['center']-before['center'])/(2*eps),atol=2e-8,rtol=2e-8)
    torch.testing.assert_close(a['velocity'],(after['position']-before['position'])/(2*eps),atol=2e-8,rtol=2e-8)


def test_gradients_through_legs_arm_root_and_velocity():
    m=FullBodyGeometryV1(ASSET,dtype=torch.float64)
    values=tuple(x.requires_grad_() for x in inputs(1))
    def f(*args):
        out=m(*args)
        return torch.cat((out['position'].flatten(1),out['center'],out['center_velocity']),1)
    assert torch.autograd.gradcheck(f,values,eps=1e-6,atol=2e-5,rtol=1e-4,fast_mode=True)
    out=f(*values)
    grad=torch.autograd.grad(out.square().sum(),values)
    assert all(torch.isfinite(g).all() for g in grad)
    assert grad[0][:,:12].abs().sum()>0 and grad[0][:,12:].abs().sum()>0


def test_asset_identity_and_joint_shape_cannot_silently_change():
    with pytest.raises(ValueError,match='native asset'):
        FullBodyGeometryV1('wrong')
    m=FullBodyGeometryV1(ASSET,dtype=torch.float64)
    q,qd,*rest=inputs()
    with pytest.raises(ValueError,match='named18'):
        m(q[:,:12],qd,*rest)


def test_depth_batch_preserves_all_links_and_derivatives_across_checkpoint_load():
    serial = FullBodyGeometryV1(ASSET, dtype=torch.float64)
    batched = FullBodyGeometryV1(ASSET).double()
    receipt = batched.enable_depth_batches()
    batched.load_state_dict(serial.state_dict(), strict=True)
    assert batched.enable_depth_batches(receipt) == receipt
    with pytest.raises(ValueError, match='differs'):
        batched.enable_depth_batches(dict(receipt, source_sha256='old'))
    assert set(batched.state_dict()) == set(serial.state_dict())

    args = tuple(v.requires_grad_() for v in inputs(5))
    expected, actual = serial(*args), batched(*args)
    generator = torch.Generator().manual_seed(41)
    weights = {k: torch.randn(v.shape, generator=generator, dtype=v.dtype) for k,v in expected.items()}
    for k in expected:
        torch.testing.assert_close(expected[k], actual[k], rtol=1e-12, atol=1e-12)
    grads = [torch.autograd.grad(sum((v * weights[k]).sum() for k,v in outputs.items()), args)
             for outputs in (expected, actual)]
    for first, second in zip(*grads, strict=True):
        torch.testing.assert_close(first, second, rtol=1e-11, atol=1e-11)
