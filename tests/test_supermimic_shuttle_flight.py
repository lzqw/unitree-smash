import torch
import pytest

from quadsmash.physics.native import physics_contract
from quadsmash.rl.supermimic_shuttle_flight_v1 import ShuttleFreeFlightV1


def test_gravity_only_matches_discrete_equations_not_continuous_shortcut():
    cfg=dict(physics_contract(),quadratic_drag_k_per_m=0.)
    m=ShuttleFreeFlightV1(contract=cfg,dtype=torch.float64)
    p=torch.zeros(2,3,dtype=torch.float64)
    v=torch.tensor([[1.,2.,3.],[-1.,0.,0.]],dtype=p.dtype)
    pn,vn=m(p,v)
    torch.testing.assert_close(vn,v+.01*m.gravity,atol=1e-12,rtol=0)
    expected=p+.01*v+m.dt*m.dt*(m.steps*(m.steps+1)/2)*m.gravity
    torch.testing.assert_close(pn,expected,atol=1e-12,rtol=0)
    assert torch.equal(p,torch.zeros_like(p))


def test_drag_reduces_relative_speed_and_preserves_direction_without_gravity():
    m=ShuttleFreeFlightV1(gravity=(0,0,0),dtype=torch.float64)
    v=torch.tensor([[10.,0,0],[0,-10.,0]],dtype=torch.float64)
    _,vn=m(torch.zeros_like(v),v)
    assert (vn.norm(dim=-1)<v.norm(dim=-1)).all()
    assert vn[0,1:].eq(0).all() and vn[1,[0,2]].eq(0).all()
    assert vn[0,0]>0 and vn[1,1]<0


def test_flight_preserves_position_and_velocity_input_gradients():
    m=ShuttleFreeFlightV1(dtype=torch.float64)
    p=torch.tensor([[1.,2.,3.]],dtype=torch.float64,requires_grad=True)
    v=torch.tensor([[5.,-2.,1.]],dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(m,(p,v),fast_mode=True)


def test_reject_fractional_substeps():
    with pytest.raises(ValueError,match='whole native substeps'):
        ShuttleFreeFlightV1(physics_dt_s=.003)
