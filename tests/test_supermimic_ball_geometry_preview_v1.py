import torch
from quadsmash.rl.supermimic_ball_geometry_preview_v1 import BallGeometryPreviewV1,GeometrySupportedBallV1
from quadsmash.rl.supermimic_ball_residual_v2 import BallResidualV2
from quadsmash.physics.full_racket_v4 import string_prism_gap,load_racket


def test_string_center_gap_matches_native_string_query():
    g=BallGeometryPreviewV1();p=torch.tensor([[.05,0.,0.],[.02,0.,0.]])
    native=string_prism_gap(p,torch.zeros_like(p),torch.tensor([[1.,0,0,0]]).expand(2,-1),load_racket(),g.radius)
    torch.testing.assert_close(g.racket_gap(p),native,atol=1e-7,rtol=1e-6)


def test_one_tick_contact_window_rejects_early_collision():
    g=BallGeometryPreviewV1();f=torch.zeros(3,25)
    f[:,:3]=torch.tensor([0.,0.,2.]);f[:,6]=torch.tensor([.3,.1,.03]);f[:,9]=-10.
    support,gaps=g(f)
    assert support[0,0]<1e-6 and support[1,0]>.99 and support[2,0]>.99
    assert gaps[0,0]>.15 and support[:,1:].max()<1e-6


def test_ground_preview_uses_actual_one_tick_motion():
    g=BallGeometryPreviewV1();f=torch.zeros(2,25)
    f[:,2]=torch.tensor([.2,.05]);f[:,5]=-5.;f[:,6]=2.
    support,_=g(f)
    assert support[0,1]<1e-6 and support[1,1]>.99


def test_net_band_has_top_and_bottom_not_infinite_plane():
    g=BallGeometryPreviewV1();p=torch.tensor([[1.5,0.,1.2],[1.5,0.,2.],[1.5,0.,.3]])
    gap=g.net_gap(p)
    assert gap[0]<0 and gap[1]>.4 and gap[2]>.4


def test_suppression_does_not_inflate_other_modes_and_keeps_gradients():
    base=BallResidualV2(gate_hidden=(8,),response_hidden=(8,))
    model=GeometrySupportedBallV1(base,BallGeometryPreviewV1())
    f=torch.zeros(2,25);f[:,2]=2.;f[:,6]=.03;f[:,9]=-1.;f.requires_grad_()
    p,_=base.components(f);corrected,_=model.components(f)
    assert (corrected[:,1:]<=p[:,1:]+1e-7).all()
    torch.testing.assert_close(corrected.sum(-1),torch.ones(2))
    grad=torch.autograd.grad(corrected[:,1].sum(),f)[0]
    assert grad.isfinite().all() and grad.abs().sum()>0
    delta,_=model(f,torch.zeros(2,dtype=torch.bool));assert delta.eq(0).all()
