import torch
from quadsmash.rl.supermimic_ball_impulse_prior_v1 import box_normal,round_normal,rotate_for_time,BallImpulsePriorV1
from quadsmash.rl.supermimic_shuttle_flight_v1 import ShuttleFreeFlightV1


def test_box_normal_and_capsule_normal_have_expected_direction():
    p=torch.tensor([[.05,0.,0.],[0.,.2,0.]])
    n=box_normal(p,torch.zeros(1,3),torch.tensor([[.001,.1,.1]]))[:,0]
    torch.testing.assert_close(n,torch.tensor([[1.,0.,0.],[0.,1.,0.]]))
    n=round_normal(torch.tensor([[1.,0.,0.]]),torch.zeros(1,3),torch.tensor([[0.,0.,1.]]),
        torch.tensor([.5]),torch.tensor([.1]),torch.tensor([True]))[:,0]
    torch.testing.assert_close(n,torch.tensor([[1.,0.,0.]]))


def test_current_angular_velocity_rotates_surface_normal():
    r=rotate_for_time(torch.eye(3)[None],torch.tensor([[0.,0.,torch.pi]]),torch.tensor([.5]))
    torch.testing.assert_close(r[0,:,0],torch.tensor([0.,1.,0.]),atol=2e-7,rtol=0)
    r=rotate_for_time(torch.eye(3)[None],torch.zeros(1,3),torch.tensor([.01]))
    torch.testing.assert_close(r,torch.eye(3)[None])


def test_uninterrupted_free_flight_matches_existing_substeps():
    m=BallImpulsePriorV1();p=torch.tensor([[0.,0.,2.],[1.,0.,3.]])
    v=torch.tensor([[5.,1.,-2.],[-4.,0.,-1.]])
    actual=m.advance(p,v,torch.full((2,),.01));expected=ShuttleFreeFlightV1()(p,v)
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,atol=2e-6,rtol=0)


def test_normal_and_rotation_gradients_are_finite():
    p=torch.tensor([[.05,.02,.01]],requires_grad=True)
    n=box_normal(p,torch.zeros(1,3),torch.tensor([[.001,.1,.1]]))
    g=torch.autograd.grad(n.sum(),p)[0];assert g.isfinite().all()
    w=torch.zeros(1,3,requires_grad=True)
    r=rotate_for_time(torch.eye(3)[None],w,torch.tensor([.01]))
    g=torch.autograd.grad(r[:,0,1].sum(),w)[0]
    assert g.isfinite().all() and g.abs().sum()>0


def test_full_forward_ground_impact_and_absent_ball():
    import json
    from quadsmash.rl.supermimic_full_body_geometry_v1 import FullBodyGeometryV1, MODEL
    geometry=FullBodyGeometryV1(json.loads(MODEL.read_text())['asset_sha256'])
    state=torch.zeros(3,55);state[:,0]=10.;state[:,2]=2.;state[:,3]=1.
    state[:,49:52]=torch.tensor([0.,0.,.05]);state[:,54]=-5.
    state[1,51]=2.
    present=torch.tensor([True,True,False])
    model=BallImpulsePriorV1()
    actual,info=model(state,geometry,present)
    assert info['active'].tolist()==[True,False,False]
    assert info['object_id'].tolist()==[1,3,3]
    torch.testing.assert_close(info['normal'][0],torch.tensor([0.,0.,1.]))
    assert .004 < info['time'][0] < .006
    assert actual[0,2] > model.preview.radius and 3.4 < actual[0,5] < 3.8
    torch.testing.assert_close(actual[2],state[2,49:],atol=0,rtol=0)
    p,v=ShuttleFreeFlightV1()(state[1:2,49:52],state[1:2,52:])
    torch.testing.assert_close(actual[1],torch.cat((p,v),-1)[0],atol=2e-6,rtol=0)
