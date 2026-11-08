from types import SimpleNamespace
import numpy as np
import pinocchio as pin
from scripts.authored_net_clearance_v1 import AuthoredNetClearanceV1
from quadsmash.physics.court_v4 import load_court


def fixture(position,kind='Cube'):
    transform=np.eye(4)
    shape=dict(body='probe',kind=kind,transform=transform,size=.1)
    if kind=='Mesh':shape['points']=np.array([[x,y,z] for x in [-.05,.05] for y in [-.05,.05] for z in [-.05,.05]])
    model=SimpleNamespace(body_frames={'probe':0},data=SimpleNamespace(oMf=[pin.SE3(np.eye(3),np.asarray(position))]))
    geometry=SimpleNamespace(model=model,shapes=[shape])
    return AuthoredNetClearanceV1(geometry,load_court())


def test_band_is_not_an_infinite_plane():
    net=fixture([1.5,0.,.4]);r=net.evaluate(None,placements_current=True,near_distance_m=10.)['probe']
    assert r['distance_m']>.3
    net=fixture([1.5,0.,1.2]);r=net.evaluate(None,placements_current=True)['probe']
    assert r['exact'] and r['distance_m']<0


def test_front_gap_and_authored_mesh_agree():
    ds=[fixture([1.3,0.,1.2],kind).evaluate(None,placements_current=True,near_distance_m=10.)['probe']['distance_m'] for kind in ['Cube','Mesh']]
    np.testing.assert_allclose(ds,[.148,.148],atol=2e-7,rtol=0.)


def test_post_collision_is_included():
    net=fixture([1.5,3.075,.7]);r=net.evaluate(None,placements_current=True)['probe']
    assert r['exact'] and r['distance_m']<0 and r['obstacle'].startswith('post_')
