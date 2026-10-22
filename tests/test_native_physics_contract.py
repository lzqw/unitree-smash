"""Independent analytical geometry, force and event regressions for G02."""
import math
import numpy as np
import pytest
import torch
from quadsmash.physics.native import ContactEpisodes, drag_force, segment_plane_crossing, sphere_box_gap, swept_sphere_box
from quadsmash.physics.sweep_reference import sphere_box_sweep
from quadsmash.physics.shuttlecock import ShuttlecockModel, ShuttlecockParams, ShuttlecockState
from quadsmash.physics.collision import RacketGeometry, resolve_racket_collision_with_reaction

IDENTITY = [1, 0, 0, 0]

@pytest.mark.parametrize("y", [-.16, -.13, -.11, 0., .11, .13, .16])
@pytest.mark.parametrize("z", [-.16, -.13, -.11, 0., .11, .13, .16])
def test_translating_box_49_analytic_paths(y, z):
    # In the box frame the center traverses from +.3 to -.3 in x.
    yz = np.linalg.norm(np.maximum(np.abs([y, z]) - .11, 0))
    hit = sphere_box_sweep([.3,y,z], [-.2,y,z], [0,0,0], [.1,0,0], IDENTITY, IDENTITY)
    assert (hit is not None) == (yz < .025)
    if hit:
        exact = (.3 - (.0075 + math.sqrt(.025**2-yz**2))) / .6
        assert abs(hit[0]-exact) < 1e-6

@pytest.mark.parametrize("distance", [.04,.05,.06,.07,.08,.09,.10])
def test_rotating_face_7_analytic_paths(distance):
    hit = sphere_box_sweep([distance,0,0], [distance,0,0], [0,0,0], [0,0,0],
                           IDENTITY, [math.sqrt(.5),0,0,math.sqrt(.5)])
    exact = math.acos(.0325/distance) / (math.pi/2)
    assert hit is not None and abs(hit[0]-exact) < 1e-6

def test_force_episode_rearm_and_jitter():
    state = ContactEpisodes(2, "cpu")
    contact = torch.tensor([[2.,0,0],[0.,0,0]])
    for _ in range(10):
        state.update(contact, torch.zeros(2))
    assert state.count.tolist() == [1,0]
    # A one-frame dropout and proximity without force are not new hits.
    state.update(torch.zeros(2,3), torch.ones(2))
    state.update(contact, torch.zeros(2))
    assert state.count.tolist() == [1,0]
    for _ in range(2):
        state.update(torch.zeros(2,3), torch.ones(2))
    state.update(contact, torch.zeros(2))
    assert state.count.tolist() == [2,0]
    state.reset(torch.tensor([0]))
    assert state.count.tolist() == [0,0]

def test_net_and_landing_interpolation():
    a, b = torch.tensor([[1.,0.,2.],[0.,0.,.1]]), torch.tensor([[2.,0.,1.],[.4,0.,-.1]])
    hit, fraction, point = segment_plane_crossing(a,b,0,1.5)
    assert hit.tolist() == [True,False] and fraction[0] == .5 and point[0,2] == 1.5
    hit, fraction, point = segment_plane_crossing(a,b,2,.025,-1)
    assert hit.tolist() == [False,True]
    assert torch.allclose(point[1], torch.tensor([.15,0,.025]), atol=1e-7)

@pytest.mark.parametrize("dt", [.002,.001,.0005])
def test_wind_drag_analytic(dt):
    params = ShuttlecockParams(gravity=(0,0,0), wind=(3,0,0))
    model = ShuttlecockModel(params)
    initial = ShuttlecockState(np.zeros(3), np.array([23.,0,0]))
    final = model.rollout(initial,.4,dt)[-1]
    k = .5*params.air_density*params.drag_coefficient*params.reference_area/params.mass
    assert abs(final.velocity[0]-(3+20/(1+k*20*.4))) < 1e-8
    assert abs(final.position[0]-(3*.4+math.log1p(k*20*.4)/k)) < 1e-8
    actual = drag_force(torch.tensor([[23.,0,0]],dtype=torch.float64),params.mass,k,params.wind)
    assert np.allclose(actual.numpy()[0]/params.mass, model.acceleration(initial.velocity),atol=1e-12)

def test_finite_mass_friction_impulse_matches_both_bodies():
    ball = np.array([-12.,4.,2.])
    racket = np.array([2.,-1.,0.])
    vb, vr, event = resolve_racket_collision_with_reaction(ball,RacketGeometry(np.zeros(3),[1,0,0]),
        racket_velocity=racket,racket_mass=.09,shuttle_mass=.0052)
    assert np.allclose(event.reaction_impulse,.09*(vr-racket),atol=1e-12)
    assert np.allclose(event.reaction_impulse,-.0052*(vb-ball),atol=1e-12)
    assert abs((vb-vr)[0]+.72*(ball-racket)[0]) < 1e-12

def test_rotated_geometry_and_invalid_quaternion():
    gap = sphere_box_gap(torch.tensor([[0.,.04,0.]]),torch.zeros(1,3),
        torch.tensor([[math.sqrt(.5),0,0,math.sqrt(.5)]]),[.0075,.11,.11],.025)
    assert abs(gap.item()-.0075) < 1e-7
    with pytest.raises(ValueError):
        sphere_box_sweep([0,0,0],[1,0,0],[0,0,0],[0,0,0],[0,0,0,0],IDENTITY)

def test_batched_rotating_sweep_matches_independent_analytic():
    distances = torch.linspace(.04,.1,7,dtype=torch.float64)
    point = torch.stack((distances,torch.zeros(7),torch.zeros(7)),dim=-1)
    q0 = torch.tensor([IDENTITY]*7,dtype=torch.float64)
    q1 = torch.tensor([[math.sqrt(.5),0,0,math.sqrt(.5)]]*7,dtype=torch.float64)
    hit,fraction,unresolved = swept_sphere_box(point,point,torch.zeros_like(point),torch.zeros_like(point),q0,q1,[.0075,.11,.11],.025)
    assert hit.all() and not unresolved.any()
    assert torch.allclose(fraction,torch.acos(.0325/distances)/(math.pi/2),atol=1e-4)
