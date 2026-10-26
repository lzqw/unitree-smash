import json
from pathlib import Path

import numpy as np

from quadsmash.control.release_flight_envelope_v4 import free_flight,net_gap,target_screen
from quadsmash.physics.shuttlecock import ShuttlecockModel
from quadsmash.control.release_flight_envelope_v4 import rk4

ROOT = Path(__file__).resolve().parents[1]
COURT = json.loads((ROOT/'configs/court_net_v4.json').read_text())
PHYSICS = json.loads((ROOT/'configs/shuttle_physics_v2.json').read_text())


def test_vacuum_first_landing_matches_closed_form():
    physics = dict(PHYSICS,quadratic_drag_k_per_m=0.)
    p,v = np.array([[.95,.2,1.75]]),np.array([[8.,-.3,-2.]])
    result = free_flight(p,v,physics,COURT,dt=.0000625)
    t = (-2.+np.sqrt(4.+2*9.81*(1.75-.025)))/9.81
    expected = p[0]+v[0]*t+np.array([0.,0.,-.5*9.81*t*t])
    assert np.max(abs(result.landing_m[0]-expected)) < 1e-7
    assert abs(result.landing_time_s[0]-t) < 1e-8


def test_batch_integrator_matches_existing_independent_shuttle_model():
    p = np.array([[.95,-.2,1.75],[1.01,.1,1.7]])
    v = np.array([[8.,-.3,-2.],[7.,.5,-1.2]])
    model = ShuttlecockModel();dt=.001
    expected = model.integrate(p,v,dt)
    actual = rk4(p,v,dt,PHYSICS['quadratic_drag_k_per_m'],np.array([0.,0.,-9.81]),np.zeros(3))
    assert all(np.max(abs(a-b)) < 1e-14 for a,b in zip(expected,actual))


def test_net_screen_uses_finite_radius_depth_and_original_posts():
    p = np.array([[1.5,0.,1.6],[1.5,0.,1.53],[1.5,0.,.3],
                  [1.5,COURT['post_y_center_abs_m'],1.]])
    gap = net_gap(p,COURT,.025)
    assert abs(gap[0]-(1.6-1.524-.025)) < 1e-12
    assert gap[1] < 0. and gap[2] > 0. and gap[3] < 0.


def test_collision_free_under_net_flight_cannot_satisfy_original_crossing_gate():
    p,v = np.array([[1.,0.,.6]]),np.array([[8.,0.,-1.]])
    result = free_flight(p,v,PHYSICS,COURT)
    assert 2. < result.landing_m[0,0] < 4. and result.buffered_net_gap_m[0] > 0.
    assert not target_screen(result,v,COURT)[0]


def test_collision_free_around_post_flight_cannot_satisfy_original_net_span_gate():
    p,v = np.array([[1.,5.5,2.1]]),np.array([[3.5,-10.,-1.]])
    result = free_flight(p,v,PHYSICS,COURT)
    assert 2. < result.landing_m[0,0] < 4. and abs(result.landing_m[0,1]) < 1.
    assert result.buffered_net_gap_m[0] > 0. and result.trailing_crossing_m[0,2] > 1.55
    assert not target_screen(result,v,COURT)[0]
