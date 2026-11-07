"""Control-memory regressions; these do not claim native landing success."""
import copy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from scripts.continuous_support_qp_v1 import ContinuousSupportQP
from scripts.whole_body_support_qp_v1 import WholeBodySupportQP
from scripts.landing_support_mpc_v1 import LandingSupportMPCV1
from scripts.landing_support_mpc_v2 import LandingSupportMPCV2
from scripts.landing_support_mpc_v3 import LandingSupportMPCV3


def fixture():
    # Exercise the production critical-damping reference and memory methods;
    # only the rigid-body snapshot/raw goal are supplied as a small fixture.
    q = ContinuousSupportQP.__new__(ContinuousSupportQP)
    q.interface = SimpleNamespace(arm_reference=None)
    q.continuous_reference = True
    q.load_aware = False
    q.reference_frequency_rad_s = 10.
    q.settings = SimpleNamespace(control_dt_s=.01, com_kp_s2=50., com_kd_s=14.)
    q.model = SimpleNamespace(names=list('abcd'), feet=list('abcd'))
    q.model.snapshot = lambda pos, vel: dict(
        body_positions_world=np.array([[-.2,-.1,0.],[.2,-.1,0.],[-.2,.1,0.],[.2,.1,0.]])+pos,
        com_world=pos+np.array([0.,0.,.3]), com_velocity_world=vel.copy())
    q.reset()
    c = LandingSupportMPCV3.__new__(LandingSupportMPCV3)
    c.qp = q
    c.posture_target = None
    c._live_reference = {}; c._prediction_reference = {}; c._observed_tick = {}
    return c


def state(x):
    return SimpleNamespace(q=np.array([x,0.,0.]), v=np.array([.4,0.,0.]),
                           contacts=np.ones(4,dtype=bool))


def raw_reference(self, s, snap, target):
    return dict(com_position=s.q+np.array([0.,0.,.3]))


def assert_memory_equal(a,b):
    assert set(a)==set(b)
    for key in a:
        np.testing.assert_equal(a[key],b[key])


@patch.object(WholeBodySupportQP, 'reference', raw_reference)
def test_interleaved_lanes_match_independent_references_and_selective_reset():
    combined, a, b = fixture(), fixture(), fixture()
    for tick in range(8):
        for c,lane,x in [(combined,1,0.),(combined,67,3.),(a,1,0.),(b,67,3.)]:
            c.observe(state(x),np.zeros(18),lane=lane,control_tick=tick)
        assert_memory_equal(combined._live_reference[1],a._live_reference[1])
        assert_memory_equal(combined._live_reference[67],b._live_reference[67])
    saved=copy.deepcopy(combined._live_reference[67])
    combined.reset(lane=1)
    assert 1 not in combined._live_reference
    assert_memory_equal(combined._live_reference[67],saved)
    combined.observe(state(3.),np.zeros(18),lane=67,control_tick=8)
    b.observe(state(3.),np.zeros(18),lane=67,control_tick=8)
    assert_memory_equal(combined._live_reference[67],b._live_reference[67])


@patch.object(WholeBodySupportQP, 'reference', raw_reference)
def test_prediction_length_and_exception_do_not_commit_live_memory():
    c=fixture(); s=state(0.); target=np.zeros(18)
    c.observe(s,target,lane=1,control_tick=0)
    saved=copy.deepcopy(c._live_reference[1])
    def forecast(self,s,target,force=False):
        for _ in range(40):
            self.qp.reference(s,self.qp.model.snapshot(s.q,s.v),target)
        raise RuntimeError('prediction rejected')
    with patch.object(LandingSupportMPCV2,'solve',forecast):
        with pytest.raises(RuntimeError,match='prediction rejected'):
            c.solve(s,target,lane=1,control_tick=0)
    assert_memory_equal(c._live_reference[1],saved)
    assert_memory_equal(c.qp.reference_state(),saved)
    c.observe(s,target,lane=1,control_tick=0)  # same tick is idempotent
    assert_memory_equal(c._live_reference[1],saved)


@patch.object(WholeBodySupportQP, 'reference', raw_reference)
def test_observation_during_held_action_advances_exactly_one_tick():
    c=fixture(); reference=fixture(); s=state(0.); target=np.zeros(18)
    for tick in range(10):
        c.observe(s,target,lane=1,control_tick=tick)
        reference.observe(s,target,lane=1,control_tick=tick)
        with patch.object(LandingSupportMPCV2,'solve',return_value={'accepted':False}):
            if tick in (0,9): c.solve(s,target,lane=1,control_tick=tick)
    assert_memory_equal(c._live_reference[1],reference._live_reference[1])
    with pytest.raises(ValueError,match='clock restarts'):
        c.observe(s,target,lane=1,control_tick=0)


def test_contact_alternatives_do_not_consume_reference_time():
    class QP:
        def __init__(self): self.time=0; self.starts=[]
        def reference_state(self): return self.time
        def restore_reference_state(self,t): self.time=t
        def solve(self,s,target):
            self.starts.append(self.time); self.time+=1
            return dict(accepted=True,acceleration=np.zeros(1),torque=np.zeros(18),
                        normalized_action=np.zeros(18),joint_position_target=np.zeros(18))
    class MPC:
        solve=LandingSupportMPCV1.solve
        def __init__(self):
            self.qp=QP(); self.settings=SimpleNamespace(horizon_steps=2,minimum_support_feet=1,
                contact_mode_sequence_search=True,contact_mode_sequence_max_candidates=3,
                terminal_tilt_rad=1.,terminal_angular_speed_rad_s=1.,terminal_horizontal_speed_mps=1.)
        def risk(self,s):
            return dict(trigger=True,tilt_rad=0.,angular_speed_rad_s=0.,horizontal_speed_mps=0.)
        def risk_score(self,r): return 0.
        def _contact_mode_candidates(self,s):
            candidate=copy.deepcopy(s);candidate.contacts[1]=True
            return [(candidate,[1])]
        def _integrate(self,s,a,plan):return copy.deepcopy(s)
        def _after_prediction(self,s):return s,[]
        def contract(self):return {}
    c=MPC(); s=SimpleNamespace(contacts=np.array([True,False,False,False]),
                               joint_position=np.zeros(18),joint_velocity=np.zeros(18))
    assert c.solve(s,np.zeros(18),force=True)['accepted']
    assert c.qp.starts == [0,0,1,1]
    assert c.qp.time == 2
