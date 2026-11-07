"""Local terminal rows must mean the same thing as the original verifier."""
import copy
import numpy as np
import pinocchio as pin
import pytest
from test_differentiated_sequence_v1 import setup_profile
from scripts.terminal_recovery_constraints_v1 import terminal_values,terminal_rows,make_proposal


@pytest.mark.parametrize('feet',[1,2,3,4])
def test_values_match_original_terminal_gate(feet):
    c,s,_=setup_profile();s.contacts[:]=False;s.contacts[:feet]=True
    for speed in [0.,.2,2.]:
        s.v[:6]=speed
        values,limits=terminal_values(c,s)
        assert bool(np.all(values<=limits))==c._terminal_ok(s,c.risk(s))


def test_terminal_gradient_matches_directional_perturbation():
    c,s,_=setup_profile();s.v[:6]=[.1,.07,-.02,.03,-.08,.04]
    d=np.zeros(24);d[3:6]=[.2,-.15,.03];s.q=pin.integrate(c.qp.model.model,s.q,d)
    s.root_rotation_world=pin.Quaternion(s.q[3:7]).matrix()
    qm=np.random.default_rng(71).normal(0,.03,(24,3));vm=np.random.default_rng(72).normal(0,.02,(24,3))
    values,jac,limits=terminal_rows(c,s,qm,vm);direction=np.array([.3,.2,-.1]);h=1e-4;responses=[]
    for sign in [-1.,1.]:
        t=copy.deepcopy(s);t.q=pin.integrate(c.qp.model.model,s.q,sign*h*qm@direction)
        t.root_position_world=t.q[:3].copy();t.root_rotation_world=pin.Quaternion(t.q[3:7]).matrix()
        t.joint_position=t.q[c.qp.model.q_indices].copy();t.v=s.v+sign*h*vm@direction
        t.joint_velocity=t.v[c.qp.model.v_indices].copy();responses.append(terminal_values(c,t)[0])
    np.testing.assert_allclose(jac@direction,(responses[1]-responses[0])/(2*h),atol=1e-6,rtol=1e-4)
    assert callable(make_proposal())
