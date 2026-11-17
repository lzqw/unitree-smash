import numpy as np
import pytest
from quadsmash.control.velocity_aware_contact_pose_v1 import refine_velocity_pose,velocity_support
from quadsmash.control.position_anchored_contact_search_v1 import anchored_pose_velocity


def fk(q):
    p=np.array([(1+q[5])*q[0],q[1],q[2]])
    J=np.zeros((3,6));J[0,0]=1+q[5];J[0,5]=q[0];J[1,1]=J[2,2]=1
    return p,np.array([1.,0,0]),J,np.zeros((3,6))


def test_redundant_configuration_improves_speed_without_moving_contact():
    q=np.zeros(6);qd=np.array([1.,0,0,0,0,0])
    r=refine_velocity_pose(fk,q,qd,-np.ones(6),np.ones(6),np.ones(6),
        np.zeros(3),[1.,0,0],[1.8,0,0])
    assert r['improved'] and r['velocity_error_m_s']<1e-4
    assert r['position_error_m']<=.0005 and r['normal_error']<=.12
    assert np.max(abs(r['qd_rad_s']))<=1+1e-10
    assert r['q_rad'][5]>.5
    np.testing.assert_array_equal(q,np.zeros(6))


def test_production_opt_in_preserves_anchor_and_original_request():
    args=(fk,[np.zeros(6)],-np.ones(6),np.ones(6),np.ones(6),np.zeros(3),[1.,0,0],[1.8,0,0])
    old=anchored_pose_velocity(*args)
    new=anchored_pose_velocity(*args,velocity_aware=True)
    assert not old['VELOCITY_FEASIBLE'] and new['VELOCITY_FEASIBLE']
    np.testing.assert_array_equal(new['q_anchor'],old['q_anchor'])
    np.testing.assert_array_equal(new['requested_velocity'],[1.8,0,0])
    assert new['requested_velocity_error_m_s']<old['requested_velocity_error_m_s']


def test_directional_support_is_optimistic_box_bound_with_base_contribution():
    J=np.c_[np.eye(3),np.eye(3)]
    r=velocity_support(J,[.5,0,0],[4.,0,0],np.ones(6))
    assert r['optimistic_projection_upper_m_s']==pytest.approx(2.5)
    assert r['optimistic_shortfall_m_s']==pytest.approx(1.5)
    assert not r['physical_feasibility_verified']


def test_no_pose_budget_widening():
    with pytest.raises(ValueError):
        refine_velocity_pose(fk,np.zeros(6),np.zeros(6),-np.ones(6),np.ones(6),np.ones(6),
            np.zeros(3),[1.,0,0],[2.,0,0],position_budget_m=.003)
