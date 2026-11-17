"""Deterministic CPU contracts for opt-in legal-return rate retargeting.

Synthetic outcome tests validate the gate wiring, not PhysX or a real
badminton contact. Original actuator and ball-side thresholds are unchanged.
"""
import numpy as np
import pytest

from quadsmash.control.impact_feasible_rate_v1 import optimize_impact_feasible_rate


def event():
    return dict(contact_time_s=.27,contact_point_local=np.array([.0006,0.,.07]),
                achieved_position_current_frame_m=np.array([.3,.2,.8]),
                achieved_normal_current_frame=np.array([1.,0.,0.]),
                achieved_velocity_current_frame_m_s=np.zeros(3),
                original_requested_racket_velocity_world_m_s=[10.,0.,0.])


def fake_outcome(metadata):
    v=np.asarray(metadata["achieved_velocity_current_frame_m_s"])
    outgoing=np.array([3.+v[0],v[1],-1.5+v[2]])
    valid=outgoing[0]>=5. and outgoing[0]<6. and abs(outgoing[1])<.2 and outgoing[2]<-1.
    return dict(outgoing_velocity=outgoing.tolist(),contact_feasible=True,
                legal_return_predicted=bool(valid),attack5_predicted=bool(valid),
                target_landing_predicted=bool(valid),ranking_cost=float((outgoing[0]-5.5)**2),
                rejection_reason=None if valid else 'predicted_out_of_court')


def test_retarged_rate_is_task_valid_but_does_not_relabel_ideal_speed():
    J=np.c_[np.eye(3),np.zeros((3,3))]
    old=event();old_velocity=np.array(old['achieved_velocity_current_frame_m_s'])
    result=optimize_impact_feasible_rate(
        J,np.zeros(3),np.zeros(6),np.full(6,5.),np.eye(3),
        old,fake_outcome,[5.5,0.,-1.5],max_nfev=25)
    assert result['accepted']
    assert not result['physical_feasibility_verified']
    assert result['original_requested_velocity_error_preserved']
    assert result['analytic_outcome']['legal_return_predicted']
    assert result['analytic_outcome']['attack5_predicted']
    assert result['analytic_outcome']['target_landing_predicted']
    assert np.max(np.abs(result['qd_rad_s']))<=5.+1e-10
    assert np.linalg.norm(result['achieved_velocity_current_frame_m_s']-[10,0,0])>1.
    np.testing.assert_array_equal(old['achieved_velocity_current_frame_m_s'],old_velocity)


def test_unreachable_valid_return_does_not_get_accepted_or_bypass_speed_bounds():
    J=np.c_[np.eye(3),np.zeros((3,3))]
    r=optimize_impact_feasible_rate(
        J,np.zeros(3),np.zeros(6),np.full(6,.2),np.eye(3),
        event(),fake_outcome,[5.5,0.,-1.5],max_nfev=10)
    assert not r['accepted']
    assert r['qd_rad_s'] is None
    assert r['physical_feasibility_verified'] is False


def test_nonapproaching_impact_without_outgoing_is_not_promoted():
    J=np.c_[np.eye(3),np.zeros((3,3))]
    def rejected(meta):
        return dict(contact_feasible=False,legal_return_predicted=False,
                    attack5_predicted=False,target_landing_predicted=False,
                    rejection_reason='nonapproaching_contact',outgoing_velocity=None)
    r=optimize_impact_feasible_rate(
        J,np.zeros(3),np.zeros(6),np.full(6,5.),np.eye(3),
        event(),rejected,[5.5,0.,-1.5],max_nfev=10)
    assert not r['accepted']
    assert r['failures']['nonapproaching_contact']>0


@pytest.mark.parametrize('invalid',[np.full(6,np.nan),np.ones(6)*6.])
def test_invalid_initial_rate_fails_closed(invalid):
    J=np.c_[np.eye(3),np.zeros((3,3))]
    with pytest.raises(ValueError,match='rate inputs'):
        optimize_impact_feasible_rate(
            J,np.zeros(3),invalid,np.full(6,5.),np.eye(3),
            event(),fake_outcome,[5.5,0.,-1.5])
