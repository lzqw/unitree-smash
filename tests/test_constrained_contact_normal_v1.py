"""Production constrained contact-normal refinement regression, no Isaac needed."""
import numpy as np
import pytest

from quadsmash.control.constrained_contact_normal_v1 import refine_contact_normal
from quadsmash.control.position_anchored_contact_search_v1 import anchored_pose_velocity


def coupled_fk(q):
    """Contact normal depends on a position-controlled joint.

    At the exact position anchor the position nullspace cannot change the
    normal at all; a physically legal 0.5-mm position budget can.
    """
    angle = 1000. * q[0]
    point = q[:3].copy()
    normal = np.array([np.cos(angle), np.sin(angle), 0.])
    jp = np.c_[np.eye(3), np.zeros((3,3))]
    jn = np.zeros((3,6))
    jn[:,0] = 1000. * np.array([-np.sin(angle), np.cos(angle), 0.])
    return point, normal, jp, jn


def test_second_stage_can_reduce_normal_error_without_losing_submillimetre_contact():
    seed=np.zeros(6)
    desired=np.array([np.cos(.30),np.sin(.30),0.])
    r=anchored_pose_velocity(coupled_fk,[seed],-np.ones(6),np.ones(6),
                            np.ones(6),np.zeros(3),desired,np.zeros(3),
                            normal_steps=4)
    assert r['constrained_normal_refinement']['attempted']
    assert r['constrained_normal_refinement']['accepted']
    assert r['NORMAL_FEASIBLE']
    assert r['normal_error']<.12
    assert r['position_error_m']<.0005+1e-9
    assert r['position_error_m']<=r['position_preservation_bound_m']+1e-9
    np.testing.assert_array_equal(r['q_anchor'],seed)
    np.testing.assert_array_equal(seed,np.zeros(6))


def test_incompatible_normal_never_fabricates_feasibility():
    # The required normal can only be reached by moving the contact point
    # outside the original 3-mm gate.
    def slowly_turning_fk(q):
        point=q[:3].copy()
        angle=q[0]
        normal=np.array([np.cos(angle),np.sin(angle),0.])
        jp=np.c_[np.eye(3),np.zeros((3,3))]
        jn=np.zeros((3,6))
        jn[:,0]=np.array([-np.sin(angle),np.cos(angle),0.])
        return point,normal,jp,jn
    desired=np.array([0.,1.,0.])
    r=anchored_pose_velocity(slowly_turning_fk,[np.zeros(6)],-np.ones(6),
                            np.ones(6),np.ones(6),np.zeros(3),desired,
                            np.zeros(3),normal_steps=3)
    assert not r['NORMAL_FEASIBLE']
    assert not r['velocity_solver_executed']
    assert r['position_error_m']<=.0005+1e-9
    assert r['rejection']=='POSE_TASK_INCOMPATIBLE'


@pytest.mark.parametrize('budget',[-1.,0.,.004,float('nan')])
def test_refiner_fails_closed_for_invalid_position_budget(budget):
    with pytest.raises(ValueError,match='position budget'):
        refine_contact_normal(coupled_fk,np.zeros(6),np.zeros(6),
                              -np.ones(6),np.ones(6),np.zeros(3),
                              [1.,0.,0.],budget)


def test_strictly_feasible_normal_preserves_original_anchor():
    seed=np.array([.1,.2,0.,0.,0.,0.])
    point,normal,*_=coupled_fk(seed)
    r=anchored_pose_velocity(coupled_fk,[seed],-np.ones(6),
                            np.ones(6),np.ones(6),point,normal,
                            np.zeros(3),normal_steps=3)
    assert r['POSITION_ANCHORED']
    assert r['NORMAL_FEASIBLE']
    assert not r['constrained_normal_refinement']['attempted']
    np.testing.assert_allclose(r['position'],point,atol=1e-9)
