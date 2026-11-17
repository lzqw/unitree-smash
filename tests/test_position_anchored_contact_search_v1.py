import numpy as np
import pytest
from quadsmash.control.position_anchored_contact_search_v1 import anchored_pose_velocity, position_anchor
from scripts.contact_first_fullbody_evaluator_v1 import ContactFirstFullBodyEvaluatorV1


def fk(q):
    n=np.array([np.cos(q[3]),np.sin(q[3]),0.])
    jp=np.c_[np.eye(3),np.zeros((3,3))]
    jn=np.zeros((3,6));jn[:,3]=[-np.sin(q[3]),np.cos(q[3]),0.]
    return q[:3].copy(),n,jp,jn


def test_nullspace_normal_correction_preserves_independent_position_anchor():
    target=np.array([.1,.2,.3]);seed=np.zeros(6)
    result=anchored_pose_velocity(fk,[seed],-np.ones(6)*2,np.ones(6)*2,np.ones(6)*2,
                                  target,[np.cos(.6),np.sin(.6),0.],[.2,.1,0.])
    assert result['NORMAL_FEASIBLE'] and result['VELOCITY_FEASIBLE']
    assert result['normal_error'] < result['anchor_normal_error']
    assert result['position_error_m'] <= result['position_preservation_bound_m']
    np.testing.assert_allclose(result['q_anchor'][3:],0.)
    np.testing.assert_array_equal(seed,np.zeros(6))
    result['q_rad'][:]=99
    assert np.max(abs(result['q_anchor']))<1


def test_incompatible_pose_keeps_anchor_and_never_solves_speed():
    def coupled(q):
        p,n,jp,jn=fk(q)
        n=np.array([np.cos(q[0]),np.sin(q[0]),0.]);jn[:]=0
        jn[:,0]=[-np.sin(q[0]),np.cos(q[0]),0.]
        return p,n,jp,jn
    r=anchored_pose_velocity(coupled,[np.zeros(6)],-np.ones(6),np.ones(6),np.ones(6),
                             np.zeros(3),[0.,1.,0.],[1.,0.,0.])
    assert r['POSITION_ANCHORED'] and not r['NORMAL_FEASIBLE']
    assert r['rejection']=='POSE_TASK_INCOMPATIBLE'
    np.testing.assert_allclose(r['qd_rad_s'],0.)
    assert r['position_error_m']<1e-8


def test_bounded_velocity_retains_original_request_error():
    r=anchored_pose_velocity(fk,[np.zeros(6)],-np.ones(6),np.ones(6),np.ones(6)*.1,
                             np.zeros(3),[1.,0.,0.],[3.,0.,0.])
    assert r['NORMAL_FEASIBLE'] and not r['VELOCITY_FEASIBLE']
    assert r['requested_velocity_error_m_s']>2.8
    np.testing.assert_allclose(r['requested_velocity'],[3,0,0])


def test_each_target_keeps_its_own_best_position():
    a=position_anchor(fk,[np.zeros(6)],-np.ones(6),np.ones(6),np.array([.1,0,0]))
    b=position_anchor(fk,[np.zeros(6)],-np.ones(6),np.ones(6),np.array([.2,0,0]))
    assert a['position_error_anchor_m']<1e-8 and b['position_error_anchor_m']<1e-8
    np.testing.assert_allclose(a['q_anchor'][0],.1)
    np.testing.assert_allclose(b['q_anchor'][0],.2)


@pytest.mark.parametrize('bad',[None,float('nan'),float('inf'),-1.])
def test_physics_diagnostics_fail_closed(bad):
    d=dict(dynamics_residual_max=0.,joint_velocity_violation_rad_s=0.,
        joint_position_violation_rad=0.,normal_force_violation_n=0.,
        friction_violation_n=0.,motor_torque_violation_nm=0.)
    assert not ContactFirstFullBodyEvaluatorV1._violations(d)
    d['joint_position_violation_rad']=bad
    assert ContactFirstFullBodyEvaluatorV1._violations(d)
    del d['joint_position_violation_rad']
    assert ContactFirstFullBodyEvaluatorV1._violations(d)


def test_production_anchored_branch_keeps_per_time_anchor_and_requires_final_impact():
    from quadsmash.control.contact_first_trajectory_search_v1 import ContactFirstTrajectorySearchV1,ContactFirstSearchConfig
    from quadsmash.control.inverse_impact_target_v1 import UnreachableImpactTarget
    planner=ContactFirstTrajectorySearchV1(ContactFirstSearchConfig(position_anchored_search=True,horizon_steps=32))
    inputs=dict(q_rad=np.zeros(6),qd_rad_s=np.zeros(6),position_jacobian=fk(np.zeros(6))[2],
        normal_jacobian=fk(np.zeros(6))[3],racket_position_world_m=np.zeros(3),racket_normal_world=np.array([1.,0.,0.]),
        target_position_world_m=np.array([.1,0,0]),target_normal_world=np.array([1.,0,0]),
        target_velocity_world_m_s=np.array([.2,0,0]),contact_time_s=.2,
        lower_rad=-np.ones(6),upper_rad=np.ones(6),velocity_limits_rad_s=np.ones(6)*5)
    def target(t):
        if abs(t-.2)<1e-9:raise UnreachableImpactTarget('nominal unavailable')
        return dict(position_world_m=np.array([t/2,0,0]),normal_world=np.array([1.,0,0]),velocity_world_m_s=np.array([.2,0,0]),
            original_requested_velocity_current_frame_m_s=[.4,0,0])
    calls=[]
    def physical(q,qd,index,metadata):
        calls.append(index)
        return dict(accepted=True,actual_racket_position_world_m=q[index,:3],
            actual_racket_normal_world=np.array([1.,0,0]),actual_racket_velocity_world_m_s=qd[index,:3])
    def impact(meta):
        return dict(legal_return_predicted=True,attack5_predicted=True,target_landing_predicted=True,ranking_cost=0.)
    out=planner.plan(**inputs,kinematics_fn=fk,target_state_fn=target,evaluator=physical,impact_evaluator=impact)
    assert calls==[18,22]
    assert out.contact_index in (18,22)
    anchors=[x for x in planner.last_diagnostics['anchors'] if 'q_anchor' in x]
    np.testing.assert_allclose([x['q_anchor'][0] for x in anchors],[.09,.11],atol=1e-8)
    assert all(x['ANALYTIC_RETURN_VALID'] for x in anchors)
    assert all(x['original_requested_velocity_error_m_s']==pytest.approx(.2,abs=1e-5) for x in anchors)
    calls.clear()
    with pytest.raises(RuntimeError):
        planner.plan(**inputs,kinematics_fn=fk,target_state_fn=target,evaluator=physical,
                     impact_evaluator=lambda m:dict(legal_return_predicted=True,attack5_predicted=False))
    assert not calls
