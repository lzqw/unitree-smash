import json
from pathlib import Path
import numpy as np
import pytest
from quadsmash.control.task_constrained_impact_goal_v2 import task_margins,attack_geometry,optimize_task_goal
from quadsmash.control.task_constrained_hit_search_v2 import optimize_task_hit
from quadsmash.control.impact_outcome_evaluator_v1 import ImpactOutcomeEvaluator
from quadsmash.physics.court_v4 import load_court
from quadsmash.physics.full_racket_v4 import load_racket


def model():
    return ImpactOutcomeEvaluator(json.loads(Path('configs/shuttle_physics_v2.json').read_text()),load_court(),load_racket())


def test_near_ideal_477_speed_cannot_be_task_valid_even_if_flags_claim_true():
    out=dict(outgoing_velocity=[4.13897,-2.07604,-1.14615],landing_position=[3,0,.025],crossing_position=[1.525,0,2],net_clearance=.451,
        contact_feasible=True,legal_return_predicted=True,attack5_predicted=True,target_landing_predicted=True)
    m=task_margins(out,load_court())
    assert not m['valid'] and m['physical']['speed']<0 and m['limiting_constraint']=='speed'
    out['outgoing_velocity']=[6,0,-1.2];out['net_clearance']=None
    assert not task_margins(out,load_court())['valid']


def test_height_is_necessary_not_a_success_certificate():
    m=model()
    r=attack_geometry([.7,0,1.549],m.physics,m.court)
    assert not r['downward_attack_geometrically_possible']
    assert r['unrealized_takeoff_requirements']['implemented'] is False
    assert attack_geometry([.7,0,2.],m.physics,m.court)['necessary_condition_only']
    assert not attack_geometry([2,0,3.],m.physics,m.court)['downward_attack_geometrically_possible']


def test_forward_constrained_goal_cannot_promote_below_net_or_weak_seed():
    m=model();r=optimize_task_goal(m,[.7,0,1.],[-8,0,-3],[.0006,0,.07],max_iterations=3)
    assert not r['accepted'] and r['initialization_only'] and not r['robot_feasibility_verified']
    r=optimize_task_goal(m,[.7,0,2.],[-8,0,-3],[.0006,0,.07],max_iterations=8)
    assert r['accepted']
    assert r['task_margins']['gamma']>=0
    assert r['forward_outcome']['attack5_predicted']


def test_joint_optimizer_changes_normal_and_rate_without_moving_anchor():
    # Analytic synthetic collision isolates joint optimization wiring, NOT PhysX.
    court=load_court();incoming=np.array([-2.,0.,0.]);q=np.zeros(6);qd=np.zeros(6)
    def fk(q,local):
        n=np.array([np.cos(q[3]),0,-np.sin(q[3])]);J=np.c_[np.eye(3),np.zeros((3,3))]
        return q[:3],n,J,np.zeros((3,6))
    def impact(meta):
        n=meta['achieved_normal_current_frame'];v=meta['achieved_velocity_current_frame_m_s']
        out=incoming+1.72*max(0.,(v-incoming)@n)*n
        return dict(outgoing_velocity=out.tolist(),landing_position=[3,0,.025],crossing_position=[1.525,0,2],net_clearance=.3,
            contact_feasible=True,legal_return_predicted=True,attack5_predicted=bool(np.linalg.norm(out)>=5 and out[0]>0 and out[2]<=-1),target_landing_predicted=True)
    strip=dict(center=[0,0,0],size=[.001,.1,.1])
    r=optimize_task_hit(fk,q,qd,-np.ones(6),np.ones(6),np.ones(6)*5,np.zeros(3),np.zeros(3),strip,
        lambda p:np.zeros(3),np.eye(3),np.zeros(3),dict(planning_to_world_rotation=np.eye(3)),impact,court,incoming)
    assert r['accepted'],r['solver_message']
    c=r['selected'];assert c['q_rad'][3]>.05 and np.linalg.norm(c['qd_rad_s'])>0
    assert c['position_error_m']<=.0005 and c['normal_error']<=.12
    assert np.max(abs(c['qd_rad_s']))<=5 and not r['physical_feasibility_verified']
    np.testing.assert_array_equal(q,np.zeros(6));np.testing.assert_array_equal(qd,np.zeros(6))


def test_mutually_exclusive_rate_and_joint_modes():
    from quadsmash.control.contact_first_trajectory_search_v1 import ContactFirstSearchConfig
    with pytest.raises(ValueError):
        ContactFirstSearchConfig(position_anchored_search=True,impact_feasible_rate_retarget=True,task_constrained_hit=True).validate()


def test_impossible_height_stops_before_pose_or_rate_search():
    from types import SimpleNamespace
    from quadsmash.control.position_anchored_contact_search_v1 import plan_anchored_events
    planner=SimpleNamespace(config=SimpleNamespace(contact_time_offsets_s=(0.,),dt_s=.01,horizon_steps=10,task_constrained_hit=True))
    m=model();meta={.05:dict(attack_geometry=attack_geometry([.7,0,1.],m.physics,m.court))}
    def forbidden(*args):raise AssertionError('No optimizer or physical evaluator allowed')
    with pytest.raises(RuntimeError,match='DOWNWARD_ATTACK_HEIGHT_INFEASIBLE'):
        plan_anchored_events(planner,np.zeros(6),np.zeros(6),-np.ones(6),np.ones(6),np.ones(6),.05,
            lambda t:(np.zeros(3),np.array([1.,0,0]),np.zeros(3)),
            lambda t:(np.eye(3),np.zeros(3),np.zeros(3),np.zeros(3)),meta,forbidden,forbidden,forbidden,forbidden)
    assert planner.last_diagnostics['anchors'][0]['position_solver_executed'] is False


@pytest.mark.parametrize('final_speed,accepted',[(6.,True),(4.77,False)])
def test_joint_path_final_forward_margins_gate_installation(monkeypatch,final_speed,accepted):
    from types import SimpleNamespace
    from quadsmash.control.position_anchored_contact_search_v1 import plan_anchored_events
    import quadsmash.control.task_constrained_hit_search_v2 as module
    cfg=SimpleNamespace(contact_time_offsets_s=(0.,),dt_s=.01,horizon_steps=10,task_constrained_hit=True,
        normal_tolerance=.12,velocity_tolerance_m_s=.75,velocity_aware_pose=False,impact_feasible_rate_retarget=False)
    planner=SimpleNamespace(config=cfg)
    def fk(q,*args):return q[:3],np.array([1.,0,0]),np.c_[np.eye(3),np.zeros((3,3))],np.zeros((3,6))
    def joint(*args,**kwargs):
        return dict(accepted=True,selected=dict(q_rad=np.zeros(6),qd_rad_s=np.zeros(6),position=np.zeros(3),
            normal=np.array([1.,0,0]),velocity=np.zeros(3),position_error_m=0.,normal_error=0.,
            task_target_normal_current=np.array([1.,0,0]),contact_point_local=np.zeros(3)))
    monkeypatch.setattr(module,'optimize_task_hit',joint)
    meta={.05:dict(attack_geometry=dict(downward_attack_geometrically_possible=True),analytic_goal_valid=True,
        contact_point_local=np.zeros(3),task_string_strip={},task_court=load_court(),incoming_velocity_planning=np.zeros(3))}
    final_calls=[]
    def physical(q,qd,index,metadata):
        final_calls.append(metadata)
        return dict(accepted=True,actual_racket_position_world_m=np.zeros(3),actual_racket_normal_world=np.array([1.,0,0]),actual_racket_velocity_world_m_s=np.zeros(3))
    def impact(metadata):
        speed=final_speed if final_calls else 6.
        # Flags alone cannot override measured final forward-task margins.
        return dict(outgoing_velocity=[(speed**2-1.2**2)**.5,0,-1.2],ranking_cost=0.,landing_position=[3,0,.025],crossing_position=[1.525,0,2],net_clearance=.3,
            contact_feasible=True,legal_return_predicted=True,attack5_predicted=True,target_landing_predicted=True)
    args=(planner,np.zeros(6),np.zeros(6),-np.ones(6),np.ones(6),np.ones(6)*5,.05,
        lambda t:(np.zeros(3),np.array([1.,0,0]),np.zeros(3)),lambda t:(np.eye(3),np.zeros(3),np.zeros(3),np.zeros(3)),meta,fk,fk,physical,impact)
    if accepted:
        candidate=plan_anchored_events(*args)
        assert candidate.score<0 and final_calls[0]['task_feasible_velocity_retarget']
    else:
        with pytest.raises(RuntimeError,match='final_task_margins'):plan_anchored_events(*args)
