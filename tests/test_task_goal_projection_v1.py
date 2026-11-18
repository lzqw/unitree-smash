import json
from pathlib import Path
import numpy as np
import pytest
from quadsmash.control.impact_outcome_evaluator_v1 import ImpactOutcomeEvaluator
from quadsmash.control.task_constrained_impact_goal_v2 import optimize_task_goal, task_margins
from quadsmash.control.task_goal_projection_v1 import project_task_goal
from quadsmash.physics.court_v4 import load_court
from quadsmash.physics.full_racket_v4 import load_racket


@pytest.fixture(scope='module')
def context():
    model = ImpactOutcomeEvaluator(json.loads(Path('configs/shuttle_physics_v2.json').read_text()), load_court(), load_racket())
    p = np.array([.7, 0., 2.]); incoming = np.array([-8., 0., -3.]); local = np.array([.0006, 0., .07])
    goal = optimize_task_goal(model, p, incoming, local, max_iterations=8)
    assert goal['accepted']
    return model, p, incoming, local, np.array(goal['normal_world']), np.array(goal['racket_velocity_world_m_s'])


def test_matching_goal_cannot_regress_or_become_robot_certificate(context):
    m,p,b,l,n,v = context
    r = project_task_goal(m,p,b,l,n,v,n,v,max_iterations=2)
    np.testing.assert_allclose(r['selected']['normal_world'],n,atol=1e-12)
    np.testing.assert_allclose(r['selected']['racket_velocity_world_m_s'],v,atol=1e-12)
    assert not r['improved'] and not r['valid_teacher_reference'] and not r['actual_commands_changed']


def test_projection_rechecks_forward_and_both_tracking_errors(context):
    m,p,b,l,n,v = context
    ref_n = n + np.array([0.,.04,0.]); ref_n /= np.linalg.norm(ref_n)
    before = [x.copy() for x in (p,b,l,n,v)]
    r = project_task_goal(m,p,b,l,n,v,ref_n,v+np.array([-.3,.1,.1]),max_iterations=8)
    s = r['selected']
    assert np.all(s['reference_errors'] <= r['original']['reference_errors']+1e-10)
    out = m.evaluate(p,b,p-m.physics['collision_radius_m']*s['normal_world'],s['racket_velocity_world_m_s'],s['normal_world'],l)
    assert task_margins(out,m.court)['valid']
    for old,new in zip(before,(p,b,l,n,v)): np.testing.assert_array_equal(old,new)
    assert r['conditional_contact_only'] and not r['robot_feasibility_verified']


def test_invalid_forward_seed_rejected(context):
    m,p,b,l,n,v = context
    with pytest.raises(ValueError,match='Seed fails actual forward'):
        project_task_goal(m,p,b,l,n,b,n,v,max_iterations=2)


@pytest.mark.parametrize('bad', [np.array([np.nan,0.,0.]), np.zeros(3)])
def test_bad_normal_rejected(context,bad):
    m,p,b,l,n,v = context
    with pytest.raises(ValueError): project_task_goal(m,p,b,l,n,v,bad,v)
