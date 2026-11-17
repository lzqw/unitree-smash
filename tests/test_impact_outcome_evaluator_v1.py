import json
from pathlib import Path
import numpy as np
import pytest
from quadsmash.control.impact_outcome_evaluator_v1 import ImpactOutcomeEvaluator
from quadsmash.physics.full_racket_v4 import load_racket
from quadsmash.physics.court_v4 import load_court


def fixture():
    physics=json.loads((Path(__file__).resolve().parents[1]/'configs/shuttle_physics_v2.json').read_text())
    model=ImpactOutcomeEvaluator(physics,load_court(),load_racket())
    surface=np.array([.5,0.,3.])
    args=dict(shuttle_position=surface+np.array([.025,0.,0.]),
        shuttle_velocity_before=np.array([-4.,0.,0.]),racket_contact_position=surface,
        racket_contact_velocity=np.zeros(3),racket_normal=np.array([1.,0.,0.]),
        contact_point_local=np.array([model.racket['assumptions']['string_thickness_m']/2,0.,0.]))
    return model,args


def test_return_uses_incoming_velocity_not_racket_velocity_and_locked_drag():
    model,args=fixture(); result=model.evaluate(**args)
    assert result['valid_string_contact'] and result['valid_return']
    assert result['net_clearance']>0 and result['predicted_target_landing']
    np.testing.assert_allclose(result['outgoing_velocity'],[2.88,0.,0.])
    assert not result['native_execution']
    baseline=model.evaluate(**args)
    model.physics['quadratic_drag_k_per_m']=0.
    assert model.evaluate(**args)['landing_position'][0]>baseline['landing_position'][0]


@pytest.mark.parametrize('change,reason',[
    ({'shuttle_velocity_before':np.array([4.,0.,0.])},'nonapproaching_contact'),
    ({'contact_point_local':np.array([.0006,.5,0.])},'outside_authored_string_surface'),
    ({'shuttle_position':np.array([1.,0.,3.])},'ball_string_contact_position_mismatch'),
])
def test_rejects_separation_frame_and_missed_contact(change,reason):
    model,args=fixture();args.update(change)
    assert model.evaluate(**args)['rejection_reason']==reason


def test_finite_mass_reaction_conserves_impulse_with_tangential_friction():
    model,args=fixture();args['shuttle_velocity_before']=np.array([-4.,1.,0.])
    args['robot_state']=dict(mass_matrix=np.eye(3)*.2,contact_jacobian_world=np.eye(3))
    result=model.evaluate(**args)
    delta=np.asarray(result['outgoing_velocity'])-args['shuttle_velocity_before']
    np.testing.assert_allclose(model.physics['mass_kg']*delta,result['predicted_impulse'],atol=1e-12)
    np.testing.assert_allclose(np.asarray(result['reaction_impulse'])+result['predicted_impulse'],0.,atol=1e-12)
    assert result['effective_racket_mass_kg']==pytest.approx(.2)
    assert result['outgoing_velocity'][1]<1.


def test_net_collision_is_not_ranked_as_a_valid_return():
    model,args=fixture()
    args['shuttle_position'][2]=1.;args['racket_contact_position'][2]=1.
    result=model.evaluate(**args)
    assert not result['valid_return']
    assert result['rejection_reason'] in ('predicted_net_intersection','no_net_crossing')


def test_slow_legal_return_is_not_attack5_or_complete5():
    model,args=fixture();result=model.evaluate(**args)
    assert result['legal_return_predicted']
    assert not result['attack5_predicted']
    assert not result['ball_side_complete_predicted']
    assert 'complete5' not in result
