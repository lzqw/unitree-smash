import json
from pathlib import Path
import numpy as np
from quadsmash.control.inverse_impact_target_v1 import inverse_impact_target
from quadsmash.control.impact_outcome_evaluator_v1 import ImpactOutcomeEvaluator
from quadsmash.physics.full_racket_v4 import load_racket
from quadsmash.physics.court_v4 import load_court


def test_recorded_lane0_has_analytic_attack_target_not_robot_certificate():
    root=Path(__file__).resolve().parents[1]
    record=json.loads((root/'reports/joint_loop_round_20261117/IMPACT_REJECTION_REPLAY.json').read_text())['impact_calls'][0]
    model=ImpactOutcomeEvaluator(json.loads((root/'configs/shuttle_physics_v2.json').read_text()),load_court(),load_racket())
    from quadsmash.control.initial_intercept import propagate_ball_state
    plan=json.loads((root/'reports/joint_loop_round_20261117/attempts/probe01/native/causal_plans.json').read_text())[0]
    ball,incoming=propagate_ball_state(plan['initial_position_m'][0],plan['initial_velocity_mps'][0],.77,model.physics)
    result=inverse_impact_target(model,ball,incoming,record['local'])
    assert result['accepted']
    out=result['forward_outcome']
    assert out['attack5_predicted'] and out['target_landing_predicted'] and out['legal_return_predicted']
    np.testing.assert_allclose(out['outgoing_velocity'],result['desired_outgoing_velocity_world_m_s'],atol=1e-9)
    assert not result['robot_feasibility_verified']
    # Below the net cannot be accepted as a downward attack.
    position=np.array(record['ball']);position[2]=1.
    assert not inverse_impact_target(model,position,record['incoming_velocity'],record['local'])['accepted']


def test_continuous_target_keeps_actual_string_box_and_separate_gates():
    from quadsmash.control.inverse_impact_target_v1 import optimize_reachable_impact
    root=Path(__file__).resolve().parents[1]
    model=ImpactOutcomeEvaluator(json.loads((root/'configs/shuttle_physics_v2.json').read_text()),load_court(),load_racket())
    p=np.array([.7,0.,2.]);v=np.array([-8.,0.,-3.]);local=np.array([.0006,0.,.07])
    seed=inverse_impact_target(model,p,v,local)
    normal=np.array(seed['normal_world'])
    def kin(q,contact):
        return p+q[:3],normal,np.c_[np.eye(3),np.zeros((3,3))],np.zeros((3,6))
    result=optimize_reachable_impact(model,p,v,local,kin,np.zeros(6),np.full(6,-1.),np.ones(6),np.ones(6)*5,max_nfev=2)
    selected=np.array(result['contact_point_local'])
    assert any(np.all(np.abs(selected-np.array(s['center']))<=.5*np.array(s['size'])+1e-9) for s in model.strips)
    assert np.all(np.abs(result['optimized_q_rad'])<=1.)
    assert not result['robot_feasibility_verified']
    if result['accepted']:
        assert result['forward_outcome']['legal_return_predicted']
        assert result['forward_outcome']['attack5_predicted']
        assert result['forward_outcome']['target_landing_predicted']
