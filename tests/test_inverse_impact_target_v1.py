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
