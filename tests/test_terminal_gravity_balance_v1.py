"""A gravity-preloaded target must actually preserve a resting model state."""
import numpy as np
from test_differentiated_sequence_v1 import setup_profile
from scripts.audit_terminal_gravity_balance_v1 import gravity_target
from scripts.implicit_support_prediction_v1 import predict_implicit_support


def test_static_target_preserves_rest_in_original_implicit_model():
    c,state,_=setup_profile();state.v[:]=0.;state.joint_velocity[:]=0.
    before_q=state.q.copy();before_command=state.previous_command_target.copy()
    result=gravity_target(c,state)
    assert result['solved']
    target=np.asarray(result['target'])
    for _ in range(10):
        prediction=predict_implicit_support(c.qp.model,c.qp.interface,state,target,
            substeps=16,contact_velocity_decay_s=c.qp.settings.contact_velocity_decay_s)
        state=prediction['state']
        assert np.max(abs(state.v))<1e-7
    np.testing.assert_allclose(state.q,before_q,atol=1e-8,rtol=0)
    assert np.max(abs(target[12:]-before_command[12:]))<=.100001


def test_no_contact_cannot_supply_static_ground_force():
    c,state,_=setup_profile();state.contacts[:]=False
    result=gravity_target(c,state)
    assert not result['solved'] and 'target' not in result
