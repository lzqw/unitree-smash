import numpy as np
import pytest
from quadsmash.control.support_risk_features_v1 import support_risk_features


def test_future_task_identity_and_clock_fields_cannot_change_features():
    rng=np.random.default_rng(9)
    raw=rng.normal(size=(3,794));action=rng.normal(size=(3,18))
    expected=support_risk_features(raw,action)
    changed=raw.copy()
    used=np.zeros(794,bool);used[:61]=True;used[690:751]=True;used[776:794]=True
    changed[:,~used]=rng.normal(100,30,size=(3,(~used).sum()))
    np.testing.assert_array_equal(support_risk_features(changed,action),expected)
    assert expected.shape==(3,158)


def test_current_command_and_executed_history_are_distinct_features():
    raw=np.zeros((1,794));raw[:,776:794]=.4
    action=np.full((1,18),.2)
    value=support_risk_features(raw,action)
    np.testing.assert_array_equal(value[:,122:140],action)
    np.testing.assert_array_equal(value[:,140:158],raw[:,776:794])
    raw[:,690:751]=.3
    np.testing.assert_allclose(support_risk_features(raw,action)[:,61:122],-3.)


def test_invalid_input_rejected():
    with pytest.raises(ValueError):support_risk_features(np.zeros((2,793)),np.zeros((2,18)))
