import numpy as np
from scripts.audit_prediction_acceptance_timing_v1 import counts


def test_acceptance_after_a_historical_fault_is_not_a_rescue_opportunity():
    records=[dict(accepted=True,applied_first_step=True,control_tick=t,horizon_steps=40)
             for t in [300,320,350,360]]
    result=counts(records,3.5)
    assert result['accepted']==4
    assert result['accepted_before_nonfoot']==2
    assert result['accepted_after_nonfoot']==2
    assert result['accepted_before_event_with_event_in_horizon']==1
    assert result['applied_before_nonfoot']==2


def test_no_nonfoot_event_has_no_fabricated_horizon_overlap():
    result=counts([dict(accepted=True,applied_first_step=True,control_tick=300,horizon_steps=40)],np.inf)
    assert result['first_nonfoot_s'] is None
    assert result['accepted_before_nonfoot']==1
    assert result['accepted_after_nonfoot']==0
    assert result['accepted_before_event_with_event_in_horizon']==0
