"""Telemetry alignment checks; these are not native physics tests."""
import numpy as np
from scripts.compare_isolation_microtrace_v1 import compare_values, STATE_FIELDS, SUBMITTED_FIELDS


def test_submission_difference_before_state_is_not_excluded():
    frames=np.arange(100,105)
    old={k:np.zeros((5,3)) for k in STATE_FIELDS+SUBMITTED_FIELDS}
    new={k:v.copy() for k,v in old.items()}
    new['submitted_effort_target'][1,0]=1
    new['joint_velocity'][2,0]=.1
    result=compare_values(frames,old,new)
    assert result['first_physical_state_difference_frame']==102
    assert not result['submitted_buffers_equal_through_first_state_difference']


def test_later_submission_does_not_precede_first_state_difference():
    frames=np.arange(100,105)
    old={k:np.zeros((5,3)) for k in STATE_FIELDS+SUBMITTED_FIELDS}
    new={k:v.copy() for k,v in old.items()}
    new['submitted_position_target'][3,0]=1
    new['joint_velocity'][2,0]=.1
    result=compare_values(frames,old,new)
    assert result['submitted_buffers_equal_through_first_state_difference']
    assert not result['cause_identified']


def test_identical_window_has_no_first_difference():
    frames=np.arange(100,105)
    old={k:np.zeros((5,3)) for k in STATE_FIELDS+SUBMITTED_FIELDS}
    result=compare_values(frames,old,old)
    assert result['first_physical_state_difference_frame'] is None
    assert result['submitted_buffers_equal_through_first_state_difference'] is None
