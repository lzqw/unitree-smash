import numpy as np
import pytest
from scripts.native_reference_tracking_v2 import load_tracking_reference


def test_reference_is_not_silently_replaced_by_new_prediction(tmp_path):
    q=np.zeros((520,25));q[:,6]=1.;q[:,0]=.25
    v=np.zeros((520,24))
    np.savez(tmp_path/'states.npz',feedback_q=q,feedback_v=v,predicted_q=q+1,predicted_v=v+2)
    manifest=dict(bundle_name='states.npz',reference_tracking_definition=dict(
        state_reference_contract='explicit_frozen_tracking_reference_v1',q_key='feedback_q',v_key='feedback_v'))
    actual_q,actual_v=load_tracking_reference(tmp_path,manifest,520)
    np.testing.assert_array_equal(actual_q,q);np.testing.assert_array_equal(actual_v,v)
    manifest['reference_tracking_definition'].pop('state_reference_contract')
    with pytest.raises(ValueError):load_tracking_reference(tmp_path,manifest,520)


def test_missing_or_nonfinite_reference_is_rejected(tmp_path):
    manifest=dict(bundle_name='states.npz',reference_tracking_definition=dict(
        state_reference_contract='explicit_frozen_tracking_reference_v1',q_key='q',v_key='v'))
    np.savez(tmp_path/'states.npz',predicted_q=np.zeros((521,25)))
    with pytest.raises(KeyError):load_tracking_reference(tmp_path,manifest,520)
    q=np.zeros((520,25));q[1,0]=np.nan
    np.savez(tmp_path/'states.npz',q=q,v=np.zeros((520,24)))
    with pytest.raises(ValueError):load_tracking_reference(tmp_path,manifest,520)
