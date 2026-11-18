import copy
import numpy as np
import pytest
from scripts.coupled_contact_prefix_v1 import selected_arm_reference


def fixture():
    original=np.zeros((4,6));commands=np.zeros((3,18));baseline=[];chosen=[]
    for k in range(3):
        baseline.append(dict(final_target_rad=commands[k].copy(),q=np.r_[np.zeros(12),np.full(6,.01)],arm_tracking_error_rad=.01))
        command=commands[k].copy();command[12:]=.1*(k+1)
        chosen.append(dict(final_target_rad=command,q=np.r_[np.zeros(12),command[12:]+.01],arm_tracking_error_rad=.01))
    source=dict(arm_sequence_rad=original,metadata=dict(contact_index=3),result=dict(replayed_target_sequence_rad=commands))
    record=dict(selected_index=1,trials=[dict(result=dict(path_valid=True,trajectory=baseline)),dict(result=dict(path_valid=True,trajectory=chosen))])
    return record,source


def test_legacy_reference_recovered_and_cross_checked_against_realized_tracking():
    record,source=fixture();ref,audit=selected_arm_reference(record,source,np.arange(12,18))
    np.testing.assert_allclose(ref[1:,0],[.1,.2,.3])
    assert audit['mode']=='legacy_additive_offset_verified' and audit['max_tracking_reconstruction_error_rad']<1e-15
    ref[:]=99
    np.testing.assert_array_equal(source['arm_sequence_rad'],np.zeros((4,6)))


def test_non_original_missing_reference_is_rejected_not_reset_to_nominal():
    record,source=fixture()
    record['trials'][0]['result']['trajectory'][0]['final_target_rad'][12:]=.2
    with pytest.raises(ValueError,match='Missing non-original'):selected_arm_reference(record,source,np.arange(12,18))


def test_explicit_reference_supports_relinearization_and_detects_corruption():
    record,source=fixture();ref,_=selected_arm_reference(record,source,np.arange(12,18))
    record['selected_arm_reference_rad']=ref.copy()
    record['trials'][0]['result']['trajectory'][0]['final_target_rad'][12:]=.2
    restored,audit=selected_arm_reference(record,source,np.arange(12,18))
    np.testing.assert_array_equal(restored,ref);assert audit['mode']=='explicit'
    record['selected_arm_reference_rad'][-1]+=.04
    with pytest.raises(ValueError,match='tracking error'):selected_arm_reference(record,source,np.arange(12,18))


def test_partial_or_nonfinite_reference_is_never_accepted():
    record,source=fixture();record['selected_arm_reference_rad']=np.full((4,6),np.nan)
    with pytest.raises(ValueError):selected_arm_reference(record,source,np.arange(12,18))
    record,source=fixture();record['trials'][1]['result']['path_valid']=False
    with pytest.raises(ValueError):selected_arm_reference(record,source,np.arange(12,18))


def test_equal_tracking_norm_does_not_hide_reference_history_corruption():
    record,source=fixture();ref,_=selected_arm_reference(record,source,np.arange(12,18))
    record['selected_arm_reference_rad']=ref.copy()
    for k,row in enumerate(record['trials'][1]['result']['trajectory']):
        row['planned_arm_reference_rad']=ref[k+1].copy()
    # Reflect one reference component across actual q: same maximum error,
    # different command history. The per-node evidence must still reject it.
    record['selected_arm_reference_rad'][1,0]+=.02
    with pytest.raises(ValueError,match='per-node history'):
        selected_arm_reference(record,source,np.arange(12,18))
