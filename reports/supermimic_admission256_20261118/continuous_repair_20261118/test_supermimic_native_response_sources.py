from copy import deepcopy
import pytest
from scripts.fit_supermimic_native_response import compatible_response_sources


def source(tick=272,head='approach'):
    return dict(task_contract={'asset':'same'},policy_execution={'gate':'causal'},
                actor_state_sha256='unchanged',train_lanes=[1,2],validation_lanes=[0],
                split_sha256='fixed',spec=dict(head=head,tick=tick))


def test_cross_time_cross_head_measured_supervision_preserves_split():
    compatible_response_sources(source(),[source(271),source(302,'recovery')])


@pytest.mark.parametrize('key,value',[
    ('task_contract',{'asset':'different'}),('policy_execution',{'gate':'legacy'}),
    ('actor_state_sha256','changed'),('train_lanes',[0,1,2]),('validation_lanes',[2]),
    ('split_sha256','different')])
def test_mixed_identity_or_heldout_fit_is_rejected(key,value):
    other=source(271);other[key]=value
    with pytest.raises(ValueError,match=key):compatible_response_sources(source(),[other])


def test_duplicate_source_cannot_reweight_dataset_silently():
    with pytest.raises(ValueError,match='Duplicate'):compatible_response_sources(source(),[deepcopy(source())])
