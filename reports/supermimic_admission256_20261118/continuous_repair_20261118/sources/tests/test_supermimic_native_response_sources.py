from copy import deepcopy
import hashlib
import pytest
import torch
from scripts.fit_supermimic_native_response import compatible_response_sources,measured_foot_labels,foot_prediction_metrics


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


def contact_fixture(tmp_path,feet):
    hashes={}
    for branch in ('minus','plus'):
        folder=tmp_path/branch;folder.mkdir()
        path=folder/'native_probe.pt';torch.save({'raw':{'next_feet':feet}},path)
        hashes[branch]=hashlib.sha256(path.read_bytes()).hexdigest()
    return {'source_sha256':hashes}


def test_observed_foot_labels_preserve_all_lanes_and_branches(tmp_path):
    feet=torch.zeros(128,4,dtype=torch.bool);feet[127,2]=True
    result=measured_foot_labels(tmp_path,contact_fixture(tmp_path,feet))
    assert torch.equal(result['next_feet_minus'],feet.float())
    assert torch.equal(result['next_feet_plus'],feet.float())


@pytest.mark.parametrize('feet',[torch.zeros(127,4),torch.full((128,4),.5),torch.full((128,4),float('nan'))])
def test_incomplete_or_soft_foot_predictions_cannot_be_native_labels(tmp_path,feet):
    provenance=contact_fixture(tmp_path,feet)
    with pytest.raises(ValueError,match='binary labels'):measured_foot_labels(tmp_path,provenance)


def test_changed_contact_file_does_not_reuse_original_provenance(tmp_path):
    provenance=contact_fixture(tmp_path,torch.zeros(128,4,dtype=torch.bool))
    (tmp_path/'minus/native_probe.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash changed'):measured_foot_labels(tmp_path,provenance)


def test_foot_diagnostic_counts_false_positive_and_negative_separately():
    result=foot_prediction_metrics(torch.tensor([[.9,.8,.2,.1]]),torch.tensor([[1,0,1,0]]))
    assert result['false_positive']==1 and result['false_negative']==1
    assert result['positives']==2 and result['boundary_labels']==4
    assert result['brier']==pytest.approx(.325)


def test_missing_foot_labels_are_not_zero_contact_truth():
    with pytest.raises(ValueError,match='dimensions'):
        foot_prediction_metrics(torch.ones(2,4),torch.zeros(1,4))


def test_nonfinite_contact_predictions_cannot_make_valid_diagnostics():
    with pytest.raises(ValueError,match='Invalid'):
        foot_prediction_metrics(torch.full((1,4),float('nan')),torch.zeros(1,4))
