import copy,importlib.util
from pathlib import Path
import pytest
import quadsmash.rl.supermimic_evidence_binding_v1 as m

def identity():
    return dict(execution_code_sha256={**{k:'same' for k in m.NATIVE_REQUIRED_CODE},'supermimic_action_ball_v1.py':'old'},schema='runtime-v1',actor_state_sha256='actor',world_state_sha256='old-world',task_contract_sha256='contract',policy_execution='causal',policy_version=0)

def test_model_edit_alone_never_passes_without_measured_current_region():
    old=identity();new=copy.deepcopy(old)
    new['execution_code_sha256']['supermimic_action_ball_v1.py']='new';new['world_state_sha256']='new-world'
    assert m.native_code_compatibility(old['execution_code_sha256'],new['execution_code_sha256'])['compatible']
    probe=dict(runtime_identity=old,validity_region={'sha256':'bundle'})
    assert not m.probe_applicability(probe,new)['current_version_valid']
    region=dict(passed=True,current_identity=new,origin_identity=old,evidence_sha256='bundle')
    assert m.probe_applicability(probe,new,region)['current_version_valid']
    assert not m.probe_applicability(probe,new,dict(region,passed=False))['current_version_valid']
    assert not m.probe_applicability(probe,new,dict(region,evidence_sha256='other'))['current_version_valid']

@pytest.mark.parametrize('filename',sorted(m.NATIVE_REQUIRED_CODE)+['supermimic_g07_memory_preview_v1.py','unknown_future_native.py'])
def test_native_and_unknown_edits_still_rejected(filename):
    old=identity();new=copy.deepcopy(old);new['execution_code_sha256'][filename]='changed'
    assert not m.native_code_compatibility(old['execution_code_sha256'],new['execution_code_sha256'])['compatible']
    region=dict(passed=True,current_identity=new,origin_identity=old,evidence_sha256='bundle')
    assert not m.probe_applicability(dict(runtime_identity=old,validity_region={'sha256':'bundle'}),new,region)['current_version_valid']

def test_incomplete_native_identity_cannot_be_migrated():
    old=identity()['execution_code_sha256'];new=copy.deepcopy(old)
    del old['supermimic_policy_execution_v1.py'];new['supermimic_action_ball_v1.py']='new'
    assert not m.native_code_compatibility(old,new)['compatible']

def test_original_hash_maps_are_immutable_and_differences_explicit():
    a=identity()['execution_code_sha256'];before=copy.deepcopy(a);b=copy.deepcopy(a)
    b['supermimic_articulated_inertia_v1.py']='new-file'
    result=m.native_code_compatibility(a,b)
    assert a==before and result['differences']['supermimic_articulated_inertia_v1.py']==dict(recorded=None,current='new-file')
    assert result['compatible']
