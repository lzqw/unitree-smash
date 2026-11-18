import copy,importlib.util
from pathlib import Path
import pytest
s=importlib.util.spec_from_file_location('quadsmash.rl._staged_evidence',Path(__file__).with_name('candidate.py'))
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def ident():
 return dict(schema='runtime-v1',execution_code_sha256={**{k:'same' for k in m.NATIVE_REQUIRED_CODE},'supermimic_probe_region_v1.py':'old'},actor_state_sha256='actor',world_state_sha256='world',task_contract_sha256='task',policy_execution='causal',policy_version=0)
def test_validator_change_requires_recomputed_bound_result():
 old=ident();new=copy.deepcopy(old);new['execution_code_sha256']['supermimic_probe_region_v1.py']='new'
 probe=dict(runtime_identity=old,validity_region={'sha256':'raw'})
 result=m.native_code_compatibility(old['execution_code_sha256'],new['execution_code_sha256'])
 assert result['compatible'] and result['evidence_only_changes']==['supermimic_probe_region_v1.py']
 assert result['current_model_and_validator_recheck_required']
 assert not m.probe_applicability(probe,new)['current_version_valid']
 good=dict(passed=True,current_identity=new,origin_identity=old,evidence_sha256='raw')
 assert m.probe_applicability(probe,new,good)['current_version_valid']
 for bad in [dict(good,passed=False),dict(good,current_identity=old),dict(good,evidence_sha256='other')]:
  assert not m.probe_applicability(probe,new,bad)['current_version_valid']
@pytest.mark.parametrize('name',['supermimic_native_pair_capture_v1.py','supermimic_native_substep_trace_v1.py','control/g07_shared_target_residual_v1.py','unknown.py'])
def test_measurement_execution_unknown_changes_remain_rejected(name):
 old=ident();new=copy.deepcopy(old);new['execution_code_sha256'][name]='changed'
 assert not m.native_code_compatibility(old['execution_code_sha256'],new['execution_code_sha256'])['compatible']
