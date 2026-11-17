import copy,json
from pathlib import Path
from types import SimpleNamespace
import pytest,torch
from torch import nn
from quadsmash.rl.supermimic_approach_data_v1 import gate_on_actor_device
from quadsmash.rl.supermimic_evidence_binding_v1 import probe_applicability
from quadsmash.rl.supermimic_probe_region_v1 import paired_metrics,load_region
from quadsmash.rl.supermimic_native_pair_capture_v1 import validate_spec,tree_digest

class Gate(nn.Module):
    def __init__(self):super().__init__();self.weight=nn.Parameter(torch.ones(1,dtype=torch.float64))
    def approach_gate(self,obs):
        assert obs.device==self.weight.device and obs.dtype==self.weight.dtype
        return obs[:,0]*self.weight

def test_gate_input_obeys_actor_dtype_and_preserves_archive():
    x=torch.ones(2,794,dtype=torch.float32);before=x.clone()
    result=gate_on_actor_device(Gate(),{'policy':x})
    assert result.dtype==torch.float64 and not result.requires_grad and torch.equal(x,before)
    with pytest.raises(ValueError):gate_on_actor_device(Gate(),x[:,:793])

class World(nn.Module):
    def __init__(self):super().__init__();self.scale=nn.Parameter(torch.tensor(1.));self.geometry=None
    def prior(self,s,*args):return s,None
    def forward(self,s,target,*args):
        result=s.clone();result[:,10:13]=target[:,:3]*self.scale
        return result,s.new_zeros(len(s),4),args[2]
class Ball(nn.Module):
    def forward(self,features,present):return features.new_zeros(len(features),6),None

def pair_fixture(monkeypatch):
    # Synthetic response only verifies checks, not real PhysX accuracy.
    import quadsmash.rl.supermimic_probe_region_v1 as m
    def transition(base,ball,state,target,*args,**kwargs):
        out=state.clone();out[:,10:13]=target[:,:3]*base.scale;out[:,52:55]=target[:,:3]*base.scale
        return out,torch.zeros(len(state),4),torch.ones(len(state))
    monkeypatch.setattr(m,'ball_transition',transition)
    adapter=nn.Module();adapter.physical_model=World();adapter.ball_model=Ball();adapter.action_ball_model=None
    minus=torch.zeros(2,18);plus=minus.clone();plus[:,0]=.02
    nominal=torch.zeros(2,55);native=nominal.clone();native[:,10]=.02;native[:,52]=.02
    p=dict(state=nominal,targets_minus=minus,targets_plus=plus,previous_action=minus,feet=torch.zeros(2,4),present=torch.ones(2),elapsed=torch.zeros(2),launch=torch.zeros(2,6),native_minus=nominal,native_plus=native,native_nominal_a=nominal,native_nominal_b=nominal.clone(),
      restore_mode='from-ground-identical-prefix',prefix_sha256={k:'prefix' for k in ('nominal_a','nominal_b','minus','plus')},executor_before_sha256={k:'memory' for k in ('nominal_a','nominal_b','minus','plus')})
    limits=dict(native_response_min=.001,native_repeat_max=1e-6,response_rmse_max=.001,response_cosine_min=.9,response_relative_error_max=.1)
    return adapter,p,limits

@pytest.mark.parametrize('head',['recovery','approach'])
def test_current_model_rechecked_after_update_and_reversed_gradient_rejected(monkeypatch,head):
    adapter,p,limits=pair_fixture(monkeypatch)
    assert paired_metrics(adapter,p,head,limits)['passed']
    with torch.no_grad():adapter.physical_model.scale.fill_(-1)
    r=paired_metrics(adapter,p,head,limits)
    assert not r['passed'] and 'local-jacobian-direction' in r['failures']

def test_no_saturated_response_or_snapshot_restore_pass(monkeypatch):
    adapter,p,limits=pair_fixture(monkeypatch);p['targets_plus']=p['targets_minus'].clone()
    assert not paired_metrics(adapter,p,'approach',limits)['passed']
    p['executor_before_sha256']['plus']='wrong'
    with pytest.raises(ValueError,match='prefix'):paired_metrics(adapter,p,'approach',limits)

def test_bounded_evidence_does_not_remove_static_contract_checks():
    origin=dict(execution_code_sha256={'x':'a'},schema='id',actor_state_sha256='a',world_state_sha256='w',task_contract_sha256='t',policy_execution='e',policy_version=0)
    current=dict(origin,world_state_sha256='updated')
    probe=dict(runtime_identity=origin,validity_region={'sha256':'raw'})
    assert not probe_applicability(probe,current)['current_version_valid']
    assessed=dict(passed=True,current_identity=current,origin_identity=origin,evidence_sha256='raw')
    assert probe_applicability(probe,current,assessed)['current_version_valid']
    changed=dict(current,task_contract_sha256='different')
    assert not probe_applicability(probe,changed,dict(assessed,current_identity=changed))['current_version_valid']
    assert not probe_applicability(probe,current,dict(assessed,passed=False))['current_version_valid']
    exact_failure=probe_applicability(probe,origin,dict(assessed,passed=False,current_identity=origin))
    assert not exact_failure['current_version_valid'] and exact_failure['revalidation_required']

def test_probe_spec_rejects_bad_lane_or_zero_request():
    spec=dict(schema='native-paired-prefix-spec-v1',tick=270,lane=17,head='approach',epsilon_request=.001,request_direction=[1.]+[0.]*17)
    assert validate_spec(spec)==spec
    for bad in (dict(spec,lane=128),dict(spec,request_direction=[0.]*18),dict(spec,tick=800)):
        with pytest.raises(ValueError):validate_spec(bad)
    assert tree_digest({'x':torch.tensor(float('nan'))})==tree_digest({'x':torch.tensor(float('nan'))})


def test_region_requires_hashed_calibration_and_bound_actor_drift(tmp_path,monkeypatch):
    import hashlib
    from quadsmash.rl.supermimic_probe_region_v1 import assess_region
    from quadsmash.rl.supermimic_physical_loop_loading_v1 import tensor_digest
    adapter,pairs,limits=pair_fixture(monkeypatch)
    actor=Gate();origin=dict(execution_code_sha256={'source':'same'},schema='runtime',actor_state_sha256=tensor_digest(actor.state_dict().items()),
        world_state_sha256='old',task_contract_sha256='task',policy_execution={'gate':'causal'},policy_version=0)
    limits.update(final_normalized_command_delta_max=.01,final_joint_target_delta_rad_max=.04,accepted_update_age_max=8)
    bundle=dict(sample_ids=['holdout'],schema='native-paired-prefix-region-v1',runtime_identity=origin,actor_state=actor.state_dict(),pairs=pairs,head='approach',
        command_starts=[dict(start={'current':'complete-start-test-double'},final_normalized=torch.zeros(2,18),final_joint_targets=torch.zeros(2,18))])
    raw=tmp_path/'probe.pt';torch.save(bundle,raw);sha=hashlib.sha256(raw.read_bytes()).hexdigest()
    calibration=tmp_path/'calibration.json';calibration.write_text(json.dumps(dict(limits=limits,probe_sha256=sha,heldout_episode_ids=['holdout'],calibration_episode_ids=['calibrate'])))
    spec=dict(path=raw.name,sha256=sha,limits=limits,minimum_samples=2,
        calibration=dict(path=calibration.name,sha256=hashlib.sha256(calibration.read_bytes()).hexdigest()))
    current=dict(origin,world_state_sha256='new',policy_version=1)
    def preview(row):return torch.zeros(2,18),torch.zeros(2,18)
    report=assess_region(spec,tmp_path/'receipt.json',current,adapter,preview,'approach')
    assert report['passed'] and report['accepted_update_age']==1
    def drift(row):return torch.ones(2,18),torch.ones(2,18)
    report=assess_region(spec,tmp_path/'receipt.json',current,adapter,drift,'approach')
    assert not report['passed'] and 'executed-joint-target-drift' in report['failures']
    assert not assess_region(spec,tmp_path/'receipt.json',dict(current,policy_version=9),adapter,preview,'approach')['passed']
    with pytest.raises(ValueError,match='different head'):assess_region(spec,tmp_path/'receipt.json',current,adapter,preview,'recovery')
    calibration.write_text('{}')
    with pytest.raises(ValueError,match='hash'):load_region(spec,tmp_path/'receipt.json')
