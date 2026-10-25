"""Controlled CPU contract/composition fixtures; no native performance."""
import copy
import hashlib
import io
import json
from types import SimpleNamespace

import pytest
import torch

import quadsmash.control.full_racket_residual_v4 as residual
from quadsmash.rl.frozen_baseline_source_v4 import verify_frozen_baseline_source


class Constant(torch.nn.Module):
    def __init__(self,value):
        super().__init__();self.register_buffer('value',torch.tensor(value,dtype=torch.float32))
    def forward(self,public):return self.value.expand(public.shape[0],18)


def blob(value):
    stream=io.BytesIO();torch.jit.save(torch.jit.script(Constant(value)),stream);return stream.getvalue()


def measured_fixture():
    # These declared positive fields exercise validation only, never evidence.
    actor=blob(.1)
    analytic=dict(schema=residual.ANALYTIC_SCHEMA,asset_sha256='CPU-fixture',parameters=torch.zeros(38),
        seed=dict(arm_target_rate_limit_rad_s=5.))
    return dict(schema=residual.SCHEMA,asset_sha256='CPU-fixture',parameters=torch.zeros(38),
        seed=copy.deepcopy(analytic['seed']),base_controller=analytic,native_execution=True,
        residual_actor_jit=actor,residual_joint_scale_rad=[.1]*18,
        residual_training_metadata=dict(actual_control_env_steps=8,actor_jit_sha256=hashlib.sha256(actor).hexdigest()))


class AnalyticFixture:
    def __init__(self,checkpoint,count,device):
        self.count=count;self.logs=[];self.resets=[]
    def action(self,*args):return torch.zeros(self.count,18)
    def reset(self,ids=None):self.resets.append(ids)


def outer(source,value=0.):
    result=copy.deepcopy(source)
    result.update(base_controller=copy.deepcopy(source),residual_actor_jit=blob(value),
        residual_base_mode='frozen_public_policy',residual_joint_scale_rad=[.03]*12+[.15]*6,
        native_execution=False,performance_inherited=False)
    return result


def test_nested_zero_action_preserves_existing_commands_and_reference(monkeypatch):
    monkeypatch.setattr(residual,'FullRacketRallyControllerV4',AnalyticFixture)
    source=measured_fixture();before=copy.deepcopy(source)
    old=residual.FullRacketResidualControllerV4(source,2,'cpu')
    new=residual.FullRacketResidualControllerV4(outer(source),2,'cpu')
    public=torch.zeros(2,794);default=torch.zeros(18);limits=torch.tensor([[-2.,2.]]*18)
    for step in range(20):
        public[:,775]=step*.01
        assert torch.equal(old.action({'policy':public},default,limits),new.action({'policy':public},default,limits))
    assert new.reference_controller is new.base.base
    assert new.logs is new.reference_controller.logs
    target=new.base.mixer.arm_rate.target.clone()
    new.reference_controller.logs=['planning diagnostic fixture']
    new.logs=[]
    assert new.reference_controller.logs==[] and torch.equal(new.base.mixer.arm_rate.target,target)
    new.reset(torch.tensor([1]));assert new.reference_controller.resets[-1].tolist()==[1]
    assert not new.mixer.nonzero_seen.any()
    assert source['residual_actor_jit']==before['residual_actor_jit']
    assert torch.equal(source['parameters'],before['parameters'])


def test_new_residual_keeps_all_joints_available_and_original_slew(monkeypatch):
    monkeypatch.setattr(residual,'FullRacketRallyControllerV4',AnalyticFixture)
    policy=residual.FullRacketResidualControllerV4(outer(measured_fixture(),1.),2,'cpu')
    public=torch.zeros(2,794);default=torch.zeros(18);limits=torch.tensor([[-2.,2.]]*18);last=None
    for step in range(20):
        public[:,775]=step*.01;action=policy.action({'policy':public},default,limits)
        assert action.shape==(2,18) and action.abs().max()<=1.
        if last is not None:assert (2*(action[:,12:]-last[:,12:])).abs().max()<=.050001
        last=action
    assert action[:,:12].ne(0).all() and action[:,12:].ne(0).all()


@pytest.mark.parametrize('change',['unexecuted','blob','asset','slew','deep'])
def test_reject_unmeasured_or_mismatched_base(change):
    source=measured_fixture()
    if change=='unexecuted':source['native_execution']=False
    elif change=='blob':source['residual_actor_jit']+=b'changed'
    elif change=='asset':source['asset_sha256']='different'
    elif change=='slew':source['base_controller']['seed']['arm_target_rate_limit_rad_s']=50.
    else:source['residual_base_mode']='frozen_public_policy'
    with pytest.raises(ValueError):residual.validate_frozen_baseline(source)


def test_implicit_nested_base_is_rejected(monkeypatch):
    monkeypatch.setattr(residual,'FullRacketRallyControllerV4',AnalyticFixture)
    policy=outer(measured_fixture());policy.pop('residual_base_mode')
    with pytest.raises(ValueError,match='explicit'):residual.FullRacketResidualControllerV4(policy,2,'cpu')
    with pytest.raises(ValueError):residual.baseline_mode('unknown')


def test_source_receipts_and_original_protocol_are_required(tmp_path):
    source=measured_fixture();path=tmp_path/'policy.pt';torch.save(source,path)
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    data={'job.json':dict(status='SUCCEEDED',exit_code=0,receipt_present=True,charged_env_steps=8),
        'completed.json':dict(env_steps=8),'INPUTS.json':dict(contract=dict(protocol_sha256='original')),
        'metrics.json':dict(policy_sha256=digest(path))}
    for name,value in data.items():(tmp_path/name).write_text(json.dumps(value))
    result=verify_frozen_baseline_source(path,source,'original')
    assert result['source_completed_control_env_steps']==8
    with pytest.raises(ValueError,match='protocol'):verify_frozen_baseline_source(path,source,'other')
    data['job.json']['status']='RUNNING';(tmp_path/'job.json').write_text(json.dumps(data['job.json']))
    with pytest.raises(ValueError,match='completed'):verify_frozen_baseline_source(path,source,'original')


def test_centered_export_keeps_base_and_starts_without_native_claim():
    from tensordict import TensorDict
    from rsl_rl.modules import ActorCritic
    from quadsmash.rl.full_racket_residual_ppo_v4 import export_residual_policy
    public=torch.zeros(2,794);obs=TensorDict({'policy':public},batch_size=[2])
    policy=ActorCritic(obs,{'policy':['policy'],'critic':['policy']},18,
        actor_hidden_dims=[16],critic_hidden_dims=[16],actor_obs_normalization=True)
    with torch.no_grad():policy.actor[-1].weight.zero_();policy.actor[-1].bias.zero_()
    source=measured_fixture();exported=export_residual_policy(source,policy,[.03]*12+[.15]*6,public,
        dict(native_execution=False,base_controller_mode='frozen_public_policy'))
    assert not exported['native_execution'] and not exported['performance_inherited']
    assert exported['base_controller']['residual_actor_jit']==source['residual_actor_jit']
    actor=torch.jit.load(io.BytesIO(exported['residual_actor_jit']))
    assert actor(public).eq(0).all() and exported['residual_base_mode']=='frozen_public_policy'


def test_error_checkpoint_preserves_completed_update_without_success_receipt(tmp_path):
    from quadsmash.rl.full_racket_residual_ppo_v4 import save_completed_update_error_checkpoint
    calls=[]
    def save(path,infos):
        calls.append(infos);path.write_bytes(b'Controlled CPU checkpoint fixture, not native evidence')
    runner=SimpleNamespace(current_learning_iteration=3,save_restart=save)
    result=save_completed_update_error_checkpoint(runner,tmp_path,{'version':'fixture'},{'mode':'fixture'},
        RuntimeError('Controlled diagnostic failure'),42,True)
    assert len(calls)==1 and result['completed_iteration']==3 and result['actual_control_env_steps']==42
    assert result['canonical_ground_restart'] and not result['job_succeeded']
    assert not (tmp_path/'completed.json').exists()
    assert json.loads((tmp_path/'error_recovery_checkpoint.json').read_text())==result


def test_incomplete_update_or_existing_checkpoint_is_not_relabelled(tmp_path):
    from quadsmash.rl.full_racket_residual_ppo_v4 import save_completed_update_error_checkpoint
    def forbidden(*args,**kwargs):raise AssertionError('Do not save an incomplete update')
    runner=SimpleNamespace(current_learning_iteration=1,save_restart=forbidden)
    assert save_completed_update_error_checkpoint(runner,tmp_path,{}, {},RuntimeError(),1,False)is None
    assert not list(tmp_path.iterdir())
    (tmp_path/'checkpoint.pt').write_bytes(b'Existing completed checkpoint fixture')
    assert save_completed_update_error_checkpoint(runner,tmp_path,{}, {},RuntimeError(),1,True)is None
    assert (tmp_path/'checkpoint.pt').read_bytes()==b'Existing completed checkpoint fixture'
