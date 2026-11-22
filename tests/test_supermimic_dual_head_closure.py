"""CPU contracts only; not native contact/gradient or policy-performance evidence."""
import copy
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from quadsmash.rl.supermimic_policy_execution_v1 import POSTHIT,LEGACY,PHYSICAL_SCHEMA,contract,resolve_contract,policy_forward
from quadsmash.rl.supermimic_dual_head_quality_v1 import quality_spec,reference_coverage,split_native_value_rows,critic_holdout
from quadsmash.rl.supermimic_execution_preview_v1 import ResidualExecutionPreviewV1,MixerMemory
from quadsmash.control.full_racket_residual_v4 import ResidualActionMixerV4
from quadsmash.rl.supermimic_v1 import ResidualPPOInitializedActor


class Export(nn.Module):
    def __init__(self):
        super().__init__();self.normalizer=nn.Identity()
        self.base=nn.Linear(794,18);self.recovery=nn.Linear(794,18);self.approach=nn.Linear(794,18)
        self.recovery_scale=.2;self.approach_scale=.1
        for layer in (self.base,self.recovery,self.approach):nn.init.zeros_(layer.weight);nn.init.constant_(layer.bias,.2)
    def gate(self,obs):return obs[:,771:772]
    def forward(self,obs):
        return self.base(obs)+obs[:,774:775]*self.recovery(obs)*self.recovery_scale+self.gate(obs)*self.approach(obs)*self.approach_scale


@pytest.mark.parametrize('head', ['recovery', 'approach'])
def test_deterministic_rollout_matches_native_requests_and_derivatives(head):
    from quadsmash.rl.supermimic_physical_learner_v1 import PhysicalHorizonTrainer

    actor = ResidualPPOInitializedActor(Export())
    reference = copy.deepcopy(actor)
    with torch.no_grad():
        actor.actor.recovery.bias.add_(.08)
        actor.actor.approach.weight[:, 0].fill_(.15)
    trainer = object.__new__(PhysicalHorizonTrainer)
    trainer.actor, trainer.reference, trainer.block_name = actor, reference, head
    obs = torch.randn(4, 794, requires_grad=True)
    rally = dict(had_contact=torch.tensor([0., 1., 1., 0.]), released=torch.tensor([0., 1., 0., 1.]))
    state = SimpleNamespace(observation=obs, task_memory={'rally': rally})
    expected = trainer._actor_action(state)[0]
    params = (obs, actor.actor.recovery.bias, actor.actor.approach.weight)
    expected_grad = torch.autograd.grad(expected.sum(), params)
    action, anchor, gate = trainer._rollout_requests(state)
    assert torch.equal(action, expected) and not anchor.requires_grad
    assert torch.equal(anchor, trainer._reference_action(state)[0])
    assert torch.equal(gate, trainer._head_gate(state).reshape(-1))
    for actual, old in zip(torch.autograd.grad(action.sum(), params), expected_grad, strict=True):
        assert torch.equal(actual, old)


def test_train_eval_deployment_requests_and_final_targets_share_memory_contract():
    actor=ResidualPPOInitializedActor(Export());mixers=[ResidualActionMixerV4(4,'cpu',[.2]*18) for _ in range(3)]
    preview=ResidualExecutionPreviewV1([.2]*18);memory=MixerMemory.capture(mixers[0])
    obs=torch.zeros(4,794);limits=torch.tensor([[-2.,2.]]*18);default=torch.zeros(18)
    for tick in range(5):
        obs[:,775]=tick*.01;obs[:,774]=1;obs[:,771]=.5
        rally=dict(had_contact=torch.tensor([0,1,1,1]),released=torch.tensor([0,0,1,1]))
        if tick==3:
            for mixer in mixers:mixer.reset(torch.tensor([3]))
            memory=memory.reset(torch.tensor([3]));rally['had_contact'][3]=0;rally['released'][3]=0
        # Separate entry shapes (mapping vs env object), identical current latches.
        outputs=[policy_forward(actor,obs,rally)[0],policy_forward(actor,obs,SimpleNamespace(**rally),mode=resolve_contract({'metadata':{'policy_execution':contract()}})[0]['recovery_gate'])[0],policy_forward(actor,obs,rally,mode=POSTHIT)[0]]
        base=torch.full((4,18),.1)
        pred=preview(base,outputs[0],obs,default,limits,memory)
        for request,mixer in zip(outputs,mixers):
            final=mixer.action(base,request,obs,default,limits)
            targets=limits[:,0]+.5*(final+1)*(limits[:,1]-limits[:,0])
            torch.testing.assert_close(targets,pred.final_joint_targets,atol=2e-6,rtol=0)
        memory=pred.next_memory
    legacy=policy_forward(actor,obs,{},mode=LEGACY)[0]
    assert not torch.equal(legacy,outputs[0]) # missed touchdown cannot enable new Recovery


def test_gate_is_fail_closed_and_does_not_cut_active_head_gradient():
    actor=ResidualPPOInitializedActor(Export());obs=torch.zeros(2,794)
    with pytest.raises(ValueError,match='released'):policy_forward(actor,obs,{'had_contact':torch.ones(2)})
    with pytest.raises(ValueError,match='explicit gate'):policy_forward(nn.Linear(794,18),obs,dict(had_contact=torch.ones(2),released=torch.ones(2)))
    action=policy_forward(actor,obs,dict(had_contact=torch.ones(2),released=torch.tensor([0.,1.])))[0]
    grad=torch.autograd.grad(action[1].sum(),actor.actor.recovery.bias)[0]
    assert grad.abs().sum()>0
    assert resolve_contract(None)[0]['recovery_gate']==LEGACY
    assert resolve_contract({'adapter_schema':PHYSICAL_SCHEMA})[0]['recovery_gate']==POSTHIT
    with pytest.raises(ValueError):resolve_contract({'metadata':{'policy_execution':dict(contract(),extra='wrong')}})


def spec():
    return {'dual_head_quality':dict(heldout_lane_ids=[1,3],
        wm_min_event_episodes=dict(recovery={k:1 for k in ('touchdown','support_loss','body_ground','stable')},approach={k:1 for k in ('legal_contact','illegal_contact','ball_side_success','ball_side_failure')}),
        native_action_probes={h:dict(same_prefix_verified=True,paired_samples=3,errors={'direction':.1},upper_bounds={'direction':.2}) for h in ('recovery','approach')},
        wm_upper_bounds=dict(recovery={'16/angular_velocity_rmse':1.,'16/foot_contact_brier':.2,'16/body_ground_brier':.2},
                            approach={'4/ball_velocity_rmse':1.,'16/ball_forward_sign_error':.2}),
        critic_upper_bounds={h:dict(mae=1.,rmse=2.) for h in ('recovery','approach')})}


def test_quality_cannot_use_contact_only_evidence_for_both_heads():
    data=spec();assert quality_spec(data)['heldout_lane_ids']==[1,3]
    data['dual_head_quality']['native_action_probes']['recovery']['paired_samples']=0
    with pytest.raises(ValueError,match='recovery'):quality_spec(data)
    data=spec();del data['dual_head_quality']['wm_upper_bounds']['recovery']['16/body_ground_brier']
    with pytest.raises(ValueError,match='head-specific'):quality_spec(data)
    with pytest.raises(ValueError,match='dual_head_quality'):quality_spec({'runtime_upper_bounds':{'64/state_rmse':1}})


def bank():
    return dict(inputs=torch.tensor([[0.],[1.],[2.],[3.]]),rtg=torch.tensor([0.,1.,2.,3.]),
        identity=[(i,8,256) for i in range(4)],head_masks={h:torch.ones(4,dtype=torch.bool) for h in ('recovery','approach')},
        policy_id='frozen-1',actor_sha256='frozen-hash')


def test_native_critic_holdout_is_policy_labelled_readonly_and_not_train_loss():
    b=bank();train,val=split_native_value_rows(b,[1,3]);assert not (train&val).any()
    critic=nn.Linear(1,1,bias=False);critic.weight.data.fill_(1)
    before=copy.deepcopy(critic.state_dict())
    result=critic_holdout(critic,b,'approach',[1,3],current_actor_sha256='new-hash',limits={'mae':.01,'rmse':.01})
    assert result['label_policy_value_fit'] and not result['passed'] and not result['current_policy_matches_labels'] and result['sample_count']==2
    assert result['metrics']['rtg_ordering_accuracy']==1
    for key,value in before.items():assert torch.equal(value,critic.state_dict()[key])
    critic.weight.data.fill_(-1)
    result=critic_holdout(critic,b,'recovery',[1,3],current_actor_sha256='frozen-hash',limits={'mae':.01,'rmse':.01})
    assert not result['passed'] and result['metrics']['rtg_ordering_accuracy']==0
    b['head_masks']['recovery'].zero_()
    assert critic_holdout(critic,b,'recovery',[1,3],current_actor_sha256='frozen-hash')['missing_fields']
    b['identity'][0]=b['identity'][1]
    with pytest.raises(ValueError,match='Duplicate'):split_native_value_rows(b,[1,3])


def test_reference_bank_coverage_does_not_conflate_heads_or_invent_tracking():
    b=SimpleNamespace(entries=[dict(mode='recovery',terminal_verified=True,score_schema='historical-independent-complete5-v1')])
    assert reference_coverage(b,'recovery',.1)['success_guided_available']
    assert reference_coverage(b,'approach',.1)['mode']=='task-bptt-only'
    assert reference_coverage(b,'recovery',0)['reason']=='tracking-disabled'


def test_recovery_validation_uses_foot_events_not_ball_contact(monkeypatch):
    from quadsmash.rl import supermimic_wm_validation_v1 as v
    t,n=100,2;ev=torch.zeros(t,n,8);ev[5:,:,0]=1;ev[5:,:,1]=1
    data=dict(event_state=ev,next_event_state=ev.clone(),episode=torch.zeros(t,n,dtype=torch.long),
              terminated=torch.zeros(t,n,dtype=torch.bool),timeout=torch.zeros(t,n,dtype=torch.bool),
              feet=torch.zeros(t,n,4),next_feet=torch.zeros(t,n,4))
    data['next_feet'][10,:,0]=1
    starts=v.recovery_event_windows(data,[1]);assert starts['touchdown'].tolist()==[[9,1]]
    assert not len(v.event_aligned_windows(data,[1])['-1']['starts'])
    adapter=nn.Linear(1,1)
    monkeypatch.setattr(v,'validate_windows',lambda *a,**k:{'16':{'foot_contact_brier':.1}})
    result=v.runtime_head_check(adapter,data,[1],torch.zeros(2,6),'recovery',{'16/foot_contact_brier':.2})
    assert not result['passed'] and 'body_ground' in result['missing_events']
    result=v.runtime_head_check(adapter,data,[1],torch.zeros(2,6),'approach',{'4/ball_velocity_rmse':1.})
    assert not result['passed'] and 'support/legal_contact' in result['failures']


def test_real_training_and_eval_entrypoints_use_shared_contract():
    # Source contract guards the actual entry points in addition to toy math.
    root=Path(__file__).resolve().parents[1]
    training=(root/'scripts/train_supermimic_physical_block.py').read_text()
    evaluation=(root/'scripts/evaluate_supermimic_v1.py').read_text()
    recording=(root/'scripts/record_supermimic_selected_case.py').read_text()
    assert "policy_forward(actor,observation['policy'],env.rally" in training
    assert 'policy_forward(actor,before,env.rally' in evaluation
    assert 'policy_forward(learned_actor' in recording
    assert 'action = actor(before, sample=False)' not in evaluation
    assert training.index('quality=check_head_quality(block_name')<training.index('stats=trainer.imagine_update(state)',training.index('quality=check_head_quality(block_name'))


def test_native_value_policy_identity_includes_execution_gate_contract():
    b=bank();b['policy_execution']=contract(LEGACY)
    critic=nn.Linear(1,1,bias=False);critic.weight.data.fill_(1)
    old=critic_holdout(critic,b,'approach',[1,3],current_actor_sha256='frozen-hash',current_execution=contract(POSTHIT))
    assert old['label_execution_known'] and not old['current_policy_matches_labels']
    same=critic_holdout(critic,b,'approach',[1,3],current_actor_sha256='frozen-hash',current_execution=contract(LEGACY))
    assert same['current_policy_matches_labels']


def test_frozen_compact_encoder_preserves_both_head_masks():
    from test_supermimic_approach_closed_loop import row,Schema
    from quadsmash.rl.supermimic_frozen_refresh_v1 import FrozenRefresh
    def encoder(start):
        tick=start['before']['elapsed_tick']
        return dict(critic_input=tick[:,None].float(),head_masks=dict(approach=tick==0,recovery=tick>0))
    f=FrozenRefresh(2,2,.9,'frozen',Schema(),stage_ticks=(0,1),start_encoder=encoder)
    f.append(row(0,[0,0],[0,0],[False,False]));f.append(row(1,[1,1],[0,0],[True,True]))
    for stage,record,mask in f.calibration_rows():
        assert record['head_masks']['approach'][mask].eq(stage==0).all()
        assert record['head_masks']['recovery'][mask].eq(stage>0).all()


def test_promotion_rejects_correct_weights_evaluated_with_wrong_gate(tmp_path):
    import json
    from quadsmash.rl.supermimic_long_run_contract_v1 import verify_native_promotion,sha256
    checkpoint=tmp_path/'weights.pt';checkpoint.write_bytes(b'cpu-contract-fixture')
    metrics=tmp_path/'metrics.json';metrics.write_text(json.dumps(dict(evaluated_checkpoint_sha256=sha256(checkpoint),
        policy_execution=contract(LEGACY),native_episode_return_mean=-10,ball_side_complete_except_recovery=53,complete5=27,numerical_failures=0)))
    with pytest.raises(ValueError,match='execution contract mismatch'):verify_native_promotion(metrics,checkpoint,expected_execution=contract(POSTHIT))
