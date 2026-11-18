import copy,json,hashlib
from pathlib import Path
import pytest
import torch
from quadsmash.rl.supermimic_joint_sampling_v1 import sample_joint
from quadsmash.rl.supermimic_actor_start_bank_v1 import ActorStartBank
from quadsmash.rl.supermimic_sequential_progress_v1 import SequentialProgress
from quadsmash.rl.supermimic_long_run_contract_v1 import validate_budgets,verify_contact_receipt,verify_native_promotion,sha256
from quadsmash.rl.supermimic_wm_validation_v1 import event_aligned_windows


def test_joint_flow_reroutes_recent_quota_to_scarce_contact():
    all_pairs=torch.tensor([[0,0],[1,0],[2,0],[3,0]])
    pools={'uniform':all_pairs,'contact':all_pairs[:2]}
    source=torch.tensor([[1],[1],[0],[0]])
    chosen,r=sample_joint(pools,source,4,torch.Generator().manual_seed(5),source_quota={1:2,0:2},stage_quota={'uniform':2,'contact':2})
    assert r['joint_flow_count']==4 and r['stage_counts']=={'uniform':2,'contact':2}
    assert r['source_counts']=={0:2,1:2} and chosen.unique(dim=0).shape[0]==4


def test_global_dedup_and_episode_cap_not_silent_repeats():
    pairs=torch.arange(6)[:,None];pairs=torch.cat((pairs,torch.zeros_like(pairs)),1)
    source=torch.zeros(6,1,dtype=torch.long);episodes=torch.tensor([[0],[0],[0],[1],[1],[1]])
    chosen,r=sample_joint({'uniform':pairs,'contact':pairs},source,4,torch.Generator().manual_seed(1),
        stage_quota={'uniform':2,'contact':2},episode_keys=episodes,max_per_episode=2)
    assert r['actual_max_per_episode']==2 and len(chosen.unique(dim=0))==4
    with pytest.raises(ValueError,match='Insufficient unique'):
        sample_joint({'uniform':pairs},source,4,torch.Generator(),episode_keys=episodes,max_per_episode=1)


def start(ep=0):
    return dict(before=dict(episode_index=torch.tensor([ep,ep])),observation=torch.zeros(2,794),
                executor_before={'values':{'count':2},'tensors':{'counter':torch.tensor([3,8])}})


def test_actor_starts_terminal_labels_restore_rng_and_full_executor():
    bank=ActorStartBank(capacity=4)
    for tick in (0,4,8):bank.append(start(),{'approach_gate':torch.tensor([1.,1.]),'had_contact':torch.zeros(2)},stream=2,policy_version=1,native_tick=tick)
    record=dict(env_id=0,episode_index=0,scoring_schema='historical-independent-complete5-v1',ball_side_complete_except_recovery=False)
    bank.label_terminal(record,2)
    good=dict(record,env_id=1,ball_side_complete_except_recovery=True);bank.label_terminal(good,2)
    saved=copy.deepcopy(bank.state_dict());restored=ActorStartBank(capacity=4);restored.load_state_dict(saved)
    for attempt in range(1,21):
        a,ma=bank.sample(attempt);b,mb=restored.sample(attempt)
        assert ma==mb and torch.equal(a['_actor_lane_mask'],b['_actor_lane_mask'])
        assert a['executor_before']['values']['count']==2
        assert set(a)=={'before','observation','executor_before','_actor_lane_mask'}
    failure,_=bank.sample(13);assert failure['_actor_lane_mask'].tolist()==[True,False]
    success,_=bank.sample(18);assert success['_actor_lane_mask'].tolist()==[False,True]
    # A reset episode must not rewrite the older episode's sampling label.
    bank.label_terminal(dict(record,episode_index=1,ball_side_complete_except_recovery=True),2)
    assert not bank.entries[0]['masks']['success_near'][0]


def test_progress_resume_does_not_finish_after_recovery_only_or_change_budget():
    s=SequentialProgress();r=s.block(0,'recovery',4,8);r.update(attempts=6,accepted=4,status='done')
    assert not s.complete()
    a=s.block(0,'approach',4,8);a.update(attempts=2,accepted=1,status='training',pending_refresh=True)
    t=SequentialProgress();t.load_state_dict(s.state_dict());assert t.block(0,'approach',4,8)['attempts']==2
    with pytest.raises(ValueError,match='changes block budget'):t.block(0,'approach',8,8)
    with pytest.raises(ValueError,match='proposal budget'):validate_budgets(64,64,32,128)


def test_contact_receipt_requires_real_artifacts_matching_models_and_bounded_metrics(tmp_path):
    models={name:tmp_path/name for name in ('physical','ball','task')}
    for name,path in models.items():path.write_text(name)
    raw=tmp_path/'native_evidence.json';raw.write_text('{}')
    contract={'control_dt':.01}
    r=dict(schema='supermimic-contact-validation-v1',model_sha256={k:sha256(v) for k,v in models.items()},
           task_contract_sha256=hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
           training_episode_ids=['train/1'],validation_episode_ids=['val/1'],
           native_action_probe=dict(same_prefix_verified=True,paired_samples=2,
                                    response_direction_verified=True,independent_episode_count=1),thresholds_declared_before_validation=True,
           event_aligned_errors={'ball':.1},action_response_errors={'direction':.1},
           upper_bounds={'event_aligned_errors':{'ball':.2},'action_response_errors':{'direction':.2}},
           evidence_files=[dict(path=raw.name,sha256=sha256(raw))])
    receipt=tmp_path/'receipt.json';receipt.write_text(json.dumps(r))
    assert verify_contact_receipt(receipt,models,contract)['validation_episodes']==1
    r['action_response_errors']['direction']=1.;receipt.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='validation failed'):verify_contact_receipt(receipt,models,contract)
    r['action_response_errors']['direction']=.1;r['native_action_probe']['same_prefix_verified']=False;receipt.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='repeatable native'):verify_contact_receipt(receipt,models,contract)
    r['native_action_probe']['same_prefix_verified']=True;r['native_action_probe']['response_direction_verified']=False;receipt.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='direction'):verify_contact_receipt(receipt,models,contract)


def test_event_alignment_is_relative_to_edge_not_arbitrary_64step_window():
    n=90;e=torch.zeros(n,1,8);e[11:,:,1]=1
    d=dict(event_state=e,next_event_state=torch.cat((e[1:],e[-1:])),episode=torch.zeros(n,1,dtype=torch.long),
           terminated=torch.zeros(n,1,dtype=torch.bool),timeout=torch.zeros(n,1,dtype=torch.bool))
    groups=event_aligned_windows(d,[0])
    for offset in (-4,-1,0,1,4):
        assert groups[str(offset)]['starts'].tolist()==[[10+offset,0]]
        assert groups[str(offset)]['event_ticks']==[10]
    d['timeout'][9]=True
    assert len(event_aligned_windows(d,[0])['-1']['starts'])==0


def test_native_promotion_requires_matching_checkpoint_and_original_thresholds(tmp_path):
    checkpoint=tmp_path/'checkpoint';checkpoint.write_text('toy-checkpoint')
    metrics=tmp_path/'metrics.json'
    m=dict(num_envs=128,control_ticks=800,evaluated_checkpoint_sha256=sha256(checkpoint),numerical_failures=0,
           native_episode_metrics=dict(native_episode_return_mean=-10.,ball_side_complete_except_recovery=53,complete5=27))
    metrics.write_text(json.dumps(m));assert verify_native_promotion(metrics,checkpoint)['checkpoint_sha256']==sha256(checkpoint)
    m['native_episode_metrics']['ball_side_complete_except_recovery']=52;metrics.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='not accepted'):verify_native_promotion(metrics,checkpoint)


def test_joint_episode_caps_are_part_of_feasibility_not_posthoc_greedy():
    # Episode 0 contains scarce contact and a flexible uniform sample. The
    # allocator must reserve its sole slot for contact, even if uniform is first.
    pairs=torch.tensor([[0,0],[1,0],[2,0],[3,0]])
    pools={'uniform':pairs,'contact':pairs[:1]}
    sources=torch.tensor([[1],[1],[0],[0]])
    episodes=torch.tensor([[0],[0],[1],[2]])
    for seed in range(5):
        chosen,r=sample_joint(pools,sources,3,torch.Generator().manual_seed(seed),
            source_quota={1:1,0:2},stage_quota={'uniform':2,'contact':1},episode_keys=episodes,max_per_episode=1)
        assert r['joint_quota_feasible'] and r['fallback_count']==0
        assert r['stage_counts']=={'uniform':2,'contact':1}
        assert chosen[:,0].sort().values.tolist()==[0,2,3]
