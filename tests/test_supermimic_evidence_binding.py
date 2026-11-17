import copy,json
from pathlib import Path
import pytest
import torch
from torch import nn
from quadsmash.rl.supermimic_evidence_binding_v1 import runtime_identity,probe_applicability,current_value_trust,verify_fit_provenance
from quadsmash.rl.supermimic_dual_head_quality_v1 import native_calibration_rows,critic_holdout
from quadsmash.rl.supermimic_long_run_contract_v1 import verify_native_promotion,sha256
from quadsmash.rl.supermimic_wm_validation_v1 import independent_event_support


def test_resume_actual_weights_invalidate_old_probe_despite_unchanged_base_files():
    actor=nn.Linear(2,1);world=nn.Linear(2,1)
    identity=runtime_identity(actor,world,{'dt':.01},{'gate':'causal'},0)
    receipt=dict(runtime_identity=copy.deepcopy(identity))
    assert probe_applicability(receipt,identity)['current_version_valid']
    with torch.no_grad():world.weight.add_(.01)
    current=runtime_identity(actor,world,{'dt':.01},{'gate':'causal'},0)
    assert not probe_applicability(receipt,current)['current_version_valid']
    for field in ('actor_state_sha256','policy_version','policy_execution','task_contract_sha256'):
        changed=copy.deepcopy(identity);changed[field]='changed'
        assert probe_applicability(receipt,changed)['scope']=='historical-reference-only'


def test_multistage_calibration_never_uses_unknown_contract_or_heldout_lanes():
    b=dict(identity=[(lane,0,t) for t in (0,260,450,790) for lane in range(4)],policy_id='frozen')
    rows,meta=native_calibration_rows(b,{'gate':1},[1,3],[0,260,450,790])
    assert not len(rows) and not meta['eligible']
    b['policy_execution']={'gate':1}
    rows,meta=native_calibration_rows(b,{'gate':1},[1,3],[0,260,450,790])
    assert len(rows)==8 and len(meta['stage_ticks'])==4
    assert all(b['identity'][i][0] in (0,2) for i in rows)


def test_old_policy_rtg_fit_is_not_current_value_trust():
    b=dict(policy_execution={'gate':1},actor_sha256='old',accepted_updates=3)
    args=dict(actor_sha='new',execution={'gate':1},accepted_updates=5)
    assert not current_value_trust(b,**args)['trusted']
    delta=dict(final_normalized_command_delta=.002,final_joint_target_delta_rad=.02)
    limits=dict(accepted_update_age=4,final_normalized_command_delta=.003,final_joint_target_delta_rad=.03)
    assert current_value_trust(b,**args,command_delta=delta,limits=limits)['status']=='bounded-reference-surrogate'
    assert not current_value_trust(b,**dict(args,accepted_updates=8),command_delta=delta,limits=limits)['trusted']


def test_duplicate_event_windows_do_not_count_as_independent_episodes():
    d={'episode':torch.zeros(100,3,dtype=torch.long)}
    assert independent_event_support(d,torch.tensor([[1,1],[2,1],[3,1]]))==1
    assert independent_event_support(d,torch.tensor([[1,1],[2,2]]))==2


def test_missing_historical_fit_manifest_cannot_certify_holdout(tmp_path):
    assert not verify_fit_provenance({},tmp_path/'receipt.json')['verified']
    m=tmp_path/'fit.json';m.write_text(json.dumps(dict(model_sha256={'physical':'x'},training_episode_ids=['fit/1'])))
    r=dict(model_sha256={'physical':'x'},training_episode_ids=['fit/1'],validation_episode_ids=['val/2'],wm_fit_provenance=dict(path=m.name,sha256=sha256(m)))
    assert verify_fit_provenance(r,tmp_path/'receipt.json')['verified']
    r['validation_episode_ids']=['fit/1']
    with pytest.raises(ValueError,match='overlap'):verify_fit_provenance(r,tmp_path/'receipt.json')


def test_last_cycle_promotion_requires_complete_protocol(tmp_path):
    c=tmp_path/'candidate';c.write_text('unchanged')
    m=dict(num_envs=128,control_ticks=799,evaluated_checkpoint_sha256=sha256(c),numerical_failures=0,
           native_episode_return_mean=-10,ball_side_complete_except_recovery=53,complete5=27)
    p=tmp_path/'metrics.json';p.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='Incomplete'):verify_native_promotion(p,c)
    m['control_ticks']=800;p.write_text(json.dumps(m));assert verify_native_promotion(p,c)
    launcher=(Path(__file__).resolve().parents[1]/'scripts/run_supermimic_phase1_long_v1.sh').read_text()
    assert launcher.index('verify_supermimic_promotion.py')<launcher.index('PREV_EVAL="$EVAL/metrics.json"')


def test_terminal_scores_are_explicit_offline_labels_not_guessed_outcomes():
    from quadsmash.rl.supermimic_evidence_binding_v1 import attach_terminal_ball_side
    d=dict(episode=torch.zeros(4,2,dtype=torch.long),terminated=torch.zeros(4,2,dtype=torch.bool),timeout=torch.zeros(4,2,dtype=torch.bool))
    d['timeout'][-1]=True
    row=dict(env_id=0,episode_index=0,scoring_schema='historical-independent-complete5-v1',ball_side_complete_except_recovery=False)
    result=attach_terminal_ball_side(d,[row])
    assert result['ball_side_terminal'][-1,0]==0 and torch.isnan(result['ball_side_terminal'][:,1]).all()
    with pytest.raises(ValueError,match='Duplicate'):attach_terminal_ball_side(d,[row,row])
    with pytest.raises(ValueError,match='Missing'):attach_terminal_ball_side(d,[dict(row,ball_side_complete_except_recovery=None)])


def test_resume_fit_provenance_binds_actual_world_and_historical_holdout():
    from quadsmash.rl.supermimic_evidence_binding_v1 import loaded_fit_provenance
    initial=dict(verified=True,manifest_sha256='base-fit')
    assert not loaded_fit_provenance(initial,{},'new-world',[1,3])['verified']
    resumed=dict(wm_fit_provenance=dict(verified=True,world_state_sha256='new-world',excluded_lanes=[1,3],initial_manifest_sha256='base-fit'))
    assert loaded_fit_provenance(initial,resumed,'new-world',[1,3])['verified']
    assert not loaded_fit_provenance(initial,resumed,'other-world',[1,3])['verified']
    assert not loaded_fit_provenance(initial,resumed,'new-world',[1,5])['verified']


def test_nan_drift_limit_never_passes():
    b=dict(policy_execution={},actor_sha256='old',accepted_updates=0)
    r=current_value_trust(b,actor_sha='new',execution={},accepted_updates=1,
        command_delta=dict(final_normalized_command_delta=.1,final_joint_target_delta_rad=.1),
        limits=dict(accepted_update_age=3,final_normalized_command_delta=float('nan'),final_joint_target_delta_rad=1))
    assert not r['trusted']


def test_mc_labels_exclude_partial_tail_and_never_cross_reset():
    from quadsmash.rl.supermimic_evidence_binding_v1 import complete_native_rtg
    d=dict(reward=torch.tensor([[1.],[2.],[100.],[200.]]),episode=torch.tensor([[0],[0],[1],[1]]),
           terminated=torch.tensor([[False],[True],[False],[False]]),timeout=torch.zeros(4,1,dtype=torch.bool))
    rtg,complete=complete_native_rtg(d,.5)
    assert rtg.flatten().tolist()==[2.,2.,0.,0.]
    assert complete.flatten().tolist()==[True,True,False,False]
    d['terminated'][1]=False
    assert not complete_native_rtg(d,.5)[1].any()


def test_stage_audit_excludes_unfinished_labels():
    from types import SimpleNamespace
    from quadsmash.rl.supermimic_physical_learner_v1 import PhysicalHorizonTrainer
    result=PhysicalHorizonTrainer.audit_native_rtg_stages([SimpleNamespace(observation=torch.zeros(2,794))],
        [torch.tensor([3.,999.])],[dict(policy_id='frozen',episode_id=[0,1],time_s=3.,complete_lane_mask=torch.tensor([True,False]))])
    assert result['total_labels']==1 and result['stages'][0]['return_mean']==3
    assert result['stages'][0]['excluded_incomplete_lanes']==1


def test_native_calibrator_aligns_bank_dtype_with_critic_without_changing_budget():
    from types import SimpleNamespace
    from quadsmash.rl.supermimic_physical_learner_v1 import PhysicalHorizonTrainer
    critic=nn.Linear(2,1).float()
    trainer=SimpleNamespace(critic=critic,critic_opt=torch.optim.Adam(critic.parameters(),lr=2e-4),critic_updates=0)
    r=PhysicalHorizonTrainer.calibrate_critic_tensors(trainer,torch.ones(3,2,dtype=torch.float64),torch.ones(3,dtype=torch.float64),epochs=2)
    assert trainer.critic_updates==2 and r['native_critic_calibration_updates']==2


def test_source_checkpoint_name_cannot_stand_in_for_actual_archive_actor():
    b=dict(identity=[(0,0,0)],policy_id='declared',policy_execution={})
    rows,meta=native_calibration_rows(b,{},[],[0],expected_actor_sha256='current')
    assert not len(rows) and 'actor-identity' in meta['reason']
    b['actor_sha256']='different'
    assert not native_calibration_rows(b,{},[],[0],expected_actor_sha256='current')[1]['eligible']
    b['actor_sha256']='current'
    assert native_calibration_rows(b,{},[],[0],expected_actor_sha256='current')[1]['eligible']


def test_unknown_actor_labels_report_unknown_not_value_fit_pass():
    result=critic_holdout(nn.Linear(1,1),{'actor_sha256':None},'recovery',[1],current_actor_sha256='actual')
    assert result['label_policy_value_fit'] is None and not result['passed']
    assert result['current_policy_value_trust']['status']=='unknown'
