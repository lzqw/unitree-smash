import hashlib
import json
from pathlib import Path

import pytest

from quadsmash.rl.supermimic_independent_panel import fixed128_cases, load_panel, panel_batch
from quadsmash.rl.supermimic_native_metrics import RETURN_DEFINITION
from scripts.evaluate_supermimic_independent512 import collect_scores


def test_frozen_research_panel_is_disjoint_from_original_suites():
    root = Path(__file__).resolve().parents[1]
    protocol_file = root/'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json'
    protocol = json.loads(protocol_file.read_text())
    digest = hashlib.sha256(protocol_file.read_bytes()).hexdigest()
    cases = []
    for i in range(4):
        batch, identity = panel_batch(i,digest,882072)
        assert len(batch)==identity['case_count']==128
        assert sum(c['panel']=='A' for c in batch)==64
        cases.extend(batch)
    def launches(rows):
        return {tuple(r['launch']['position_m']+r['launch']['velocity_mps']) for r in rows}
    assert len(launches(cases)) == len({r['case_id'] for r in cases}) == 512
    fixed, marker = fixed128_cases(digest,882072)
    assert len(launches(fixed)) == len({r['case_id'] for r in fixed}) == marker['case_count'] == 128
    assert sum(r['panel']=='A' for r in fixed) == 64
    assert launches(fixed).isdisjoint(launches(cases))
    for suite in [protocol['training_suite'],*protocol['formal_suites'].values()]:
        old = json.loads((root/suite['path']).read_text())['cases']
        assert launches(cases).isdisjoint(launches(old))
        assert launches(fixed).isdisjoint(launches(old))
    with pytest.raises(ValueError):
        panel_batch(0,digest,0)


def test_combined_metrics_use_all_contacts_and_reject_wrong_checkpoint(tmp_path):
    panel = load_panel()
    for i in range(4):
        path = tmp_path/f'batch_{i}'
        path.mkdir()
        cases, identity = panel_batch(i,panel['protocol_sha256'],882072)
        task = dict(protocol_sha256=panel['protocol_sha256'],independent_evaluation=identity)
        rows = [dict(case_id=c['case_id'],env_id=n,numerical_failures=0,
                     qualified_airborne_string_contact=n<=i,recovery_success=n==0,
                     ball_side_complete_except_recovery=False,complete5=False,native_episode_return=float(i),
                     native_episode_length_steps=800,native_return_definition=RETURN_DEFINITION)
                for n,c in enumerate(cases)]
        files = dict(startup=dict(argv=['--seed','882072']),completed=dict(control_ticks=800,episodes=128),
            episodes_summary=rows, task_contract=task,policy_execution=dict(contract={}),physics={},initial_states=[],
            metrics=dict(num_envs=128,actor_parameters_unchanged=True,optimizer_updates=dict(actor=0),
                         policy_execution={},task_contract=task,actor_state_sha256='actor',
                         evaluated_checkpoint_sha256='checkpoint',scoring_schema='historical-independent-complete5-v1'))
        for name, payload in files.items():
            (path/(name+'.json')).write_text(json.dumps(payload))
    summary=collect_scores(tmp_path,'checkpoint',panel)
    assert summary['episodes']==512
    assert summary['metrics']['conditional_recovery_count']==4
    assert summary['metrics']['conditional_recovery_denominator']==10
    assert summary['metrics']['native_episode_return_mean']==pytest.approx(1.5)
    with pytest.raises(ValueError):
        collect_scores(tmp_path,'other',panel)


def test_research_fixed128_reuses_matching_zero_and_rejects_training_suite(tmp_path):
    from scripts.evaluate_supermimic_research_arm import score_fixed128
    folder = tmp_path/'zero'
    folder.mkdir()
    protocol = load_panel()['protocol_sha256']
    cases, marker = fixed128_cases(protocol,882072)
    task = dict(protocol_sha256=protocol, research_evaluation=marker)
    rows = [dict(case_id=c['case_id'],env_id=i) for i,c in enumerate(cases)]
    metrics = dict(num_envs=128, actor_parameters_unchanged=True, optimizer_updates=dict(actor=0),
                   policy_execution={}, task_contract=task, actor_state_sha256='actor',
                   evaluated_checkpoint_sha256='checkpoint', scoring_schema='historical-independent-complete5-v1')
    files = dict(startup=dict(argv=['--seed','882072']), completed=dict(control_ticks=800,episodes=128),
                 metrics=metrics, episodes_summary=rows, task_contract=task,policy_execution=dict(contract={}),
                 physics={},initial_states=[])
    for name, value in files.items():
        (folder/(name+'.json')).write_text(json.dumps(value))
    def cached(actor):
        return score_fixed128(folder, tmp_path/'weights', 'checkpoint', actor, tmp_path, 882072)
    assert cached('actor') == metrics
    with pytest.raises(ValueError):
        cached('different')
    rows[0]['case_id'] = 'training-case'
    (folder/'episodes_summary.json').write_text(json.dumps(rows))
    with pytest.raises(ValueError):
        cached('actor')


def test_fixed_command_metrics_keep_empty_gate_groups_distinct_from_zero_motion():
    from scripts.evaluate_supermimic_research_arm import command_scalars

    report = dict(before_checkpoint_sha256='old_ck', after_checkpoint_sha256='new_ck',
                  before_actor_sha256='old_actor', after_actor_sha256='new_actor',archive_manifest_sha256='archive',
                  parameter_delta={'rms':.01}, groups={'recovery':{'groups':{
                      'recovery_only':dict(rows=3, final_target_rad=dict(rms=.02,changed_rows=2),gaussian_kl_per_dim=.1),
                      'both_heads_inactive':dict(rows=1, final_target_rad=dict(rms=0.,changed_rows=0),gaussian_kl_per_dim=0.),
                      'both_heads_active':{'rows':0},'approach_only':{'rows':0}}}})
    identity = dict(before_checkpoint='old_ck',after_checkpoint='new_ck',before_actor='old_actor',
                    after_actor='new_actor',archive_hash='archive')
    metrics=command_scalars(report,**identity)
    assert metrics['recovery/recovery_only/changed_rows']==2
    assert metrics['recovery/both_heads_inactive/final_target_rms_rad']==0.
    assert metrics['recovery/approach_only/rows']==0
    assert 'recovery/approach_only/final_target_rms_rad' not in metrics
    with pytest.raises(ValueError,match='different checkpoints'):
        command_scalars(report,**dict(identity,archive_hash='other'))
