"""Admit completed stance4/cold128 before any shared-target GPU job."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):return json.loads(Path(path).read_text())


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_stance_source(path,source,evaluation,root,protocol_sha256):
    from scripts.initialize_centered_policy_v4 import same_tree
    from quadsmash.control.full_racket_train_evaluation_v4 import training_cases,summarize_training
    from quadsmash.control.full_racket_body_search_v4 import summarize_candidate
    from quadsmash.evaluation_v4 import independent_episode
    path,evaluation,root=Path(path),Path(evaluation),Path(root)
    reports=root.parent/'unitree_smash/reports/G07_full_racket_v5'
    factory_path=reports/'public_stance_recovery_20261027/native_factory.py'
    protocol_path=root/'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json'
    assert source['schema']=='g07-public-stance-recovery-v5' and source['native_execution']
    assert source['factory_sha256']==sha(factory_path)
    assert source['protocol_sha256']==sha(protocol_path)==protocol_sha256
    model_path=reports/'centroidal_recovery_model_20261027/MODEL.json'
    core_path=reports/'public_stance_recovery_20261027/stance_controller.py'
    assert source['known_model_sha256']==sha(model_path) and source['known_model']==read(model_path)
    assert source['stance_core_sha256']==sha(core_path)
    earlier_path=reports/'momentum_strike4_n128_01/best_candidate.pt'
    earlier_cold=reports/'momentum_strike4_best_cold128_01'
    earlier=torch.load(earlier_path,map_location='cpu',weights_only=False)
    assert same_tree(source['source_public_policy'],earlier)
    assert source['source_policy_sha256']==sha(earlier_path)
    support_path=reports/'stance_recovery_queue01/source_support.py'
    support=load('strike_preserved_countermotion_source_admission',support_path)
    previous=support.verify_strike_source(earlier_path,earlier,earlier_cold,root,protocol_sha256)
    assert torch.equal(source['parameters'],earlier['parameters']) and source['asset_sha256']==earlier['asset_sha256']
    protocol=read(protocol_path);protocol['_root']=str(root)
    cases,indices=training_cases(root,protocol)
    assert indices==list(range(128))
    search=Path(source['actual_native_search']['output'])
    assert search==reports/'stance_recovery4_n128_01'
    search_job,search_metrics,search_completed=[read(search/n) for n in ['job.json','metrics.json','completed.json']]
    assert search_job['status']=='SUCCEEDED' and search_job['exit_code']==0 and search_job['receipt_present']
    assert not Path('/proc',str(search_job['child_pid'])).exists()
    assert 0<search_job['charged_env_steps']==search_completed['env_steps']==search_metrics['actual_control_env_steps']<=409600
    assert [entry['candidate_id'] for entry in search_metrics['history']]==list(range(4))
    files=[path,factory_path,model_path,core_path,support_path,protocol_path,
        *[search/n for n in ['job.json','metrics.json','completed.json','best_candidate.pt']]]
    for entry in search_metrics['history']:
        suffix=f"{entry['candidate_id']:03d}"
        rows=read(search/f'episodes_{suffix}.json')
        initial=read(search/f'initial_ground_states_{suffix}.json')
        assert len(rows)==128 and initial['all_clocks_zero']
        assert initial['actual_environment_origins_m']==[[0.,0.,0.]]*128
        assert [row['episode_index'] for row in rows]==initial['episode_ids']
        assert [row['case_id'] for row in rows]==[case['case_id'] for case in cases]
        assert [row['source_training_index'] for row in rows]==indices
        assert all(row['ground_init'] and row['env_id']==i for i,row in enumerate(rows))
        summary=summarize_candidate(rows,protocol)
        assert all(summary[k]==entry['summary'][k] for k in summary)
        independent=[independent_episode(row,protocol) for row in rows]
        assert summary['complete']==sum(x['complete'] for x in independent)
        assert summary['airborne_contacts']==sum(x['airborne_contact'] for x in independent)
        assert summary['conditional_recovery_count']==sum(x['conditional_recovery'] for x in independent)
        assert summary['numerical_failures']==0
        files.extend([search/f'episodes_{suffix}.json',search/f'initial_ground_states_{suffix}.json'])
    selected=max(search_metrics['history'],key=lambda e:tuple(e['summary']['rank']))
    selected_id=selected['candidate_id']
    assert source['actual_native_search']['candidate_id']==selected_id
    assert source['actual_native_search']['original128_summary']==selected['summary']
    assert sha(search/f'candidate_{selected_id:03d}.pt')==sha(search/'best_candidate.pt')==sha(path)
    files.append(search/f'candidate_{selected_id:03d}.pt')
    job,inputs,metrics,completed=[read(evaluation/n) for n in ['job.json','INPUTS.json','metrics.json','completed.json']]
    assert job['status']=='SUCCEEDED' and job['exit_code']==0 and job['receipt_present']
    assert not Path('/proc',str(job['child_pid'])).exists()
    assert 0<job['charged_env_steps']==completed['env_steps']<=102400
    assert inputs['mode']=='cold' and inputs['source_policy_sha256']==sha(path)
    assert inputs['protocol_sha256']==protocol_sha256 and inputs['training_indices']==indices
    assert inputs['origin_spacing_m']==0. and inputs['reset_noise']=='none' and inputs['seed']==724201
    assert inputs['formal_cases_read'] is False and len(metrics['history'])==1
    assert metrics['history'][0]['candidate_id']==selected_id
    rows=read(evaluation/'episodes.json')
    initial=read(evaluation/f'initial_ground_states_{selected_id:03d}.json')
    assert initial['all_clocks_zero'] and initial['episode_ids']==[0]*128
    assert initial['actual_environment_origins_m']==[[0.,0.,0.]]*128
    assert [row['episode_index'] for row in rows]==[0]*128
    assert [row['case_id'] for row in rows]==[case['case_id'] for case in cases]
    assert [row['source_training_index'] for row in rows]==indices
    assert all(row['ground_init'] and row['env_id']==i for i,row in enumerate(rows))
    summary=summarize_training(rows,cases,indices,protocol)
    for key in ['complete','airborne_contacts','conditional_recovery_count','numerical_failures',
                'actual_racket_contacts','crossed_net','landed_legal_court','landed_target','fault_counts']:
        assert summary[key]==metrics['history'][0]['summary'][key]
    assert summary['numerical_failures']==0
    hashes=read(evaluation/'execution_source_hashes.json')
    assert len(hashes)==394 and all(sha(root/n)==h for n,h in hashes.items())
    files.extend(evaluation/n for n in ['job.json','INPUTS.json','metrics.json','episodes.json','completed.json',
        'execution_source_hashes.json',f'initial_ground_states_{selected_id:03d}.json',
        f'public_and_actions_{selected_id:03d}.npz'])
    expected=dict(previous['source_sha256']);expected.update({str(p):sha(p) for p in files})
    return dict(source_policy_sha256=sha(path),source_cold_summary=summary,
        selected_stance_candidate_id=selected_id,previous_strike_source_check=previous,
        source_sha256=expected,source_weights_physics_and_cases_unchanged=True,
        all4_actual_stance_search_summaries_independently_recomputed=True,
        formal_cases_read=False,formal_acceptance=False)

