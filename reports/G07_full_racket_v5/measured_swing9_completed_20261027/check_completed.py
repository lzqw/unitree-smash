"""Independently recompute all1152 actual completed search episodes."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
MAIN=HERE.parents[2]
ROOT=MAIN.parent/'unitree_smash_contact_recovery'
SEARCH=HERE.parent/'measured_swing9_n128_01'
sys.path.insert(0,str(ROOT))
from quadsmash.control.full_racket_train_evaluation_v4 import training_cases
from quadsmash.control.full_racket_body_search_v4 import summarize_candidate
from quadsmash.evaluation_v4 import independent_episode
from scripts.initialize_centered_policy_v4 import same_tree


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    job,completed,inputs,metrics=[read(SEARCH/n) for n in ['job.json','completed.json','INPUTS.json','metrics.json']]
    assert job['status']=='SUCCEEDED' and job['exit_code']==0 and job['receipt_present']
    assert not Path('/proc',str(job['child_pid'])).exists() and not Path('/proc',str(job['pid'])).exists()
    assert job['gpu']==0 and job['charged_env_steps']==completed['env_steps']==921600
    assert len(metrics['history'])==9 and inputs['training_indices']==list(range(128))
    assert inputs['origin_spacing_m']==0. and inputs['seed']==724201 and inputs['formal_cases_read'] is False
    protocol_path=ROOT/'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json'
    protocol=read(protocol_path);protocol['_root']=str(ROOT)
    cases,indices=training_cases(ROOT,protocol)
    original_path=MAIN/'reports/G07_full_racket_v4/gain_body8_sequential_n128_01/best_candidate.pt'
    original=torch.load(original_path,map_location='cpu',weights_only=False)
    assert sha(original_path)==inputs['source_policy_sha256']
    execution=read(SEARCH/'execution_source_hashes.json')
    assert len(execution)==394 and all(sha(ROOT/n)==h for n,h in execution.items())
    results=[];sources=[Path(__file__),original_path,protocol_path]
    for entry in metrics['history']:
        i=entry['candidate_id'];suffix=f'{i:03d}'
        rows_path=SEARCH/f'episodes_{suffix}.json';initial_path=SEARCH/f'initial_ground_states_{suffix}.json'
        policy_path=SEARCH/f'candidate_{suffix}.pt'
        rows,initial=read(rows_path),read(initial_path)
        assert len(rows)==128 and initial['all_clocks_zero']
        assert initial['actual_environment_origins_m']==[[0.,0.,0.]]*128
        assert [r['episode_index'] for r in rows]==initial['episode_ids']==entry['summary']['native_generation_start_episode_ids']
        for lane,(row,case) in enumerate(zip(rows,cases)):
            assert row['env_id']==lane and row['source_training_index']==indices[lane]
            assert row['case_id']==case['case_id'] and row['ground_init']
        summary=summarize_candidate(rows,protocol)
        assert all(summary[k]==entry['summary'][k] for k in summary)
        independent=[independent_episode(row,protocol) for row in rows]
        assert summary['complete']==sum(x['complete'] for x in independent)
        assert summary['airborne_contacts']==sum(x['airborne_contact'] for x in independent)
        assert summary['conditional_recovery_count']==sum(x['conditional_recovery'] for x in independent)
        assert summary['numerical_failures']==0 and entry['env_steps']==102400 and entry['control_ticks']==800
        policy=torch.load(policy_path,map_location='cpu',weights_only=False)
        assert policy['native_execution'] and policy['actual_native_search']['candidate_id']==i
        assert same_tree(policy['source_public_policy'],original)
        assert torch.equal(policy['parameters'],original['parameters'])
        assert policy['runtime_overrides']==entry['candidate']['overrides']
        result=dict(candidate_id=i,name=entry['candidate']['name'],actual_episodes=128,
            complete=summary['complete'],airborne_contacts=summary['airborne_contacts'],
            conditional_recovery_count=summary['conditional_recovery_count'],attack_outgoing=summary['attack_outgoing'],
            mean_TRAIN_partial_score=summary['mean_TRAIN_partial_score'],numerical_failures=0,
            original_actor_and_normalizer_tree_unchanged=True,actual_episode_ids_preserved=True,
            policy_sha256=sha(policy_path))
        results.append(result);sources.extend([rows_path,initial_path,policy_path])
    selected=max(metrics['history'],key=lambda e:tuple(e['summary']['rank']))
    selected_id=selected['candidate_id']
    assert selected_id==0
    assert sha(SEARCH/'best_candidate.pt')==sha(SEARCH/f'candidate_{selected_id:03d}.pt')
    sources.extend(SEARCH/n for n in ['job.json','completed.json','INPUTS.json','metrics.json','execution_source_hashes.json','best_candidate.pt'])
    result=dict(checked_at=datetime.now(timezone.utc).isoformat(),completed_actual_episodes=1152,
        actual_env_steps=921600,charged_GPU0_hours=job['charged_gpu_hours'],results=results,
        selected_candidate_id=selected_id,selected_runtime_overrides={},
        selected_reason='Original predeclared rank: no candidate completed or attacked/recovered; unchanged0 has highest mean TRAIN partial score despite7 having5 air contacts.',
        actual_execution_source_files_checked=394,
        source_sha256={str(p):sha(p) for p in sources},independent_selected_cold128_still_separate=True,
        formal_cases_read=False,formal_acceptance=False,goal_complete=False)
    with (HERE/'COMPLETED_SEARCH_CHECK.json').open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(actual_episodes=1152,env_steps=921600,charged_GPU0_hours=job['charged_gpu_hours'],
        selected_candidate_id=selected_id,airborne_counts=[r['airborne_contacts'] for r in results],
        all_complete_and_recovery_zero=True,numerical_failures=0)))


if __name__=='__main__':main()
