"""Recompute completed5 lateral trials; keep later active generations separate."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys

import torch

HERE=Path(__file__).resolve().parent
MAIN=HERE.parents[2]
ROOT=MAIN.parent/'unitree_smash_contact_recovery'
RUN=HERE.parent/'early_lateral7_n128_01'
sys.path.insert(0,str(ROOT))
from quadsmash.control.full_racket_train_evaluation_v4 import training_cases
from quadsmash.control.full_racket_body_search_v4 import summarize_candidate
from quadsmash.evaluation_v4 import independent_episode
from scripts.initialize_centered_policy_v4 import same_tree


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    protocol_path=ROOT/'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json'
    protocol=read(protocol_path);protocol['_root']=str(ROOT)
    cases,indices=training_cases(ROOT,protocol);inputs=read(RUN/'INPUTS.json')
    assert inputs['training_indices']==indices==list(range(128))
    assert inputs['origin_spacing_m']==0. and inputs['reset_noise']=='none' and inputs['seed']==724201
    assert inputs['formal_cases_read'] is False
    source_path=HERE.parent/'measured_swing9_n128_01/best_candidate.pt'
    assert sha(source_path)==inputs['source_policy_sha256']
    source=torch.load(source_path,map_location='cpu',weights_only=False)
    entries=read(RUN/'metrics.json')['history'][:5]
    assert [e['candidate_id'] for e in entries]==[0,1,2,3,5]
    results=[];files=[Path(__file__),protocol_path,RUN/'INPUTS.json',source_path,
        ROOT/'quadsmash/evaluation_v4.py',ROOT/'quadsmash/control/full_racket_body_search_v4.py']
    for entry in entries:
        suffix=f"{entry['candidate_id']:03d}"
        paths=[RUN/f'episodes_{suffix}.json',RUN/f'initial_ground_states_{suffix}.json',RUN/f'candidate_{suffix}.pt']
        rows,initial=map(read,paths[:2])
        assert len(rows)==128 and initial['all_clocks_zero']
        assert initial['actual_environment_origins_m']==[[0.,0.,0.]]*128
        assert [row['case_id'] for row in rows]==[case['case_id'] for case in cases]
        assert [row['source_training_index'] for row in rows]==indices
        assert [row['episode_index'] for row in rows]==initial['episode_ids']==entry['summary']['native_generation_start_episode_ids']
        assert all(row['env_id']==i and row['ground_init'] for i,row in enumerate(rows))
        summary=summarize_candidate(rows,protocol);checked=[independent_episode(row,protocol) for row in rows]
        assert all(summary[k]==entry['summary'][k] for k in summary)
        assert summary['airborne_contacts']==sum(x['airborne_contact'] for x in checked)
        assert summary['complete']==sum(x['complete'] for x in checked)==0
        assert summary['conditional_recovery_count']==sum(x['conditional_recovery'] for x in checked)==0
        assert summary['attack_outgoing']==summary['numerical_failures']==0
        policy=torch.load(paths[2],map_location='cpu',weights_only=False)
        assert policy['native_execution'] and policy['schema']=='g07-public-early-lateral-intercept-v5'
        assert policy['actual_native_search']['candidate_id']==entry['candidate_id']
        assert same_tree(policy['source_public_policy'],source) and torch.equal(policy['parameters'],source['parameters'])
        assert policy['early_lateral']=={k:entry['candidate'][k] for k in ['gain','start_clock_s']}
        results.append(dict(candidate_id=entry['candidate_id'],candidate=entry['candidate'],summary=summary,
            original128_order_ground_initialization_actual_episode_ids_and_source_actor_tree_preserved=True,
            actual_control_env_steps=entry['env_steps'],policy_sha256=sha(paths[2])))
        files.extend(paths)
    assert [x['summary']['airborne_contacts'] for x in results]==[4,13,11,7,5]
    record=dict(checked_at=datetime.now(timezone.utc).isoformat(),completed_development_episodes_checked=640,
        actual_control_env_steps_checked=sum(x['actual_control_env_steps'] for x in results),results=results,
        source_sha256={str(p):sha(p) for p in files},later_candidates_still_running=True,
        independent_selected_cold_pending=True,formal_cases_read=False,formal_acceptance=False,goal_complete=False)
    with (HERE/'ACTUAL_FIRST5_CHECK.json').open('x') as f:f.write(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(episodes=640,air_counts=[4,13,11,7,5],complete=0,recovery=0,attack=0,NF=0)))


if __name__=='__main__':main()
