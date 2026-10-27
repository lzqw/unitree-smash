"""Recompute trial6 and revalidate the sealed first5 original128 evidence."""
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
    previous_path=HERE.parent/'public_stance_recovery_20261027/ACTUAL_FIRST5_CHECK.json'
    previous=read(previous_path)
    assert previous['completed_development_episodes_checked']==640
    assert all(sha(Path(n))==h for n,h in previous['source_sha256'].items())
    protocol_path=ROOT/'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json'
    protocol=read(protocol_path);protocol['_root']=str(ROOT)
    cases,indices=training_cases(ROOT,protocol)
    entries=read(RUN/'metrics.json')['history']
    assert [e['candidate_id'] for e in entries[:6]]==[0,1,2,3,5,6]
    for old,entry in zip(previous['results'],entries[:5]):
        assert old['candidate_id']==entry['candidate_id']
        actual=summarize_candidate(read(RUN/f"episodes_{entry['candidate_id']:03d}.json"),protocol)
        assert all(actual[k]==old['summary'][k]==entry['summary'][k] for k in actual)
    entry=entries[5]
    paths=[RUN/'episodes_006.json',RUN/'initial_ground_states_006.json',RUN/'candidate_006.pt']
    rows,initial=read(paths[0]),read(paths[1])
    assert len(rows)==128 and indices==list(range(128))
    assert [row['source_training_index'] for row in rows]==indices
    assert [row['case_id'] for row in rows]==[case['case_id'] for case in cases]
    assert [row['episode_index'] for row in rows]==initial['episode_ids']==entry['summary']['native_generation_start_episode_ids']
    assert initial['all_clocks_zero'] and initial['actual_environment_origins_m']==[[0.,0.,0.]]*128
    assert all(row['env_id']==i and row['ground_init'] for i,row in enumerate(rows))
    summary=summarize_candidate(rows,protocol);checked=[independent_episode(row,protocol) for row in rows]
    assert all(summary[k]==entry['summary'][k] for k in summary)
    assert summary['complete']==sum(x['complete'] for x in checked)==0
    assert summary['airborne_contacts']==sum(x['airborne_contact'] for x in checked)==5
    assert summary['conditional_recovery_count']==sum(x['conditional_recovery'] for x in checked)==0
    assert summary['attack_outgoing']==summary['numerical_failures']==0
    source_path=HERE.parent/'measured_swing9_n128_01/best_candidate.pt'
    source=torch.load(source_path,map_location='cpu',weights_only=False)
    policy=torch.load(paths[2],map_location='cpu',weights_only=False)
    assert same_tree(policy['source_public_policy'],source) and torch.equal(policy['parameters'],source['parameters'])
    assert policy['native_execution'] and policy['early_lateral']==dict(gain=1.,start_clock_s=2.3)
    assert policy['actual_native_search']['candidate_id']==6
    expected=dict(previous['source_sha256']);expected.update({str(p):sha(p) for p in [Path(__file__),previous_path,protocol_path,source_path,*paths]})
    record=dict(checked_at=datetime.now(timezone.utc).isoformat(),checked_completed_development_episodes=768,
        first6_air_counts=[4,13,11,7,5,5],complete=0,qualified_recovery=0,attack=0,numerical_failures=0,
        sixth_actual_summary=summary,first5_raw_data_and_summaries_revalidated=True,
        actual_case_order_ground_initialization_episode_ids_and_source_actor_tree_preserved=True,
        source_sha256=expected,last_candidate_and_selected_cold_pending=True,
        formal_cases_read=False,formal_acceptance=False,goal_complete=False)
    with (HERE/'ACTUAL_FIRST6_CHECK.json').open('x') as f:f.write(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(episodes=768,air_counts=[4,13,11,7,5,5],complete=0,recovery=0,attack=0,NF=0)))


if __name__=='__main__':main()
