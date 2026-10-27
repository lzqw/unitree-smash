"""Recompute the first3 completed original128 lateral trials while later ones run."""
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
from scripts.initialize_centered_policy_v4 import same_tree
from quadsmash.control.full_racket_train_evaluation_v4 import training_cases
from quadsmash.control.full_racket_body_search_v4 import summarize_candidate
from quadsmash.evaluation_v4 import independent_episode


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    protocol_path=ROOT/'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json'
    protocol=read(protocol_path);protocol['_root']=str(ROOT)
    cases,indices=training_cases(ROOT,protocol)
    inputs=read(RUN/'INPUTS.json')
    assert indices==inputs['training_indices']==list(range(128))
    assert inputs['origin_spacing_m']==0. and inputs['seed']==724201
    assert inputs['reset_noise']=='none' and inputs['formal_cases_read'] is False
    source_path=HERE.parent/'measured_swing9_n128_01/best_candidate.pt'
    assert inputs['source_policy_sha256']==sha(source_path)
    source=torch.load(source_path,map_location='cpu',weights_only=False)
    metrics=read(RUN/'metrics.json')
    assert len(metrics['history'])>=3
    results=[]
    files=[Path(__file__),protocol_path,RUN/'INPUTS.json',source_path,
        ROOT/'quadsmash/evaluation_v4.py',ROOT/'quadsmash/control/full_racket_body_search_v4.py']
    for expected_id,entry in enumerate(metrics['history'][:3]):
        assert entry['candidate_id']==expected_id
        suffix=f'{expected_id:03d}'
        paths=[RUN/f'episodes_{suffix}.json',RUN/f'initial_ground_states_{suffix}.json',RUN/f'candidate_{suffix}.pt']
        hashes={str(p):sha(p) for p in paths}
        rows,initial=map(read,paths[:2])
        assert len(rows)==128 and initial['all_clocks_zero']
        assert initial['actual_environment_origins_m']==[[0.,0.,0.]]*128
        assert [row['episode_index'] for row in rows]==initial['episode_ids']==entry['summary']['native_generation_start_episode_ids']
        assert [row['case_id'] for row in rows]==[case['case_id'] for case in cases]
        assert [row['source_training_index'] for row in rows]==indices
        assert all(row['env_id']==i and row['ground_init'] for i,row in enumerate(rows))
        summary=summarize_candidate(rows,protocol)
        assert all(summary[k]==entry['summary'][k] for k in summary)
        checked=[independent_episode(row,protocol) for row in rows]
        assert summary['complete']==sum(x['complete'] for x in checked)
        assert summary['airborne_contacts']==sum(x['airborne_contact'] for x in checked)
        assert summary['conditional_recovery_count']==sum(x['conditional_recovery'] for x in checked)
        assert summary['numerical_failures']==0
        policy=torch.load(paths[2],map_location='cpu',weights_only=False)
        assert policy['native_execution'] and policy['schema']=='g07-public-early-lateral-intercept-v5'
        assert policy['actual_native_search']['candidate_id']==expected_id
        assert same_tree(policy['source_public_policy'],source)
        assert torch.equal(policy['parameters'],source['parameters'])
        assert policy['early_lateral']=={k:entry['candidate'][k] for k in ['gain','start_clock_s']}
        air=[row for row,x in zip(rows,checked) if x['airborne_contact']]
        result=dict(candidate_id=expected_id,candidate=entry['candidate'],summary=summary,
            original128_case_order_ground_states_and_native_episode_ids_preserved=True,
            original_actor_and_normalizer_tree_unchanged=True,
            actual_air_contact_target_first_landings=sum(row['landed_target'] for row in air),
            actual_air_contact_body_landings=sum(row['contact_flight_recovery']['body_landing'] for row in air),
            policy_sha256=hashes[str(paths[2])])
        results.append(result)
        assert all(sha(Path(p))==h for p,h in hashes.items())
        files.extend(paths)
    assert [x['summary']['airborne_contacts'] for x in results]==[4,13,11]
    record=dict(checked_at=datetime.now(timezone.utc).isoformat(),completed_actual_episodes_checked=384,
        actual_env_steps_checked=307200,results=results,
        source_sha256={str(p):sha(p) for p in files},
        search_still_running=True,selected_policy_not_yet_frozen=True,independent_cold_pending=True,
        formal_cases_read=False,formal_acceptance=False,goal_complete=False)
    with (HERE/'ACTUAL_FIRST3_CHECK.json').open('x') as f:f.write(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(episodes=384,air_counts=[x['summary']['airborne_contacts'] for x in results],
        all_complete_and_recovery_and_attack_zero=True,NF=0)))


if __name__=='__main__':main()
