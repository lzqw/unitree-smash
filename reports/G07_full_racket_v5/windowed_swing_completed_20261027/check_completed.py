"""Check all7 real completed windowed-swing trials; optionally wait on the live job."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

HERE=Path(__file__).resolve().parent
MAIN=HERE.parents[2]
ROOT=MAIN.parent/'unitree_smash_contact_recovery'
RUN=HERE.parent/'windowed_swing7_n128_01'
sys.path.insert(0,str(ROOT))
from quadsmash.control.full_racket_train_evaluation_v4 import training_cases
from quadsmash.control.full_racket_body_search_v4 import summarize_candidate
from quadsmash.evaluation_v4 import independent_episode
from scripts.initialize_centered_policy_v4 import same_tree


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def wait_terminal(maximum_seconds):
    started=time.monotonic();observations=[]
    identities={1994961:'71584847',1994976:'71584859'}
    while True:
        job=read(RUN/'job.json') if (RUN/'job.json').exists() else None
        live=[]
        for pid,ticks in identities.items():
            proc=Path('/proc')/str(pid)
            if proc.exists():
                actual=(proc/'stat').read_text().rsplit(')',1)[1].split()[19]
                if actual==ticks:live.append(pid)
        if job and job['status'] in ['SUCCEEDED','FAILED','BUDGET_EXHAUSTED'] and not live:
            assert job['status']=='SUCCEEDED',job
            return observations
        assert maximum_seconds>0,'Search has not been confirmed terminal; use --wait-seconds to observe it'
        if time.monotonic()-started>maximum_seconds:
            raise TimeoutError('Observation expired; this does not mean the native job stopped. Reinspect the same live handles without restarting it.')
        observation=dict(observed_at=datetime.now(timezone.utc).isoformat(),live_original_job_handles=live,
            actual_progress=read(RUN/'progress.json'),job_status=(job['status'] if job else None))
        if not observations or observation['actual_progress']!=observations[-1]['actual_progress']:
            observations.append(observation)
            print(json.dumps(dict(waiting_for_terminal=True,**observation)),flush=True)
        time.sleep(5)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-seconds',type=float,default=0.)
    args=parser.parse_args()
    observations=wait_terminal(args.wait_seconds)
    protocol_path=ROOT/'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json'
    protocol=read(protocol_path);protocol['_root']=str(ROOT)
    cases,indices=training_cases(ROOT,protocol)
    job,done,inputs,metrics=[read(RUN/n) for n in ['job.json','completed.json','INPUTS.json','metrics.json']]
    assert job['status']=='SUCCEEDED' and job['exit_code']==0 and job['receipt_present'] and job['gpu']==0
    assert 0<job['charged_env_steps']==done['env_steps']==metrics['actual_control_env_steps']<=716800
    assert inputs['training_indices']==indices==list(range(128))
    assert inputs['origin_spacing_m']==0. and inputs['reset_noise']=='none' and inputs['seed']==724201
    assert inputs['protocol_sha256']==sha(protocol_path) and inputs['formal_cases_read'] is False
    assert [e['candidate_id'] for e in metrics['history']]==list(range(7))
    source_path=HERE.parent/'early_lateral7_n128_01/best_candidate.pt'
    assert inputs['source_policy_sha256']==sha(source_path)
    source=torch.load(source_path,map_location='cpu',weights_only=False)
    factory=HERE.parent/'windowed_swing_20261027/window_controller.py'
    execution=read(RUN/'execution_source_hashes.json')
    assert len(execution)==394 and all(sha(ROOT/n)==h for n,h in execution.items())
    assert sha(ROOT/protocol['asset']['path'])==protocol['asset']['sha256']
    assert all(sha(ROOT/n)==h for n,h in protocol['physical_configs'].items())
    files=[Path(__file__),protocol_path,source_path,factory,*[RUN/n for n in
        ['job.json','completed.json','INPUTS.json','metrics.json','execution_source_hashes.json','best_candidate.pt']],
        ROOT/'quadsmash/evaluation_v4.py',ROOT/'quadsmash/control/full_racket_body_search_v4.py']
    results=[]
    for entry in metrics['history']:
        i=entry['candidate_id'];suffix=f'{i:03d}'
        paths=[RUN/f'episodes_{suffix}.json',RUN/f'initial_ground_states_{suffix}.json',RUN/f'candidate_{suffix}.pt']
        rows,initial=read(paths[0]),read(paths[1])
        assert len(rows)==128 and initial['all_clocks_zero']
        assert initial['actual_environment_origins_m']==[[0.,0.,0.]]*128
        assert [row['case_id'] for row in rows]==[case['case_id'] for case in cases]
        assert [row['source_training_index'] for row in rows]==indices
        assert [row['episode_index'] for row in rows]==initial['episode_ids']==entry['summary']['native_generation_start_episode_ids']
        assert all(row['ground_init'] and row['env_id']==lane for lane,row in enumerate(rows))
        summary=summarize_candidate(rows,protocol);independent=[independent_episode(row,protocol) for row in rows]
        assert all(summary[k]==entry['summary'][k] for k in summary)
        assert summary['complete']==sum(x['complete'] for x in independent)
        assert summary['airborne_contacts']==sum(x['airborne_contact'] for x in independent)
        assert summary['conditional_recovery_count']==sum(x['conditional_recovery'] for x in independent)
        assert summary['numerical_failures']==0
        policy=torch.load(paths[2],map_location='cpu',weights_only=False)
        assert policy['native_execution'] and policy['schema']=='g07-windowed-swing-contact-recovery-v5'
        assert policy['factory_sha256']==sha(factory)
        assert policy['actual_native_search']['candidate_id']==i and policy['actual_native_search']['original128_summary']==entry['summary']
        assert same_tree(policy['source_public_policy'],source) and torch.equal(policy['parameters'],source['parameters'])
        assert policy['windowed_swing']=={k:entry['candidate'][k] for k in ['joint_velocity_rad_s']}
        results.append(dict(candidate_id=i,candidate=entry['candidate'],summary=summary,actual_env_steps=entry['env_steps'],
            original_actor_normalizer_and_ground_case_order_and_actual_episode_ids_preserved=True,policy_sha256=sha(paths[2])))
        files.extend(paths)
    selected=max(metrics['history'],key=lambda e:tuple(e['summary']['rank']))
    assert sha(RUN/'best_candidate.pt')==sha(RUN/f"candidate_{selected['candidate_id']:03d}.pt")
    record=dict(checked_at=datetime.now(timezone.utc).isoformat(),actual_completed_development_episodes=896,
        actual_control_env_steps=done['env_steps'],charged_GPU0_hours=job['charged_gpu_hours'],results=results,
        selected_candidate_id=selected['candidate_id'],selected_definition=selected['candidate'],
        original_predeclared_rank_used=True,actual_runtime_source_files_checked=394,
        source_sha256={str(p):sha(p) for p in files},verified_wait_observations=observations,
        independent_selected_cold128_is_separate=True,formal_cases_read=False,formal_acceptance=False,goal_complete=False)
    with (HERE/'ACTUAL_COMPLETED_SEARCH_CHECK.json').open('x') as f:f.write(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(episodes=896,actual_env_steps=done['env_steps'],charged_GPU0_hours=job['charged_gpu_hours'],
        air_counts=[x['summary']['airborne_contacts'] for x in results],selected_candidate_id=selected['candidate_id'],
        selected_definition=selected['candidate'],complete_counts=[x['summary']['complete'] for x in results],
        recovery_counts=[x['summary']['conditional_recovery_count'] for x in results],NF=0)),flush=True)


if __name__=='__main__':main()
