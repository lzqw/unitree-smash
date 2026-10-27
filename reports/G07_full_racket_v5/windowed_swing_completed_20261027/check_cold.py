"""Observe the existing independent cold128, then verify actual raw evidence."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
MAIN=HERE.parents[2]
ROOT=MAIN.parent/'unitree_smash_contact_recovery'
SEARCH=HERE.parent/'windowed_swing7_n128_01'
COLD=HERE.parent/'windowed_swing7_best_cold128_01'
sys.path.insert(0,str(ROOT))
from quadsmash.control.full_racket_train_evaluation_v4 import training_cases,summarize_training
from quadsmash.evaluation_v4 import independent_episode


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())


def wait_terminal(seconds):
    started=time.monotonic();observations=[];identities={}
    while True:
        owner=read(HERE.parent/'windowed_swing_queue01/state.json')
        guard=owner.get('guard_pid')
        if owner.get('output')==str(COLD) and guard:
            proc=Path('/proc',str(guard))
            if proc.exists():
                command=(proc/'cmdline').read_bytes()
                assert str(COLD).encode() in command and b'run_gpu_job.py' in command
                ticks=(proc/'stat').read_text().rsplit(')',1)[1].split()[19]
                assert guard not in identities or identities[guard]==ticks
                identities[guard]=ticks
                for value in (proc/'task'/str(guard)/'children').read_text().split():
                    child=Path('/proc',value)
                    if child.exists() and str(COLD).encode() in (child/'cmdline').read_bytes():
                        ticks=(child/'stat').read_text().rsplit(')',1)[1].split()[19]
                        assert int(value) not in identities or identities[int(value)]==ticks
                        identities[int(value)]=ticks
        job=read(COLD/'job.json') if (COLD/'job.json').exists() else None
        live=[]
        for pid,ticks in identities.items():
            p=Path('/proc',str(pid))
            if p.exists() and (p/'stat').read_text().rsplit(')',1)[1].split()[19]==ticks:live.append(pid)
        if job and job['status'] in ['SUCCEEDED','FAILED','BUDGET_EXHAUSTED'] and not live:
            assert job['status']=='SUCCEEDED',job
            assert not Path('/proc',str(job['child_pid'])).exists()
            return identities,observations
        assert seconds>0,'Independent cold job has not been confirmed terminal'
        if time.monotonic()-started>seconds:
            raise TimeoutError('Read-only observation expired. This does not mean any native job stopped. Reinspect the same actual handles without restarting it.')
        progress=read(COLD/'progress.json') if (COLD/'progress.json').exists() else None
        observation=dict(observed_at=datetime.now(timezone.utc).isoformat(),actual_queue_state=owner['state'],
            live_existing_cold_handles=live,actual_progress=progress)
        if not observations or progress!=observations[-1]['actual_progress'] or observation['actual_queue_state']!=observations[-1]['actual_queue_state']:
            observations.append(observation);print(json.dumps(dict(waiting_for_existing_cold=True,**observation)),flush=True)
        time.sleep(5)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--wait-seconds',type=float,default=0.)
    args=parser.parse_args();identities,observations=wait_terminal(args.wait_seconds)
    selected_check=read(HERE/'ACTUAL_COMPLETED_SEARCH_CHECK.json')
    assert all(sha(Path(p))==h for p,h in selected_check['source_sha256'].items())
    policy_path=SEARCH/'best_candidate.pt';source=torch.load(policy_path,map_location='cpu',weights_only=False)
    protocol_path=ROOT/'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json'
    support_path=HERE.parent/'recovery_prior_queue01/source_support.py'
    spec=importlib.util.spec_from_file_location('independent_actual_early_source_admission',support_path)
    support=importlib.util.module_from_spec(spec);spec.loader.exec_module(support)
    admitted=support.verify_window_source(policy_path,source,COLD,ROOT,sha(protocol_path))
    job,completed,inputs,metrics=[read(COLD/n) for n in ['job.json','completed.json','INPUTS.json','metrics.json']]
    assert job['status']=='SUCCEEDED' and job['exit_code']==0 and job['receipt_present'] and job['gpu']==0
    assert 0<job['charged_env_steps']==completed['env_steps']<=102400
    assert not Path('/proc',str(job['child_pid'])).exists()
    protocol=read(protocol_path);protocol['_root']=str(ROOT)
    cases,indices=training_cases(ROOT,protocol)
    selected=selected_check['selected_candidate_id'];assert source['actual_native_search']['candidate_id']==selected
    rows=read(COLD/'episodes.json');initial=read(COLD/f'initial_ground_states_{selected:03d}.json')
    assert len(rows)==128 and indices==list(range(128))
    assert [row['case_id'] for row in rows]==[case['case_id'] for case in cases]
    assert [row['source_training_index'] for row in rows]==indices
    assert [row['episode_index'] for row in rows]==initial['episode_ids']==[0]*128
    assert all(row['ground_init'] and row['env_id']==i for i,row in enumerate(rows))
    assert initial['all_clocks_zero'] and initial['actual_environment_origins_m']==[[0.,0.,0.]]*128
    roots=np.asarray(initial['root_state_local'])
    assert abs(roots[:,:3]-protocol['initialization']['root_position_m']).max()<1e-5
    assert abs(roots[:,3:7]-[1.,0.,0.,0.]).max()<1e-5 and abs(roots[:,7:]).max()<1e-5
    assert abs(np.asarray(initial['joint_velocity_rad_s'])).max()<1e-5
    summary=summarize_training(rows,cases,indices,protocol);checked=[independent_episode(row,protocol) for row in rows]
    assert summary==admitted['source_cold_summary']
    assert summary['complete']==sum(x['complete'] for x in checked)
    assert summary['airborne_contacts']==sum(x['airborne_contact'] for x in checked)
    assert summary['conditional_recovery_count']==sum(x['conditional_recovery'] for x in checked)
    assert summary['numerical_failures']==0
    files=[Path(__file__),support_path,HERE/'ACTUAL_COMPLETED_SEARCH_CHECK.json',protocol_path,policy_path,
        *[COLD/n for n in ['job.json','completed.json','INPUTS.json','metrics.json','episodes.json',
            f'initial_ground_states_{selected:03d}.json',f'public_and_actions_{selected:03d}.npz']]]
    record=dict(checked_at=datetime.now(timezone.utc).isoformat(),selected_candidate_id=selected,selected_policy_sha256=sha(policy_path),
        independent_actual128_summary=summary,actual_control_env_steps=completed['env_steps'],
        charged_GPU0_hours=job['charged_gpu_hours'],search_and_cold_GPU0_hours=selected_check['charged_GPU0_hours']+job['charged_gpu_hours'],
        source_check=admitted,source_sha256={str(p):sha(p) for p in files},
        observed_original_handle_start_ticks=identities,verified_wait_observations=observations,
        cold128_and_search896_separate=True,original128_ground_initialization_zero_origins_and_episode_ids_preserved=True,
        formal_cases_read=False,formal_acceptance=False,goal_complete=False)
    with (HERE/'ACTUAL_INDEPENDENT128_CHECK.json').open('x') as f:f.write(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(selected_candidate_id=selected,complete=summary['complete'],airborne_contacts=summary['airborne_contacts'],
        conditional_recovery=summary['conditional_recovery_count'],attack=summary['attack_outgoing'],NF=summary['numerical_failures'],
        charged_GPU0_hours=job['charged_gpu_hours'],search_and_cold_GPU0_hours=record['search_and_cold_GPU0_hours'])),flush=True)


if __name__=='__main__':main()
