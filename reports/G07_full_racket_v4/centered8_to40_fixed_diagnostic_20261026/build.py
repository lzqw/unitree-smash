"""Pair every original TRAIN case across frozen centered learners8 and40.

This reads completed native assessments only. It cannot change any label,
physical parameter, case, acceptance gate or pending continuation decision.
"""
from datetime import datetime,timezone
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from quadsmash.control.full_racket_train_evaluation_v4 import training_cases,summarize_training
from quadsmash.evaluation_v4 import independent_episode


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def attack(row,protocol):
    v=np.asarray(row['outgoing_velocity_mps'])
    return bool(independent_episode(row,protocol)['airborne_contact']and row['released']and
        np.isfinite(v).all()and np.linalg.norm(v)>=8.0 and v[0]>0 and v[2]<=-1.0)


def angles(a,b):
    a=a/np.linalg.norm(a,axis=1,keepdims=True);b=b/np.linalg.norm(b,axis=1,keepdims=True)
    return np.rad2deg(2*np.arccos(np.clip(np.abs(np.sum(a*b,axis=1)),0,1)))


def stats(values):return dict(minimum=float(np.min(values)),median=float(np.median(values)),maximum=float(np.max(values)))


def main():
    assert not(HERE/'REPORT.json').exists()
    reports=ROOT/'reports/G07_full_racket_v4'
    runs=[reports/'residual_ppo_centered_base8_n2048_probe8_fixed_train128_02',
        reports/'residual_ppo_centered_base8_n2048_iter40_fixed_train128_01']
    policies=[reports/'residual_ppo_centered_base8_n2048_probe8_02/policy.pt',
        reports/'residual_ppo_centered_base8_n2048_resume32_01/policy.pt']
    protocol_path=ROOT/'configs/g07_full_racket_eval_v4_r3.json';protocol=read(protocol_path);protocol['_root']=str(ROOT)
    cases,indices=training_cases(ROOT,protocol);summaries=[];all_rows=[];sources=[Path(__file__),protocol_path]
    for run,policy in zip(runs,policies):
        job=read(run/'job.json');assert job['status']=='SUCCEEDED'and job['exit_code']==0 and job['receipt_present']
        assert read(run/'completed.json')['env_steps']==102400
        inputs=read(run/'INPUTS.json');assert inputs['policy_sha256']==sha(policy)and inputs['protocol_sha256']==sha(protocol_path)
        assert inputs['reset_noise_seed']==724201 and not inputs['formal_cases_read']
        assert all(v==0 for v in inputs['reset_noise_amplitudes'].values())
        rows=read(run/'episodes.json');assert [r['source_training_index']for r in rows]==indices==list(range(128))
        assert all(row['ground_init']and row['episode_index']==0 for row in rows)
        summary=summarize_training(rows,cases,indices,protocol);metrics=read(run/'metrics.json')
        assert all(metrics[k]==v for k,v in summary.items())
        summaries.append(summary);all_rows.append(rows)
        sources += [policy,*[run/name for name in ['job.json','completed.json','INPUTS.json','episodes.json','metrics.json','contacts.json','all_first_episodes.npz','geometry.json']]]
    base=torch.load(policies[0],map_location='cpu',weights_only=False)
    centered=torch.load(policies[1],map_location='cpu',weights_only=False)
    assert centered['residual_base_mode']=='frozen_public_policy'
    assert base['residual_base_mode']=='frozen_public_policy'
    assert centered['base_controller']['residual_actor_jit']==base['base_controller']['residual_actor_jit']
    assert centered['residual_training_metadata']['base_policy_sha256']==base['residual_training_metadata']['base_policy_sha256']
    paired=[];attacks=[]
    for before,after in zip(*all_rows):
        index=before['source_training_index'];assert index==after['source_training_index']
        old=independent_episode(before,protocol);new=independent_episode(after,protocol)
        paired.append(dict(TRAIN=index,before_air=old['airborne_contact'],after_air=new['airborne_contact'],
            before_conditional_recovery=old['conditional_recovery'],after_conditional_recovery=new['conditional_recovery'],
            before_complete=old['complete'],after_complete=new['complete'],before_attack=attack(before,protocol),after_attack=attack(after,protocol),
            before_target=before['landed_target'],after_target=after['landed_target']))
        if attack(after,protocol):attacks.append(dict(TRAIN=index,independent_outcome=new,
            actual_outgoing_velocity_mps=after['outgoing_velocity_mps'],actual_speed_mps=float(np.linalg.norm(after['outgoing_velocity_mps'])),
            actual_contact_point_m=after['contact_point_m'],first_landing_m=after['first_landing_m'],
            crossed_net=after['crossed_net'],landed_target=after['landed_target'],jump=after['jump'],
            fault_flags={name:value for name,value in after.items()if name.endswith('_fault')or name=='engineering_body_crossing'}))
    assert sum(row['before_attack']for row in paired)==summaries[0]['attack_outgoing']==3
    assert sum(row['after_attack']for row in paired)==len(attacks)==summaries[1]['attack_outgoing']==4
    with (HERE/'ALL128_PAIRED_CASES.csv').open('x',newline='')as stream:
        writer=csv.DictWriter(stream,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)
    motion=[]
    with np.load(runs[0]/'all_first_episodes.npz')as before_file,np.load(runs[1]/'all_first_episodes.npz')as after_file:
        keys=['link_poses','valid_first_episode','episode_clock_s','ball_position_m']
        before={key:before_file[key]for key in keys};after={key:after_file[key]for key in keys}
        names=read(runs[0]/'geometry.json')['body_names'];assert names==read(runs[1]/'geometry.json')['body_names']
        racket=names.index('Racket');initial_error=float(np.max(np.abs(before['link_poses'][0]-after['link_poses'][0])))
        for time in [2.,2.5,2.7]:
            values=[]
            for case in range(128):
                ai=np.flatnonzero(before['valid_first_episode'][:,case]&(np.abs(before['episode_clock_s'][:,case]-time)<2e-6))
                bi=np.flatnonzero(after['valid_first_episode'][:,case]&(np.abs(after['episode_clock_s'][:,case]-time)<2e-6))
                if len(ai)==len(bi)==1:
                    a=before['link_poses'][ai[0],case];b=after['link_poses'][bi[0],case]
                    values.append(dict(TRAIN=case,root_position_difference_m=float(np.linalg.norm(a[0,:3]-b[0,:3])),
                        root_orientation_difference_degrees=float(angles(a[0,3:][None],b[0,3:][None])[0]),
                        strings_center_difference_m=float(np.linalg.norm(a[racket,:3]-b[racket,:3])),
                        racket_orientation_difference_degrees=float(angles(a[racket,3:][None],b[racket,3:][None])[0]),
                        ball_position_difference_m=float(np.linalg.norm(before['ball_position_m'][ai[0],case]-after['ball_position_m'][bi[0],case]))))
            assert values
            motion.append(dict(actual_episode_clock_s=time,paired_valid_first_episode_cases=len(values),
                summary={k:stats([v[k]for v in values])for k in values[0]if k!='TRAIN'},per_case=values))
    result=dict(created_at=datetime.now(timezone.utc).isoformat(),
        scope='Descriptive paired diagnostics on the same original128 TRAIN cases, ground initialization and zero-noise seed. Both policies were actually evaluated. No formal A/B score or isolated causal attribution to a method component.',
        centered8=summaries[0],centered40=summaries[1],
        initial_full_body_pose_max_absolute_difference=initial_error,
        measured_motion_comparison=motion,actual_centered40_attack_cases=attacks,
        paired_case_table_sha256=sha(HERE/'ALL128_PAIRED_CASES.csv'),
        source_sha256={str(p.relative_to(ROOT)):sha(p)for p in sources},
        interpretation='Complete remains0. Air count26->22 and recovery count13->10, while attacks3->4 and targets6->4. More qualifying attacks accompany reduced air/recovery counts; neither policy meets original formal readiness. The predeclared16/24/32 native comparison remains pending; two changing-policy TRAIN successes are not frozen performance.',
        formal_cases_read=False,formal_acceptance=False,goal_complete=False)
    (HERE/'REPORT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(centered8={k:summaries[0][k]for k in ['complete','airborne_contacts','conditional_recovery_count','attack_outgoing','landed_target']},
        centered40={k:summaries[1][k]for k in ['complete','airborne_contacts','conditional_recovery_count','attack_outgoing','landed_target']},
        actual_attack_cases=[{k:row[k]for k in ['TRAIN','independent_outcome','actual_speed_mps','first_landing_m','crossed_net','landed_target']}for row in attacks],
        motion=[{k:v for k,v in m.items()if k!='per_case'}for m in motion]),indent=2))


if __name__=='__main__':main()
