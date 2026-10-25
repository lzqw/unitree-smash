"""Describe separately measured legacy and selected frozen full-racket motion.

This CPU analysis reads completed native runs, never formal full-racket cases.
Different tasks/protocols prevent a matched or causal performance claim.
"""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from quadsmash.evaluation_v4 import independent_episode
from quadsmash.control.full_racket_train_evaluation_v4 import training_cases,summarize_training


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    values=np.asarray(values,dtype=float)
    assert values.size and np.isfinite(values).all()
    return dict(count=len(values),minimum=np.min(values,axis=0).tolist(),
        median=np.median(values,axis=0).tolist(),maximum=np.max(values,axis=0).tolist())


def main():
    output=HERE/'REPORT.json';assert not output.exists(),'Preserve prior analysis'
    reports=ROOT/'reports/G07_full_racket_v4'
    run=reports/'residual_ppo_joint_placement_actor32_n2048_probe8_fixed_train128_01'
    policy=reports/'residual_ppo_joint_placement_actor32_n2048_probe8_01/policy.pt'
    protocol_path=ROOT/'configs/g07_full_racket_eval_v4_r3.json';protocol=read(protocol_path);protocol['_root']=str(ROOT)
    assert read(run/'job.json')['status']=='SUCCEEDED'
    assert read(run/'completed.json')['env_steps']==102400
    inputs=read(run/'INPUTS.json');assert inputs['policy_sha256']==sha(policy)
    assert not inputs['formal_cases_read'] and inputs['protocol_sha256']==sha(protocol_path)
    cases,indices=training_cases(ROOT,protocol);rows=read(run/'episodes.json');contacts=read(run/'contacts.json')
    summary=summarize_training(rows,cases,indices,protocol);metrics=read(run/'metrics.json')
    assert all(metrics[k]==v for k,v in summary.items())
    assert len(rows)==128 and len(contacts)==71
    outcomes={r['env_id']:independent_episode(r,protocol)for r in rows}
    selected=[];all_velocity_checks=[]
    for event in contacts:
        point=np.asarray(event['point_m']);flange=np.asarray(event['flange_position_m'])
        linear=np.asarray(event['flange_linear_velocity_mps']);angular=np.asarray(event['flange_angular_velocity_rad_s'])
        rotation=np.cross(angular,point-flange);actual=np.asarray(event['point_velocity_mps'])
        all_velocity_checks.append(dict(env_id=event['env_id'],episode_index=event['episode_index'],event_index=event['event_index'],
            reconstruction_error_mps=float(np.linalg.norm(actual-linear-rotation)),
            flange_linear_speed_mps=float(np.linalg.norm(linear)),rotational_term_speed_mps=float(np.linalg.norm(rotation)),
            actual_point_speed_mps=float(np.linalg.norm(actual))))
    for row in rows:
        if not outcomes[row['env_id']]['airborne_contact']:continue
        actual=[e for e in contacts if e['env_id']==row['env_id']and e['episode_index']==row['episode_index']and e['string_contact']and e['legal_initial_contact']]
        assert actual,'Canonical airborne label requires a matching real strings event'
        selected.append((row,min(actual,key=lambda e:e['time_s'])))
    assert len(selected)==summary['airborne_contacts']==20
    legacy_path=reports/'INTERIM_SHORTBOARD_FULL_RACKET_COMPARISON_0950.json';legacy=read(legacy_path)
    for name,digest in legacy['sources'].items():assert sha(ROOT/name)==digest,'Legacy source changed: '+name
    result=dict(created_at=datetime.now(timezone.utc).isoformat(),
        scope='Separately measured historical short-board formal panels and selected frozen full-racket TRAIN128. These differ in physical assets, rules, incoming balls and evaluation protocol; no matched/casual performance or dynamic reach ceiling is established.',
        geometry=legacy['geometry'],historical_short_board=legacy['historical_short_board'],
        current_selected_frozen_full_racket=dict(run=str(run.relative_to(ROOT)),policy_sha256=sha(policy),
            independent_TRAIN128_summary=summary,
            observed_legal_air_strings_contact_points_m=stats([e['point_m']for _,e in selected]),
            observed_legal_air_strings_point_speed_mps=stats([np.linalg.norm(e['point_velocity_mps'])for _,e in selected]),
            observed_legal_air_strings_outgoing_speed_mps=stats([np.linalg.norm(r['outgoing_velocity_mps'])for r,_ in selected]),
            observed_legal_air_strings_continuous_recovery_window_s=stats([r['jump']['recovery_s']for r,_ in selected]),
            contact_position_note='Actual legal first strings contacts in the canonical airborne subset. Observed contact height is not a maximum dynamically reachable head height.'),
        actual_rigid_contact_point_velocity_check=dict(formula='v_hit = v_flange + omega_flange cross (p_hit - p_flange)',
            actual_contact_events=len(all_velocity_checks),
            reconstruction_error_mps=stats([r['reconstruction_error_mps']for r in all_velocity_checks]),
            rotational_term_speed_mps=stats([r['rotational_term_speed_mps']for r in all_velocity_checks]),
            event_results=all_velocity_checks,scope='Read-only recomputation of actual saved native point, flange velocity and angular velocity; no threshold or physical parameter changed.'),
        limits=['Historical and full-racket rates cannot be treated as a controlled improvement/regression caused by geometry alone.',
            'The current selected full-racket policy still has0/128 complete successes; the later centered learner is currently training and contributes no score here.',
            'Frozen original full-racket A200/B200 acceptance and matched dynamic reach measurements remain unproven.'],
        source_sha256={str(p.relative_to(ROOT)):sha(p)for p in [Path(__file__).resolve(),legacy_path,policy,protocol_path,
            run/'job.json',run/'completed.json',run/'INPUTS.json',run/'episodes.json',run/'contacts.json',run/'metrics.json']},
        formal_cases_read=False,formal_acceptance=False,goal_complete=False)
    temporary=output.with_suffix('.json.tmp');temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.replace(output)
    print(json.dumps(dict(report=str(output),actual_contacts=len(contacts),canonical_air_strings=len(selected),
        contact_height_m=result['current_selected_frozen_full_racket']['observed_legal_air_strings_contact_points_m'],
        point_speed=result['current_selected_frozen_full_racket']['observed_legal_air_strings_point_speed_mps'],
        actual_rigid_velocity_error=result['actual_rigid_contact_point_velocity_check']['reconstruction_error_mps'])))


if __name__=='__main__':main()
