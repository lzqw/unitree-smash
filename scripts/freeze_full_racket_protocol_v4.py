#!/usr/bin/env python3
"""Freeze court-scale incoming cases before any v4 policy training results."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from quadsmash.physics.court_v4 import load_court,top_height


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def flight(position,velocity,time,coefficient,steps=900,record=False):
    p,v=position.copy(),velocity.copy()
    dt=np.asarray(time)[:,None]/steps
    gravity=np.array([0.,0.,-9.81])
    acceleration=lambda u:gravity-coefficient*np.linalg.norm(u,axis=-1,keepdims=True)*u
    history=[p.copy()] if record else None
    for _ in range(steps):
        a=acceleration(v);b=acceleration(v+dt*a/2);c=acceleration(v+dt*b/2);d=acceleration(v+dt*c)
        p+=dt*(v+2*(v+dt*a/2)+2*(v+dt*b/2)+(v+dt*c))/6
        v+=dt*(a+2*b+2*c+d)/6
        if record:history.append(p.copy())
    return (p,v,np.stack(history)) if record else (p,v)


def incoming_cases(count,seed,panel):
    rng=np.random.default_rng(seed)
    b=panel=='B'
    starts=np.column_stack((rng.uniform(6.,7.8,count),rng.uniform(-1. if b else -.8,1. if b else .8,count),rng.uniform(1.3,2.,count)))
    times=rng.uniform(.64 if b else .69,.86 if b else .81,count)
    target=np.column_stack((np.full(count,1.10),rng.uniform(-.20 if b else -.10,.20 if b else .10,count),rng.uniform(1.70 if b else 1.76,1.90 if b else 1.84,count)))
    coefficient=json.loads((ROOT/'configs/shuttle_physics_v2.json').read_text())['quadratic_drag_k_per_m']
    initial=(target-starts)/times[:,None]
    initial[:,2]+=.5*9.81*times
    for _ in range(6):
        predicted,_=flight(starts,initial,times,coefficient)
        error=target-predicted
        if np.linalg.norm(error,axis=1).max()<1e-9:break
        jacobian=np.empty((count,3,3))
        for axis in range(3):
            perturbed=initial.copy();perturbed[:,axis]+=1e-4
            position,_=flight(starts,perturbed,times,coefficient)
            jacobian[:,:,axis]=(position-predicted)/1e-4
        initial+=np.linalg.solve(jacobian,error[:,:,None])[:,:,0]
    endpoint,incoming,history=flight(starts,initial,times,coefficient,steps=1800,record=True)
    errors=np.linalg.norm(endpoint-target,axis=1)
    assert errors.max()<1e-6
    crossing=(history[:-1,:,0]>=1.5)&(history[1:,:,0]<1.5)
    assert (crossing.sum(0)==1).all()
    index=crossing.argmax(0);lane=np.arange(count)
    before,after=history[index,lane],history[index+1,lane]
    fraction=(1.5-before[:,0])/(after[:,0]-before[:,0])
    point=before+fraction[:,None]*(after-before)
    margins=point[:,2]-.025-top_height(point[:,1],load_court())
    assert margins.min()>.05, 'Incoming distribution itself must clear the regulation net'
    assert history[:,:,2].min()>.5
    assert np.linalg.norm(initial,axis=1).max()<20
    rows=[]
    for i in range(count):
        rows.append(dict(case_id=f'{panel}-{seed}-{i:03d}',panel=panel,
            launch=dict(position_m=starts[i].tolist(),velocity_mps=initial[i].tolist()),
            prediction_only=dict(intercept_plane_x_m=1.10,intercept_position_m=target[i].tolist(),
                flight_time_s=float(times[i]),incoming_velocity_mps=incoming[i].tolist(),
                incoming_net_margin_m=float(margins[i]),inverse_flight_error_m=float(errors[i])),
            actor_information='No case ID or predicted future label. Only current/history proprioception, ball observations after active launch, generic known command and clock.'))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Frozen output must be new')
    config_path=ROOT/'configs/g07_full_racket_eval_v4.json'
    if config_path.exists():raise ValueError('Protocol config already exists; preserve it')
    # Case generation is numerical initial-condition design only; no robot,
    # candidate policy or observed success/failure is used to select cases.
    panels={'A':incoming_cases(200,724001,'A'),'B':incoming_cases(200,724002,'B')}
    train=incoming_cases(64,720401,'A')+incoming_cases(64,720402,'B')
    args.output.mkdir(parents=True)
    suite_paths={}
    for name,rows in {**panels,'training':train}.items():
        path=args.output/f'{name}_cases.json'
        path.write_text(json.dumps(dict(cases=rows),indent=2)+'\n')
        suite_paths[name]=dict(path=str(path),sha256=digest(path),count=len(rows))
    asset=ROOT/'assets/quadsmash_full_racket_v4.usd'
    config=dict(version='full-racket-singles-rally-v4-frozen-1',
        frozen_at=datetime.now(timezone.utc).isoformat(),before_v4_policy_training=True,
        formal_suites={k:v for k,v in suite_paths.items() if k!='training'},
        training_suite=suite_paths['training'],
        asset=dict(path=str(asset.relative_to(ROOT)),sha256=digest(asset)),
        physical_configs={name:digest(ROOT/name) for name in ['configs/full_racket_v4.json','configs/court_net_v4.json','configs/shuttle_physics_v2.json','configs/evaluation_protocol_v2.json']},
        physics_dt_s=.0000625,policy_dt_s=.01,episode_seconds=8.,preparation_seconds=2.,
        initialization=dict(root_position_m=[0.,0.,.323],ground_only=True,
            reset_seeds={'A':724101,'B':724102},
            perturbations={'A':dict(joint_position_rad=.01,joint_velocity_rad_s=.02,root_xy_m=.02,root_yaw_rad=.03),
                           'B':dict(joint_position_rad=.025,joint_velocity_rad_s=.04,root_xy_m=.04,root_yaw_rad=.08)},
            distribution='Independent uniform [-amplitude,+amplitude], joint positions clipped only to actual joint limits; no root height or external-force aid.'),
        launches='Exactly one active initial condition at t=2 s after explicit preparation. No post-launch ball state correction. Ball observations masked before active launch.',
        generic_command=dict(position_m=[1.10,0.,1.80],normal=[1.,0.,0.],time_s=2.75,velocity_mps=[0.,0.,0.]),
        aggregation='Compute all ratios separately on each complete 200-case panel; use the worse panel for each threshold; retain every case and every numerical fault.',
        thresholds=dict(complete_success_rate=.50,airborne_contact_rate=.70,conditional_recovery_rate=.85,numerical_failures=0),
        metrics=dict(airborne_contact='Actual string force during measured flight and legal first own-side racket contact; later faults remain recorded.',
            conditional_recovery='Among airborne-contact episodes, original qualified-jump/continuous0.8s recovery and no rally-invalidating fault.',
            complete_success='Ground initialization AND legal airborne strings AND outgoing speed>=8 m/s,vx>0,vz<=-1 m/s AND actual over-net return AND first landing in target AND original recovery AND no fault.',
            separate_counts=['actual_racket_contacts','strings','airborne','legal_initial_contact','attack','crossed_net','landed_legal_court','landed_target','recovery','complete','each_fault']),
        law_source=dict(version='BWF Laws of Badminton In Force 26 April 2025 V5.0(1)',
            url='https://badminton.dk/wp-content/uploads/2025/11/Section-4.1-Laws-of-Badminton-26-April-2025-V5.0-1-.pdf',
            future_2027_laws_applied=False),
        engineering_constraints=dict(nonracket_stays_own_half=True,body_proxy='Conservative body-local bounds of actual collider shapes; fixture is nonracket.',
            head_strings_same_stroke='Continuous contact episodes; frame+strings exception grouped within40ms, normal change<15deg and no head-velocity reversal. Engineering resolution, not BWF wording.',
            net='64 GPU primitive strips, suspended band; rigid equivalent without cloth holes.',
            shuttle='Existing 5.2g sphere/radius25mm/uncalibrated quadratic drag; real-shuttle validation remains outside these simulation claims.'),
        policy_lock='Freeze source SHA plus dirty snapshot, all physical hashes and weight SHA before launching formal panels; cases/thresholds never adjusted based on results.',
        no_old_passes_inherited=True,formal_results_available=False)
    config_path.write_text(json.dumps(config,indent=2)+'\n')
    lock=dict(protocol=str(config_path.relative_to(ROOT)),sha256=digest(config_path),suites=suite_paths,
        selection='All generated cases retained. No v4 robot policy results used.',formal_acceptance=False)
    (args.output/'FROZEN.json').write_text(json.dumps(lock,indent=2)+'\n')
    all_rows=panels['A']+panels['B']+train
    summary=dict(cases=len(all_rows),formal_cases=400,training_cases=128,
        initial_speed_range_mps=[min(np.linalg.norm(r['launch']['velocity_mps']) for r in all_rows),max(np.linalg.norm(r['launch']['velocity_mps']) for r in all_rows)],
        incoming_net_margin_min_m=min(r['prediction_only']['incoming_net_margin_m'] for r in all_rows),
        inverse_flight_error_max_m=max(r['prediction_only']['inverse_flight_error_m'] for r in all_rows),
        formal_acceptance=False,policy_training=False)
    (args.output/'generation_validation.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
