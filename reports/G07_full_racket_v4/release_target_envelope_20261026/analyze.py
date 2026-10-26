"""Analyze actual leg06 frozen releases and CPU-only alternative directions."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
MAIN = ROOT.parent/'unitree_smash'
HERE = Path(__file__).resolve().parent
RUN = MAIN/'reports/G07_full_racket_v4/residual_ppo_centered_leg06_after075_intermediates_n2048_probe8_fixed_train128_01'
sys.path.insert(0,str(ROOT))
from quadsmash.control.release_flight_envelope_v4 import free_flight,target_screen
from quadsmash.evaluation_v4 import independent_episode


def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def atomic(path,value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n'); temporary.replace(path)


def main():
    assert not any((HERE/name).exists() for name in ['PLAN.json','RESULT.json','FAILURE.json'])
    job = read(RUN/'job.json'); assert job['status'] == 'SUCCEEDED' and job['charged_env_steps'] == 102400
    assert read(RUN/'INPUTS.json')['reset_noise_seed'] == 724201
    assert read(RUN/'metrics.json')['noise_level'] == 'none'
    protocol = read(ROOT/'configs/g07_full_racket_eval_v4_r3.json'); protocol['_root'] = str(ROOT)
    physics,court = read(ROOT/'configs/shuttle_physics_v2.json'),read(ROOT/'configs/court_net_v4.json')
    rows,contacts = read(RUN/'episodes.json'),read(RUN/'contacts.json')
    air = [row for row in rows if independent_episode(row,protocol)['airborne_contact']]
    assert len(rows) == 128 and len(air) == 27 and sum(independent_episode(row,protocol)['complete'] for row in rows) == 0
    measured = []
    for row in air:
        events = [event for event in contacts if event['env_id'] == row['env_id'] and event['episode_index'] == 0 and event['first_racket_contact']]
        assert len(events) == 1 and row['released']
        event = events[0]
        assert np.array_equal(event['ball_velocity_mps'],row['outgoing_velocity_mps'])
        measured.append(event)
    sources = [Path(__file__),ROOT/'quadsmash/control/release_flight_envelope_v4.py',
        ROOT/'quadsmash/physics/court_v4.py',ROOT/'configs/shuttle_physics_v2.json',ROOT/'configs/court_net_v4.json',
        ROOT/'configs/g07_full_racket_eval_v4_r3.json',RUN/'episodes.json',RUN/'contacts.json',RUN/'job.json',RUN/'INPUTS.json',RUN/'metrics.json']
    main_plan_path = MAIN/'reports/G07_full_racket_v4/centered_leg06_full_cohort_warmup13_queue01/PLAN.json'
    main_bound = {str(MAIN/name):digest for name,digest in read(main_plan_path)['source_hashes'].items()}
    main_bound[str(main_plan_path)] = sha(main_plan_path)
    assert all(sha(Path(name)) == digest for name,digest in main_bound.items())
    plan = dict(created_at=datetime.now(timezone.utc).isoformat(),source_sha256={str(p):sha(p) for p in sources},
        source_all128_native_complete=0,actual_air_releases=27,actual_first_event_and_last_touch_velocity_equal=True,
        coarse_dt_s=.0005,refined_dt_s=.0000625,native_free_landing_validation_max_error_m=.02,
        minimum_native_free_landings_for_validation=5,refined_selected_landing_max_difference_m=.0001,
        pitch_grid_degrees=[-45.,-5.,.1],yaw='Aim horizontally at original target center(3,0)',
        speed='Keep actual norm if>=8m/s; otherwise hypothetical8m/s. Not proof this output can be generated.',
        selection='Middle feasible grid pitch per point after finite-radius net/target/attack planning screen; no case deletion or changed acceptance.',
        native_reference_filter='Free-flight comparison only: one contact, recorded first ground, no shuttle-net/body-ball interaction. All cases remain in original evaluation.',
        net_screen='Actual64 suspended boxes and posts, sphere distance with half-step speed buffer; avoids net touch as a conservative planning subset. Does not forbid legal tape-touch returns in native rules.',
        scope='CPU release flight/geometry diagnostic; no racket impact solution, reachable robot trajectory, policy performance, or recovery proof.',
        main302_sources_unchanged=True,new_native_execution=False,formal_cases_read=False,formal_acceptance=False,goal_complete=False)
    atomic(HERE/'PLAN.json',plan)
    p = np.asarray([event['ball_position_m'] for event in measured]); v = np.asarray([event['ball_velocity_mps'] for event in measured])
    baseline = free_flight(p,v,physics,court,dt=plan['coarse_dt_s'])
    comparisons,excluded = [],[]
    for i,row in enumerate(air):
        eligible = (row['ground_seen'] and np.isfinite(row['first_landing_m']).all()
            and row['contact_count'] == 1 and not row['shuttle_net_contact'] and not row['body_ball_fault'])
        if eligible:
            delta = baseline.landing_m[i]-row['first_landing_m']
            comparisons.append(dict(TRAIN_index=row['source_training_index'],actual_first_landing_m=row['first_landing_m'],
                predicted_first_landing_m=baseline.landing_m[i].tolist(),error_m=float(np.linalg.norm(delta)),
                elapsed_time_error_s=float(baseline.landing_time_s[i]-(row['first_landing_time_s']-row['contact_time_s']))))
        else:
            excluded.append(dict(TRAIN_index=row['source_training_index'],ground_seen=row['ground_seen'],
                contact_count=row['contact_count'],shuttle_net_contact=row['shuttle_net_contact'],body_ball_fault=row['body_ball_fault']))
    validation_passed = (len(comparisons) >= plan['minimum_native_free_landings_for_validation']
        and max((x['error_m'] for x in comparisons),default=float('inf')) <= plan['native_free_landing_validation_max_error_m'])
    atomic(HERE/'ACTUAL_NATIVE_FLIGHT_VALIDATION.json',dict(checked_at=datetime.now(timezone.utc).isoformat(),
        comparisons=comparisons,excluded_from_free_flight_validation_only=excluded,passed=validation_passed,
        native_scores_unchanged=True,new_native_execution=False,formal_acceptance=False))
    if not validation_passed:
        atomic(HERE/'FAILURE.json',dict(reason='Predeclared actual free-flight validation gate failed; do not use output envelope',new_native_execution=False))
        raise RuntimeError('Free-flight replay validation failed; preserve evidence')
    pitch = np.deg2rad(np.arange(-45.,-4.999,.1))
    speed = np.maximum(np.linalg.norm(v,axis=1),8.)
    yaw = np.arctan2(-p[:,1],3.-p[:,0])
    candidate_v = np.stack([speed[:,None]*np.cos(pitch)[None]*np.cos(yaw)[:,None],
        speed[:,None]*np.cos(pitch)[None]*np.sin(yaw)[:,None],
        speed[:,None]*np.sin(pitch)[None]*np.ones((len(p),1))],axis=2)
    candidate_p = np.repeat(p[:,None,:],len(pitch),axis=1)
    coarse = free_flight(candidate_p.reshape(-1,3),candidate_v.reshape(-1,3),physics,court,dt=plan['coarse_dt_s'])
    feasible = target_screen(coarse,candidate_v.reshape(-1,3),court).reshape(len(p),len(pitch))
    chosen,proposals = [],[]
    for i,row in enumerate(air):
        valid = np.flatnonzero(feasible[i])
        entry = dict(TRAIN_index=row['source_training_index'],actual_ball_position_m=p[i].tolist(),
            actual_outgoing_velocity_mps=v[i].tolist(),actual_speed_mps=float(np.linalg.norm(v[i])),
            actual_pitch_deg=float(np.rad2deg(np.arctan2(v[i,2],np.linalg.norm(v[i,:2])))),
            hypothetical_speed_increase=bool(np.linalg.norm(v[i]) < 8.),sampled_feasible_pitch_count=len(valid))
        if len(valid):
            index = int(valid[len(valid)//2]); flat = i*len(pitch)+index
            chosen.append((i,index,flat))
            entry.update(sampled_pitch_extrema_deg=[float(np.rad2deg(pitch[valid[0]])),float(np.rad2deg(pitch[valid[-1]]))],
                proposed_pitch_deg=float(np.rad2deg(pitch[index])),proposed_velocity_mps=candidate_v[i,index].tolist(),
                proposed_landing_m=coarse.landing_m[flat].tolist(),
                coarse_buffered_net_gap_m=float(coarse.buffered_net_gap_m[flat]),
                required_velocity_change_norm_mps=float(np.linalg.norm(candidate_v[i,index]-v[i])))
        proposals.append(entry)
    if chosen:
        fine = free_flight(np.asarray([p[i] for i,_,_ in chosen]),
            np.asarray([candidate_v[i,index] for i,index,_ in chosen]),physics,court,dt=plan['refined_dt_s'])
        fine_pass = target_screen(fine,np.asarray([candidate_v[i,index] for i,index,_ in chosen]),court)
        for number,(i,index,flat) in enumerate(chosen):
            difference = float(np.linalg.norm(fine.landing_m[number]-coarse.landing_m[flat]))
            proposals[i].update(refined_landing_m=fine.landing_m[number].tolist(),
                refined_buffered_net_gap_m=float(fine.buffered_net_gap_m[number]),
                coarse_refined_landing_difference_m=difference,refined_screen_passed=bool(fine_pass[number]))
            assert difference <= plan['refined_selected_landing_max_difference_m'] and fine_pass[number]
    assert all(sha(Path(name)) == digest for name,digest in plan['source_sha256'].items())
    assert all(sha(Path(name)) == digest for name,digest in main_bound.items())
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(),actual_air_releases=len(air),
        native_validation_cases=len(comparisons),maximum_native_landing_error_m=max(x['error_m'] for x in comparisons),
        actual_native_complete_still0_of128=True,CPU_grid_with_conservative_target_witness=len(chosen),proposals=proposals,
        unresolved_robot_impact_reachability_and_recovery=True,main302_sources_unchanged=True,
        scope=plan['scope'],new_native_execution=False,formal_cases_read=False,formal_acceptance=False,goal_complete=False)
    atomic(HERE/'RESULT.json',result)
    print(json.dumps({k:result[k] for k in ['actual_air_releases','native_validation_cases','maximum_native_landing_error_m',
        'CPU_grid_with_conservative_target_witness']}),flush=True)


if __name__ == '__main__': main()
