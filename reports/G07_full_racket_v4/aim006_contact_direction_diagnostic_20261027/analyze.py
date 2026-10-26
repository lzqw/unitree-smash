"""Measure the real Aim006 funnel, face error and precontact flight consistency.

No robot/ball state is assigned. Native contact metadata contains post-impact
velocities. A restitution comparison using that racket velocity is explicitly
not a pre-impact collision-law test or a demonstrated executable target.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import solve_ivp
import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from quadsmash.control.full_racket_body_search_v4 import summarize_candidate
from quadsmash.control.full_racket_contact_aim_v4 import ContactAimControllerV4
from quadsmash.evaluation_v4 import independent_episode


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    assert not any((HERE/name).exists() for name in ['PLAN.json','RESULT.json'])
    run = ROOT/'reports/G07_full_racket_v4/contact_aim7_sequential_n128_01'
    owner = ROOT/'reports/G07_full_racket_v4/prepose_branch_after_aim_queue01'
    held = read(owner/'PLAN.json')['source_sha256']
    assert all(sha(Path(path)) == digest for path, digest in held.items())
    source_path = ROOT/'reports/G07_full_racket_v4/body_strike_grid16_n2048_01/best_candidate.pt'
    physics = read(ROOT/'configs/shuttle_physics_v2.json')
    protocol = read(ROOT/'configs/g07_full_racket_eval_v4_r3.json'); protocol['_root'] = str(ROOT)
    rows, contacts = read(run/'episodes_002.json'), read(run/'contacts_002.json')
    initial, plans = read(run/'initial_ground_states_002.json'), read(run/'causal_plans_002.json')
    entry = next(item for item in read(run/'metrics.json')['history'] if item['candidate_id'] == 2)
    summary = summarize_candidate(rows, protocol)
    assert summary == entry['summary'] and entry['env_steps'] == 102400
    assert len(rows) == 128 and summary['complete'] == 5 and summary['numerical_failures'] == 0
    for lane, row in enumerate(rows):
        assert row['env_id'] == lane and row['episode_index'] == initial['episode_ids'][lane]
    paths = [run/name for name in ['episodes_002.json','contacts_002.json','initial_ground_states_002.json',
                                  'causal_plans_002.json','all_first_episodes_002.npz','INPUTS.json']]
    paths += [Path(__file__), source_path, ROOT/'configs/shuttle_physics_v2.json',
              ROOT/'configs/g07_full_racket_eval_v4_r3.json', ROOT/'configs/court_net_v4.json',
              ROOT/'quadsmash/control/full_racket_body_search_v4.py', ROOT/'quadsmash/control/initial_intercept.py']
    atomic(HERE/'PLAN.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
        source_sha256={str(path):sha(path) for path in paths},
        first_episode_filter='Preserve original global env_id and actual native episode_index2.',
        calibration_filter='First actual string contact and no prior body-ball fault. All128 rows remain in original scores.',
        incoming_forecast='RK4-equivalent DOP853 using only previously recorded public causal plan initial ball state and locked drag/gravity.',
        precontact_position_diagnostic_bound_m=.002, solver_rtol=1e-10, solver_atol=1e-11,
        caveat='Contact point and ball velocities in metadata are post-impact. An apparent normal mass is a diagnostic surrogate with articulated/tangent/actuator coupling, not an identified physical constant.',
        new_native_execution=False, formal_cases_read=False, formal_acceptance=False))
    source = torch.load(source_path, map_location='cpu', weights_only=False)
    teacher = ContactAimControllerV4(source, [0., .06], 1, 'cpu').reference.teacher
    reference = teacher.normal_world.numpy(); angle = float(source['parameters'][37])
    desired = np.array([np.cos(angle)*reference[0]+np.sin(angle)*reference[2], reference[1],
                        -np.sin(angle)*reference[0]+np.cos(angle)*reference[2]])
    desired *= teacher.face_sign; desired /= np.linalg.norm(desired)
    forecast = {}
    for plan in plans:
        for index, lane in enumerate(plan['env_ids']):
            if lane not in forecast:
                forecast[lane] = dict(clock_s=plan['observation_clock_s'][index],
                    p=plan['initial_position_m'][index], v=plan['initial_velocity_mps'][index])
    wind, drag = np.asarray(physics['wind_m_s']), physics['quadratic_drag_k_per_m']

    def dynamics(_, state):
        v = state[3:]; relative = v-wind
        return np.r_[v, [0.,0.,-9.81]-drag*np.linalg.norm(relative)*relative]

    measurements = []
    with np.load(run/'all_first_episodes_002.npz') as arrays:
        for row in rows:
            lane = row['env_id']; checked = independent_episode(row, protocol)
            velocity = np.asarray(row['outgoing_velocity_mps'])
            attack = bool(checked['airborne_contact'] and row['released'] and np.isfinite(velocity).all()
                          and np.linalg.norm(velocity) >= 8. and velocity[0] > 0 and velocity[2] <= -1.)
            legal_cross = bool(attack and row['crossed_net'] and not checked['fault'])
            first = [event for event in contacts if event['env_id'] == lane
                     and event['episode_index'] == row['episode_index'] and event['first_racket_contact']]
            assert len(first) <= 1
            event = first[0] if first else None
            measurement = dict(TRAIN_index=lane, actual_native_episode_index=row['episode_index'],
                airborne_contact=checked['airborne_contact'], conditional_recovery=checked['conditional_recovery'],
                complete=checked['complete'], attack=attack, attack_legal_cross=legal_cross,
                attack_legal_cross_target=bool(legal_cross and row['landed_target']),
                body_landing=row['jump']['body_landing'], shuttle_net_contact=row['shuttle_net_contact'],
                first_landing_m=row['first_landing_m'] if row['ground_seen'] else None,
                actual_outgoing_velocity_mps=row['outgoing_velocity_mps'] if row['released'] else None)
            if event:
                normal = np.asarray(event['face_normal']); normal /= np.linalg.norm(normal)
                measurement.update(actual_string_contact=event['string_contact'],
                    actual_face_normal_world=normal.tolist(),
                    actual_face_pitch_deg=float(np.rad2deg(np.arctan2(normal[2],np.linalg.norm(normal[:2])))),
                    actual_face_yaw_deg=float(np.rad2deg(np.arctan2(normal[1],normal[0]))),
                    desired_vs_actual_face_angle_deg=float(np.rad2deg(np.arccos(np.clip(normal@desired,-1.,1.)))),
                    actual_post_contact_point_velocity_mps=event['point_velocity_mps'])
                if event['string_contact'] and lane in forecast and not row['body_ball_fault']:
                    f = forecast[lane]; end = event['time_s']-f['clock_s']
                    assert 0. < end < 1.5
                    fit = solve_ivp(dynamics, [0., end], np.r_[f['p'],f['v']], dense_output=True,
                                    rtol=1e-10,atol=1e-11,max_step=.002,method='DOP853')
                    assert fit.success
                    clock = arrays['episode_clock_s'][:,lane]
                    keep = arrays['valid_first_episode'][:,lane] & (clock >= f['clock_s']) & (clock <= event['time_s']-.002)
                    times = clock[keep]-f['clock_s']; assert len(times)
                    predicted = fit.sol(times)[:3].T
                    errors = np.linalg.norm(predicted-arrays['ball_position_m'][keep,lane],axis=1)
                    state = fit.y[:,-1]; vin = state[3:]; vout = np.asarray(event['ball_velocity_mps'])
                    point = np.asarray(event['point_velocity_mps']); e = physics['restitution']
                    a, b = float((vout-point)@normal), float((vin-point)@normal)
                    ratio = (a+e*b)/(e*(a-b)) if abs(a-b)>1e-10 else None
                    apparent_mass = physics['mass_kg']/ratio if ratio is not None and ratio > 0 else None
                    measurement.update(precontact_position_samples=len(times),
                        maximum_precontact_forecast_position_error_m=float(errors.max()),
                        precontact_forecast_within_diagnostic_bound=bool(errors.max() <= .002),
                        predicted_incoming_velocity_mps=vin.tolist(),
                        post_contact_ball_velocity_mps=vout.tolist(),
                        apparent_normal_effective_mass_kg=apparent_mass,
                        apparent_mass_identified_physical_constant=False,
                        infinite_mass_normal_residual_using_POST_point_velocity_mps=float(a+e*b))
            measurements.append(measurement)
    funnel = dict(all_original_cases=128,air=sum(item['airborne_contact'] for item in measurements),
        attack=sum(item['attack'] for item in measurements),
        attack_legal_cross=sum(item['attack_legal_cross'] for item in measurements),
        attack_legal_cross_target=sum(item['attack_legal_cross_target'] for item in measurements),
        complete=sum(item['complete'] for item in measurements))
    assert funnel == dict(all_original_cases=128,air=60,attack=21,attack_legal_cross=19,attack_legal_cross_target=7,complete=5)
    target_recovery_failures = [item['TRAIN_index'] for item in measurements if item['attack_legal_cross_target'] and not item['complete']]
    outside = [item['TRAIN_index'] for item in measurements if item['attack_legal_cross'] and not item['attack_legal_cross_target']]
    calibration = [item for item in measurements if 'maximum_precontact_forecast_position_error_m' in item]
    mass = [item['apparent_normal_effective_mass_kg'] for item in calibration if item['precontact_forecast_within_diagnostic_bound']
            and item['apparent_normal_effective_mass_kg'] is not None]
    attack_rows = [item for item in measurements if item['attack']]
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(),candidate_original128_summary=summary,
        funnel=funnel, attack_legal_cross_outside_target_TRAIN_indices=outside,
        target_chain_recovery_failure_TRAIN_indices=target_recovery_failures,
        source_requested_normal_world=desired.tolist(),
        attack_contacts_face_angle_error_median_deg=float(np.median([item['desired_vs_actual_face_angle_deg'] for item in attack_rows])),
        attack_contacts_face_yaw_median_deg=float(np.median([item['actual_face_yaw_deg'] for item in attack_rows])),
        attack_contacts_face_pitch_median_deg=float(np.median([item['actual_face_pitch_deg'] for item in attack_rows])),
        attack_contacts_post_point_vy_median_mps=float(np.median([item['actual_post_contact_point_velocity_mps'][1] for item in attack_rows])),
        precontact_flight_calibration_cases=len(calibration),
        precontact_flight_calibration_within_2mm=sum(item['precontact_forecast_within_diagnostic_bound'] for item in calibration),
        maximum_precontact_forecast_position_error_m=max(item['maximum_precontact_forecast_position_error_m'] for item in calibration),
        diagnostic_apparent_normal_mass_median_kg=float(np.median(mass)) if mass else None,
        measurements=measurements,
        interpretation='Actual face direction and lateral point motion are measured error sources. A constant post-velocity impact model is not an identified robot model; future directional targets require real native tracking and cold assessment.',
        unchanged_original_target_and_all128_scores=True, all429_held_inputs_unchanged=True,
        new_native_execution=False,formal_cases_read=False,formal_acceptance=False)
    assert all(sha(Path(path)) == digest for path,digest in read(HERE/'PLAN.json')['source_sha256'].items())
    assert all(sha(Path(path)) == digest for path,digest in held.items())
    atomic(HERE/'RESULT.json',result)
    print(json.dumps({key:value for key,value in result.items() if key not in ['measurements','interpretation']}),flush=True)


if __name__ == '__main__':
    main()
