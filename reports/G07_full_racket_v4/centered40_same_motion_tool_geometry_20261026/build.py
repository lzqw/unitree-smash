"""Compare two rigid tool reference points on one actually executed motion.

This is a CPU kinematic counterfactual. It never executes the old-mass robot,
predicts an old-tool contact, changes native scores or claims a reach ceiling.
"""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from quadsmash.control.full_racket_train_evaluation_v4 import training_cases, summarize_training
from quadsmash.evaluation_v4 import independent_episode


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rotate(quaternion, vector):
    quaternion = np.asarray(quaternion,dtype=float)
    quaternion = quaternion/np.linalg.norm(quaternion,axis=-1,keepdims=True)
    v = np.broadcast_to(np.asarray(vector,dtype=float),quaternion.shape[:-1]+(3,))
    xyz = quaternion[...,1:]
    return v+2*np.cross(xyz,np.cross(xyz,v)+quaternion[...,:1]*v)


def stats(values):
    a = np.asarray(values,dtype=float)
    assert a.size and np.isfinite(a).all()
    return dict(count=len(a),minimum=np.min(a,axis=0).tolist(),median=np.median(a,axis=0).tolist(),maximum=np.max(a,axis=0).tolist())


def write_csv(path, rows):
    with path.open('x',newline='') as file:
        writer = csv.DictWriter(file,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def main():
    outputs = [HERE/name for name in ['REPORT.json','ALL128_OBSERVED_POSE_PEAKS.csv','ALL_ACTUAL_CONTACT_POINTS.csv']]
    assert not any(p.exists() for p in outputs)
    reports = ROOT/'reports/G07_full_racket_v4'
    run = reports/'residual_ppo_centered_base8_n2048_iter40_fixed_train128_01'
    policy = reports/'residual_ppo_centered_base8_n2048_resume32_01/policy.pt'
    protocol_path = ROOT/'configs/g07_full_racket_eval_v4_r3.json'
    protocol = read(protocol_path);protocol['_root'] = str(ROOT)
    old = read(ROOT/'configs/assembly_v2.json')['racket']
    new = read(ROOT/'configs/full_racket_v4_r3.json')
    old_offset = np.asarray(old['blade_center_relative_to_link6_m'])
    new_offset = np.asarray(new['string_center_in_flange_m'])
    assert read(run/'job.json')['status'] == 'SUCCEEDED' and read(run/'completed.json')['env_steps'] == 102400
    assert read(run/'INPUTS.json')['policy_sha256'] == sha(policy)
    cases,indices = training_cases(ROOT,protocol)
    rows,events = read(run/'episodes.json'),read(run/'contacts.json')
    summary = summarize_training(rows,cases,indices,protocol)
    assert all(read(run/'metrics.json')[key] == value for key,value in summary.items())
    names = read(run/'geometry.json')['body_names']
    flange_index,racket_index = names.index('link6'),names.index('Racket')
    with np.load(run/'all_first_episodes.npz',allow_pickle=False) as a:
        poses,clocks,valid = a['link_poses'].astype(float),a['episode_clock_s'],a['valid_first_episode'].astype(bool)
    assert poses.shape == (400,128,28,7) and valid.shape == (400,128) and valid.any(axis=0).all()
    flange = poses[:,:,flange_index]
    actual_new = poses[:,:,racket_index,:3]
    reconstructed_new = flange[:,:,:3]+rotate(flange[:,:,3:],new_offset)
    hypothetical_old = flange[:,:,:3]+rotate(flange[:,:,3:],old_offset)
    pose_errors = np.linalg.norm(reconstructed_new-actual_new,axis=-1)[valid]
    assert np.max(pose_errors) <= 5e-5
    paired = []
    for row in rows:
        lane = row['env_id'];ticks = np.flatnonzero(valid[:,lane])
        new_tick = ticks[np.argmax(actual_new[ticks,lane,2])]
        old_tick = ticks[np.argmax(hypothetical_old[ticks,lane,2])]
        paired.append(dict(TRAIN=row['source_training_index'],valid_50Hz_samples=len(ticks),
            observed_full_racket_center_peak_z_m=float(actual_new[new_tick,lane,2]),
            old_center_z_on_same_new_peak_pose_m=float(hypothetical_old[new_tick,lane,2]),
            same_pose_center_height_difference_m=float(actual_new[new_tick,lane,2]-hypothetical_old[new_tick,lane,2]),
            new_peak_observed_clock_s=float(clocks[new_tick,lane]),
            hypothetical_old_center_observed_peak_z_m=float(hypothetical_old[old_tick,lane,2]),
            old_counterfactual_peak_observed_clock_s=float(clocks[old_tick,lane]),
            original_complete_success=independent_episode(row,protocol)['complete']))
    actual_points,by_event = [],{}
    for event in events:
        body = np.asarray(event['body_link_poses_local'],dtype=float)
        p = np.asarray(event['flange_position_m']);v = np.asarray(event['flange_linear_velocity_mps'])
        omega = np.asarray(event['flange_angular_velocity_rad_s'])
        q = body[flange_index,3:]
        old_lever,new_lever = rotate(q,old_offset),rotate(q,new_offset)
        old_point,new_point = p+old_lever,p+new_lever
        old_velocity,new_velocity = v+np.cross(omega,old_lever),v+np.cross(omega,new_lever)
        hit = np.asarray(event['point_m']);actual_velocity = np.asarray(event['point_velocity_mps'])
        error = float(np.linalg.norm(v+np.cross(omega,hit-p)-actual_velocity))
        center_error = float(np.linalg.norm(new_point-body[racket_index,:3]))
        assert error <= 1e-4 and center_error <= 5e-5
        record = dict(TRAIN=event['env_id'],episode_index=event['episode_index'],event_index=event['event_index'],
            actual_contact_time_s=event['time_s'],actual_string_contact=event['string_contact'],
            actual_legal_initial_contact=event['legal_initial_contact'],actual_contact_x_m=float(hit[0]),actual_contact_z_m=float(hit[2]),
            new_center_z_m=float(new_point[2]),old_center_z_same_pose_m=float(old_point[2]),
            same_pose_center_height_difference_m=float(new_point[2]-old_point[2]),
            new_center_speed_mps=float(np.linalg.norm(new_velocity)),old_center_speed_same_twist_mps=float(np.linalg.norm(old_velocity)),
            same_twist_center_speed_difference_mps=float(np.linalg.norm(new_velocity)-np.linalg.norm(old_velocity)),
            actual_contact_point_speed_mps=float(np.linalg.norm(actual_velocity)),
            actual_point_velocity_formula_error_mps=error,new_center_native_pose_error_m=center_error)
        actual_points.append(record);by_event[(event['env_id'],event['episode_index'],event['event_index'])] = record
    air = []
    for row in rows:
        if not independent_episode(row,protocol)['airborne_contact']:continue
        matches = [e for e in events if e['env_id'] == row['env_id'] and e['episode_index'] == row['episode_index']
                   and e['string_contact'] and e['legal_initial_contact']]
        assert matches
        event = min(matches,key=lambda e:e['time_s'])
        air.append(by_event[(event['env_id'],event['episode_index'],event['event_index'])])
    assert len(air) == summary['airborne_contacts']
    sources = [Path(__file__),protocol_path,policy,ROOT/'configs/assembly_v2.json',ROOT/'configs/full_racket_v4_r3.json']
    sources += [run/name for name in ['job.json','completed.json','INPUTS.json','episodes.json','contacts.json','metrics.json','geometry.json','all_first_episodes.npz']]
    result = dict(created_at=datetime.now(timezone.utc).isoformat(),
        scope='CPU same-executed-motion rigid geometry comparison on completed frozen40 originalTRAIN128. Hypothetical old point does not execute the old-mass robot or create a contact.',
        native_source_policy_sha256=sha(policy),independent_original_TRAIN128=summary,
        old_flange_to_center_m=old_offset.tolist(),new_flange_to_center_m=new_offset.tolist(),
        original_short_board_mass_kg=old['mass_kg'],new_racket_and_fixture_mass_kg=sum(new['assumptions'][key] for key in ['unstrung_mass_kg','strings_mass_kg','added_grip_mass_kg','clamp_mass_kg']),
        observed_50Hz_poses_checked=int(valid.sum()),reconstructed_full_racket_native_center_error_m=stats(pose_errors),
        all128_observed_new_center_peak_z_m=stats([p['observed_full_racket_center_peak_z_m'] for p in paired]),
        old_center_height_on_each_same_new_peak_pose_m=stats([p['old_center_z_on_same_new_peak_pose_m'] for p in paired]),
        same_pose_height_gain_at_each_observed_new_peak_m=stats([p['same_pose_center_height_difference_m'] for p in paired]),
        all_actual_contact_events=len(events),canonical_legal_air_events=len(air),
        canonical_air_actual_contact_positions_m=stats([[p['actual_contact_x_m'],p['actual_contact_z_m']] for p in air]),
        canonical_air_same_pose_center_height_gain_m=stats([p['same_pose_center_height_difference_m'] for p in air]),
        canonical_air_same_twist_center_speed_gain_mps=stats([p['same_twist_center_speed_difference_mps'] for p in air]),
        canonical_air_actual_contact_point_speed_mps=stats([p['actual_contact_point_speed_mps'] for p in air]),
        all_contact_actual_point_velocity_formula_error_mps=stats([p['actual_point_velocity_formula_error_mps'] for p in actual_points]),
        formula='v_point = v_flange + omega_flange cross (p_point - p_flange)',
        limits=['Observed50Hz peaks can miss continuous-time maxima and are not dynamically maximal reachable heights.',
                'Old point placement on new-racket motion is a kinematic counterfactual; old inertia, collisions, control response, contact, legality and recovery are not re-simulated.',
                'String center speed and actual off-center contact point speed are different quantities; neither equals outgoing shuttle speed.',
                'Current frozen40 remains0/128 complete,22 airborne and10 conditional recoveries. New075 training supplies no score here.',
                'Historical short-board comparisons remain separately scoped in current_legacy_comparison_20261026/REPORT.json; no causal old/new performance or formal full-racket acceptance is claimed.'],
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},new_native_execution=False,
        formal_cases_read=False,formal_acceptance=False,goal_complete=False)
    write_csv(outputs[1],paired);write_csv(outputs[2],actual_points)
    with outputs[0].open('x') as file:json.dump(result,file,indent=2);file.write('\n')
    print(json.dumps({key:result[key] for key in ['observed_50Hz_poses_checked','reconstructed_full_racket_native_center_error_m',
        'same_pose_height_gain_at_each_observed_new_peak_m','canonical_air_same_pose_center_height_gain_m',
        'canonical_air_same_twist_center_speed_gain_mps','all_contact_actual_point_velocity_formula_error_mps']},indent=2))


if __name__ == '__main__':
    main()
