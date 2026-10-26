"""Posthoc sensitivity at actual earlier50Hz ball positions; no robot replay."""
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


def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,data):
    assert not path.exists()
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def main():
    original = read(HERE/'RESULT.json'); rows = read(RUN/'episodes.json')
    sources = [Path(__file__),ROOT/'quadsmash/control/release_flight_envelope_v4.py',HERE/'RESULT.json',
        HERE/'ACTUAL_NATIVE_FLIGHT_VALIDATION.json',RUN/'all_first_episodes.npz',RUN/'geometry.json',RUN/'episodes.json',
        ROOT/'configs/court_net_v4.json',ROOT/'configs/shuttle_physics_v2.json']
    fingerprints = {str(p):sha(p) for p in sources}
    save(HERE/'EARLIER_POINT_PLAN.json',dict(created_at=datetime.now(timezone.utc).isoformat(),
        declared_after_original_release_grid=True,source_sha256=fingerprints,
        fixed_hypothetical_speed_mps=8.01,pitch_grid_degrees=[-45.,-5.,.1],dt_s=.0005,
        actual_earlier_point_selection='Last valid actual50Hz ball sample at least5ms before first contact; then the preceding50Hz sample. No interpolation or shifted ball coordinates.',
        pose_comparison='Distance from recorded earlier ball to recorded earlier racket front center only; no inverse kinematics or robot reachability claim.',
        scope='Counterfactual outgoing-flight planning at actually recorded earlier ball positions. These were NOT earlier robot contacts or new native trials.',
        new_native_execution=False,formal_cases_read=False,formal_acceptance=False,goal_complete=False))
    physics,court = read(ROOT/'configs/shuttle_physics_v2.json'),read(ROOT/'configs/court_net_v4.json')
    base = original['proposals']; cases = [p['TRAIN_index'] for p in base]
    p = np.asarray([x['actual_ball_position_m'] for x in base])
    v = np.asarray([x['actual_outgoing_velocity_mps'] for x in base])
    native_shadow = free_flight(p,v,physics,court,dt=.0000625)
    crossing_errors = []
    for i,index in enumerate(cases):
        row = rows[index]
        if np.isfinite(row['crossing_point_m']).all() and not row['shuttle_net_contact'] and row['contact_count'] == 1:
            error = native_shadow.trailing_crossing_m[i]-row['crossing_point_m']
            crossing_errors.append(dict(TRAIN_index=index,xyz_error_m=error.tolist(),error_norm_m=float(np.linalg.norm(error))))
    sets = [p]; sample_details = []
    geometry = read(RUN/'geometry.json'); body_names = geometry['body_names']
    racket_id = body_names.index('Racket')
    with np.load(RUN/'all_first_episodes.npz') as arrays:
        first,second = [],[]
        for index in cases:
            row = rows[index]
            valid = np.flatnonzero(arrays['valid_first_episode'][:,index].astype(bool)
                & (arrays['episode_clock_s'][:,index] <= row['contact_time_s']-.005)
                & (arrays['episode_clock_s'][:,index] >= 2.2))
            assert len(valid) >= 2
            entry = dict(TRAIN_index=index,actual_first_contact_time_s=row['contact_time_s'],earlier_samples=[])
            for which,sample in enumerate(valid[-2:][::-1]):
                point = arrays['ball_position_m'][sample,index].astype(float)
                pose = arrays['link_poses'][sample,index,racket_id].astype(float)
                w,q = pose[3],pose[4:];offset = np.array([.0256,0.,0.])
                front = pose[:3]+offset+2*np.cross(q,np.cross(q,offset)+w*offset)
                clock = float(arrays['episode_clock_s'][sample,index])
                entry['earlier_samples'].append(dict(actual_time_s=clock,earlier_by_s=row['contact_time_s']-clock,
                    actual_ball_position_m=point.tolist(),actual_robot_racket_front_center_m=front.tolist(),
                    actual_ball_to_current_racket_front_center_distance_m=float(np.linalg.norm(point-front))))
                (first if which == 0 else second).append(point)
            sample_details.append(entry)
        sets += [np.asarray(first),np.asarray(second)]
    variants = []
    pitch = np.deg2rad(np.arange(-45.,-4.999,.1))
    for name,positions in zip(['actual_contact_position','actual_earlier50Hz_ball_position','actual_one_more50Hz_earlier_ball_position'],sets):
        yaw = np.arctan2(-positions[:,1],3.-positions[:,0]);speed = 8.01
        candidate = np.stack([speed*np.cos(pitch)[None]*np.cos(yaw)[:,None],
            speed*np.cos(pitch)[None]*np.sin(yaw)[:,None],speed*np.sin(pitch)[None]*np.ones((len(positions),1))],axis=2)
        starts = np.repeat(positions[:,None,:],len(pitch),axis=1)
        flight = free_flight(starts.reshape(-1,3),candidate.reshape(-1,3),physics,court,dt=.0005)
        feasible = target_screen(flight,candidate.reshape(-1,3),court).reshape(len(positions),len(pitch))
        variants.append(dict(name=name,CPU_geometric_witness_count=int(feasible.any(1).sum()),
            per_case=[dict(TRAIN_index=index,sampled_feasible_pitch_count=int(feasible[i].sum()),
                sampled_pitch_extrema_deg=([float(np.rad2deg(pitch[np.flatnonzero(feasible[i])[0]])),
                    float(np.rad2deg(pitch[np.flatnonzero(feasible[i])[-1]]))] if feasible[i].any() else None))
                for i,index in enumerate(cases)]))
    validation = read(HERE/'ACTUAL_NATIVE_FLIGHT_VALIDATION.json')['comparisons']
    landing_error = np.asarray([np.asarray(x['predicted_first_landing_m'])-x['actual_first_landing_m'] for x in validation])
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(),source_sha256=fingerprints,
        native_free_flight_landing_max_abs_xyz_error_m=np.abs(landing_error).max(0).tolist(),
        native_trailing_crossing_errors=crossing_errors,
        native_trailing_crossing_max_abs_xyz_error_m=np.abs([x['xyz_error_m'] for x in crossing_errors]).max(0).tolist(),
        original_norm_preserving_or_min8_grid_witnesses=original['CPU_grid_with_conservative_target_witness'],
        fixed8_01_sensitivity_variants=variants,actual_earlier_ball_and_racket_samples=sample_details,
        native_complete_still0_of128=True,earlier_robot_contacts_not_executed=True,
        robot_dynamics_reachability_impact_and_recovery_unproven=True,
        new_native_execution=False,formal_cases_read=False,formal_acceptance=False,goal_complete=False)
    assert all(sha(Path(path)) == digest for path,digest in fingerprints.items())
    save(HERE/'EARLIER_POINT_RESULT.json',result)
    print(json.dumps(dict(variants=[dict(name=x['name'],witnesses=x['CPU_geometric_witness_count']) for x in variants],
        landing_max_abs_xyz_error_m=result['native_free_flight_landing_max_abs_xyz_error_m'],
        crossing_max_abs_xyz_error_m=result['native_trailing_crossing_max_abs_xyz_error_m'])),flush=True)


if __name__ == '__main__': main()
