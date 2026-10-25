"""Measure precontact forecasts on real fixed-policy TRAIN trajectories.

Exact 100Hz public velocities exist for two cases. The other contact cases
use causal backward differences of recorded50Hz world poses; those derivative
estimates are explicitly separate and are checked against the exact subset.
No candidate motion is executed and no future label enters the predictor.
"""
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RUN = ROOT/'reports/G07_full_racket_v4/residual_ppo_joint_placement_actor32_n2048_probe8_fixed_train128_01'
sys.path[:0] = [str(ROOT), str(ROOT/'reports/G07_full_racket_v4/live_tangent_trial_20261025')]
from reactive import PublicTangentCorrection
from quadsmash.control.full_racket_kinematics_v4 import FullRacketKinematicsV4


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rotation(quaternion):
    shape = quaternion.shape
    value = quaternion.reshape(-1, 4)[:, [1, 2, 3, 0]]
    return Rotation.from_quat(value).as_matrix().reshape(*shape[:-1], 3, 3)


def statistics(rows):
    result = {}
    for lower, upper in [(0., .04), (.04, .08), (.08, .12), (.12, .16), (.16, .20)]:
        group = [row for row in rows if lower < row['actual_time_to_first_contact_s'] <= upper]
        if not group:
            continue
        result[f'{lower:.2f}..{upper:.2f}s'] = dict(samples=len(group),
            distinct_episodes=len({row['env_id'] for row in group}),
            timing_absolute_error_q10_q50_q90_ms=np.quantile([abs(row['timing_error_ms']) for row in group], [.1, .5, .9]).tolist(),
            local_plane_error_q10_q50_q90_m=np.quantile([row['local_plane_vector_error_m'] for row in group], [.1, .5, .9]).tolist(),
            negative_local_direction_dot=sum(row['predicted_actual_local_direction_dot'] < 0. for row in group))
    return result


def make_row(env_id, time, tau, predicted_local, actual_local, event, source):
    lead = event['time_s']-time
    return dict(env_id=int(env_id), source_TRAIN_index=int(env_id), source=source,
        contact_kind='strings' if event['string_contact'] else 'frame',
        observation_clock_s=float(time), actual_first_contact_s=event['time_s'],
        actual_time_to_first_contact_s=float(lead), predicted_time_to_nominal_plane_s=float(tau),
        timing_error_ms=float((tau-lead)*1000),
        predicted_local_yz_m=np.asarray(predicted_local).tolist(),
        actual_post_contact_local_yz_m=np.asarray(actual_local).tolist(),
        local_plane_vector_error_m=float(np.linalg.norm(predicted_local-actual_local)),
        predicted_actual_local_direction_dot=float(np.dot(predicted_local, actual_local)))


def main():
    assert not (HERE/'REPORT.json').exists()
    assert json.loads((RUN/'completed.json').read_text())['env_steps'] == 102400
    events = json.loads((RUN/'contacts.json').read_text())
    assert len(events) == 71 and all(event['first_racket_contact'] for event in events)
    names = json.loads((RUN/'geometry.json').read_text())['body_names']
    racket = names.index('Racket')
    data = np.load(RUN/'all_first_episodes.npz')
    pose, ball, clock, valid = [data[key] for key in ['link_poses', 'ball_position_m', 'episode_clock_s', 'valid_first_episode']]
    root_rotation = rotation(pose[:, :, 0, 3:])
    tool_rotation = rotation(pose[:, :, racket, 3:])
    normal = tool_rotation[:, :, :, 0]
    relative = ball-pose[:, :, racket, :3]
    dt = np.diff(clock, axis=0).astype(float)
    # Derivatives only use the present and previous saved sample.
    dr_world = np.zeros_like(relative, dtype=float)
    dr_world[1:] = np.diff(relative, axis=0)/dt[:, :, None]
    root_omega = np.zeros_like(relative, dtype=float)
    tool_omega = np.zeros_like(relative, dtype=float)
    for rotations, output in [(root_rotation, root_omega), (tool_rotation, tool_omega)]:
        changes = rotations[1:]@rotations[:-1].swapaxes(-1, -2)
        output[1:] = Rotation.from_matrix(changes.reshape(-1, 3, 3)).as_rotvec().reshape(*changes.shape[:2], 3)/dt[:, :, None]
    distance = np.sum(normal*relative, axis=2)-.0256
    closing = np.sum(normal*dr_world, axis=2)+np.sum(np.cross(tool_omega, normal)*relative, axis=2)
    tau = distance/np.maximum(-closing, .25)
    body_relative_derivative_world = dr_world-np.cross(root_omega, relative)
    prediction = relative+body_relative_derivative_world*np.clip(tau, 0., .2)[:, :, None]
    prediction -= normal*np.sum(normal*prediction, axis=2, keepdims=True)
    prediction *= .14/np.maximum(np.linalg.norm(prediction, axis=2, keepdims=True), .14)
    local_prediction = np.einsum('tnji,tnj->tni', tool_rotation, prediction)
    by_id = {event['env_id']: event for event in events}
    actual_local = {}
    approximate_rows = []
    for env_id, event in by_id.items():
        event_pose = np.asarray(event['body_link_poses_local'][racket])
        actual_local[env_id] = rotation(event_pose[3:]).T@(np.asarray(event['ball_position_m'])-event_pose[:3])
        lead = event['time_s']-clock[:, env_id]
        keep = (valid[:, env_id] & (lead > 0.) & (lead <= .2) & (distance[:, env_id] > 0.)
            & (closing[:, env_id] < -.25) & (tau[:, env_id] > 0.) & (tau[:, env_id] < .2))
        keep[0] = False
        for tick in np.flatnonzero(keep):
            approximate_rows.append(make_row(env_id, clock[tick, env_id], tau[tick, env_id],
                local_prediction[tick, env_id, 1:], actual_local[env_id][1:], event,
                'Causal backward differences of actual50Hz poses; approximate velocities'))
    public_data = np.load(RUN/'public_and_actions_first2.npz')
    public = torch.tensor(public_data['observations'])
    defaults = json.loads((ROOT/'configs/full_racket_reset_v4.json').read_text())['arm_joint_positions_rad']
    correction = PublicTangentCorrection(defaults, .7)
    kin = FullRacketKinematicsV4().float()
    with torch.inference_mode():
        _, exact_tau, gate, error, exact_distance, exact_closing = correction.diagnostics(public.reshape(-1, 794))
        _, _, _, _, body_tool, _, _ = kin(torch.tensor(defaults)+public[:, :, 21:27].reshape(-1, 6), return_pose=True)
        local = torch.einsum('nji,nj->ni', body_tool, error).numpy().reshape(-1, 2, 3)
    exact_tau = exact_tau.numpy().reshape(-1, 2)
    exact_closing = exact_closing.numpy().reshape(-1, 2)
    exact_distance = exact_distance.numpy().reshape(-1, 2)
    gate = gate.numpy().reshape(-1, 2)
    exact_rows, derivative_checks = [], []
    for env_id in [0, 1]:
        event = by_id[env_id]
        for tick in np.flatnonzero(gate[:, env_id] & public_data['valid_first_episode'][:, env_id]):
            time = float(public[tick, env_id, 775])
            if time >= event['time_s']:
                continue
            exact_rows.append(make_row(env_id, time, exact_tau[tick, env_id], local[tick, env_id, 1:],
                actual_local[env_id][1:], event, 'Exact actual100Hz public positions and velocities'))
            if tick % 2 == 0:
                pose_tick = tick//2
                assert abs(clock[pose_tick, env_id]-time) < 2e-6
                assert abs(distance[pose_tick, env_id]-exact_distance[tick, env_id]) < 3e-5
                derivative_checks.append(dict(env_id=env_id, clock_s=time,
                    distance_error_m=float(distance[pose_tick, env_id]-exact_distance[tick, env_id]),
                    closing_rate_error_mps=float(closing[pose_tick, env_id]-exact_closing[tick, env_id]),
                    TTC_error_s=float(tau[pose_tick, env_id]-exact_tau[tick, env_id])))
    assert len(exact_rows) == 35 and approximate_rows
    all_rows = [*exact_rows, *approximate_rows]
    (HERE/'predictions.json').write_text(json.dumps(all_rows, indent=2, allow_nan=False)+'\n')
    with (HERE/'predictions.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]), lineterminator='\n');writer.writeheader()
        writer.writerows({key:json.dumps(value) if isinstance(value, list) else value for key,value in row.items()} for row in all_rows)
    approximate_stats = {kind:statistics([row for row in approximate_rows if row['contact_kind']==kind]) for kind in ['strings', 'frame']}
    exact_stats = statistics(exact_rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for kind, color in [('strings', '#0072B2'), ('frame', '#D55E00')]:
        rows = [row for row in approximate_rows if row['contact_kind']==kind]
        x = [row['actual_time_to_first_contact_s']*1000 for row in rows]
        axes[0].scatter(x, [row['timing_error_ms'] for row in rows], s=8, alpha=.35, color=color, label=kind)
        axes[1].scatter(x, [row['local_plane_vector_error_m']*100 for row in rows], s=8, alpha=.35, color=color)
    axes[0].axhline(0, color='black', linewidth=.7);axes[0].legend()
    axes[0].set(xlabel='Actual lead to first contact (ms)', ylabel='Predicted plane time minus contact time (ms)')
    axes[1].set(xlabel='Actual lead to first contact (ms)', ylabel='Local-plane prediction residual (cm)')
    figure.suptitle('Fixed joint8 actual contacts; causal50Hz derivative estimates, not new executions')
    figure.tight_layout();figure.savefig(HERE/'forecast_error.png', dpi=150);plt.close(figure)
    report = dict(created_at=datetime.now(timezone.utc).isoformat(),
        scope='Offline forecast residuals on a completed frozen TRAIN128 execution. The 71 contacted cases condition this analysis;57 no-contact cases have no physical contact-time label and are not counted as accurate predictions.',
        source_full_TRAIN_episodes=128, actual_first_contacts=71,
        exact_public_subset=dict(episodes=[0, 1], both_contacts_are_frames=True, rows=len(exact_rows), statistics=exact_stats),
        approximate_50Hz_contact_subset=dict(rows=len(approximate_rows), distinct_episodes=len({row['env_id'] for row in approximate_rows}),
            contacted_episodes_without_eligible_forecast=sorted(set(by_id)-{row['env_id'] for row in approximate_rows}), statistics=approximate_stats),
        derivative_validation_on_exact_public_subset=dict(rows=len(derivative_checks),
            absolute_closing_rate_error_q10_q50_q90_mps=np.quantile([abs(row['closing_rate_error_mps']) for row in derivative_checks], [.1,.5,.9]).tolist(),
            absolute_TTC_error_q10_q50_q90_ms=(1000*np.quantile([abs(row['TTC_error_s']) for row in derivative_checks], [.1,.5,.9])).tolist(), raw_checks=derivative_checks),
        next_action='Keep both already predeclared200ms native trials and all their failures. Use their exact128-case100Hz public/action logs to assess prediction and tracking after actual execution. If long-horizon error persists, test a separately versioned shorter horizon or observed-acceleration forecast; do not silently modify pinned proposals.',
        limitations=['Exact public diagnostic contains only two frame contacts; it is not representative of128 cases.',
            '50Hz backward differences are causal numerical estimates, not recorded native instantaneous velocities. Their measured errors are reported separately.',
            'Forecast predicts a nominal flat strings plane; actual frame contacts have different collision surfaces.',
            'Local Y/Z residual compares a current frozen racket basis with the actual future racket basis. Racket rotation and trajectory curvature contribute; it is not an isolated ball-prediction error.',
            'Actual contact samples are post PhysX states; no incident collision velocity is inferred.',
            'Offline prediction error does not prove any corrective controller succeeds, fails, or improves physical performance. Original scoring is unchanged.'],
        source_sha256={str(path.relative_to(ROOT)):sha(path) for path in [Path(__file__).resolve(),
            RUN/'contacts.json', RUN/'all_first_episodes.npz', RUN/'public_and_actions_first2.npz',
            ROOT/'reports/G07_full_racket_v4/live_tangent_trial_20261025/reactive.py']},
        outputs={name:sha(HERE/name) for name in ['predictions.json','predictions.csv','forecast_error.png']},
        native_job_started=False, formal_cases_read=False, formal_acceptance=False, goal_complete=False)
    (HERE/'REPORT.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(exact_rows=len(exact_rows), approximate_rows=len(approximate_rows),
        exact_stats=exact_stats, approximate_stats=approximate_stats,
        derivative_TTC_q10_q50_q90_ms=report['derivative_validation_on_exact_public_subset']['absolute_TTC_error_q10_q50_q90_ms'])), flush=True)


if __name__ == '__main__':
    main()
