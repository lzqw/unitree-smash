"""Diagnose every actual contact in the completed joint8 fixed TRAIN128 run.

Rigid-body identities and snapshot command poses are geometry checks. They do
not reconstruct incident collision velocity or execute an alternative policy.
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
RUN = ROOT / 'reports/G07_full_racket_v4/residual_ppo_joint_placement_actor32_n2048_probe8_fixed_train128_01'
POLICY = ROOT / 'reports/G07_full_racket_v4/residual_ppo_joint_placement_actor32_n2048_probe8_01/policy.pt'
sys.path.insert(0, str(ROOT))
from quadsmash.control.full_racket_kinematics_v4 import FullRacketKinematicsV4
from quadsmash.physics.full_racket_v4 import load_racket


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rotation(pose):
    return Rotation.from_quat(np.asarray(pose)[[4, 5, 6, 3]]).as_matrix()


def quantiles(values):
    return np.quantile(np.asarray(values), [.1, .5, .9], axis=0).tolist()


def main():
    assert not (HERE / 'REPORT.json').exists(), 'Preserve existing evidence'
    completed = json.loads((RUN / 'completed.json').read_text())
    assert completed['env_steps'] == 102400
    inputs = json.loads((RUN / 'INPUTS.json').read_text())
    assert inputs['policy_sha256'] == sha(POLICY)
    policy = torch.load(POLICY, map_location='cpu', weights_only=False)
    assert policy['schema'] == 'full-racket-residual-ppo-v4'
    base = policy['base_controller']
    geometry = json.loads((RUN / 'geometry.json').read_text())
    body_names = geometry['body_names']
    racket_index, flange_index = body_names.index('Racket'), body_names.index('link6')
    assert body_names[0] == 'base'
    events = json.loads((RUN / 'contacts.json').read_text())
    episodes = json.loads((RUN / 'episodes.json').read_text())
    assert len(episodes) == 128 and {r['source_training_index'] for r in episodes} == set(range(128))
    assert all(e['first_racket_contact'] and e['episode_index'] == 0 for e in events)
    assert len({e['env_id'] for e in events}) == len(events) == 71
    config = load_racket()
    semiaxes = np.asarray(config['assumptions']['strings_semiaxes_yz_m'])
    kin = FullRacketKinematicsV4().double()
    rows = []
    for event in events:
        poses = np.asarray(event['body_link_poses_local'])
        root, tool, flange = poses[0], poses[racket_index], poses[flange_index]
        root_rotation, tool_rotation, flange_rotation = map(rotation, [root, tool, flange])
        q = np.asarray(event['joint_position_rad'][12:])
        command = np.asarray(event['commanded_joint_position_rad'][12:])
        with torch.inference_mode():
            values = kin(torch.tensor(np.stack([q, command]), dtype=torch.float64), return_pose=True)
        centers, normals, _, _, orientations, _, _ = [v.numpy() for v in values]
        point = np.asarray(event['point_m'])
        ball = np.asarray(event['ball_position_m'])
        point_local = tool_rotation.T @ (point-tool[:3])
        ball_local = tool_rotation.T @ (ball-tool[:3])
        command_center = root[:3]+root_rotation@centers[1]
        command_rotation = root_rotation@orientations[1]
        command_ball_local = command_rotation.T@(ball-command_center)
        fk_error = float(np.linalg.norm(root[:3]+root_rotation@centers[0]-tool[:3]))
        flange_error = float(np.linalg.norm(flange[:3]+flange_rotation@np.asarray(config['string_center_in_flange_m'])-tool[:3]))
        expected_velocity = (np.asarray(event['flange_linear_velocity_mps'])+
            np.cross(event['flange_angular_velocity_rad_s'], point-np.asarray(event['flange_position_m'])))
        velocity_error = float(np.linalg.norm(expected_velocity-event['point_velocity_mps']))
        normal_error = float(np.linalg.norm(root_rotation@normals[0]-tool_rotation[:, 0]))
        assert fk_error < 3e-5 and flange_error < 2e-5 and velocity_error < 3e-5
        assert normal_error < 1e-4
        assert not event['fixture_contact']
        row = dict(env_id=event['env_id'], source_TRAIN_index=episodes[event['env_id']]['source_training_index'],
            episode_index=event['episode_index'], time_s=event['time_s'],
            kind='strings' if event['string_contact'] else 'frame',
            actual_point_tool_m=point_local.tolist(), actual_ball_tool_m=ball_local.tolist(),
            actual_point_ellipse_rho=float(np.linalg.norm(point_local[1:]/semiaxes)),
            actual_ball_ellipse_rho=float(np.linalg.norm(ball_local[1:]/semiaxes)),
            snapshot_command_ball_tool_m=command_ball_local.tolist(),
            snapshot_command_ball_ellipse_rho=float(np.linalg.norm(command_ball_local[1:]/semiaxes)),
            snapshot_command_center_shift_in_actual_tool_m=(tool_rotation.T@(command_center-tool[:3])).tolist(),
            command_minus_actual_arm_rad=(command-q).tolist(),
            source_URDF_center_error_m=fk_error, flange_transform_error_m=flange_error,
            post_state_rigid_velocity_identity_error_mps=velocity_error,
            source_URDF_normal_error=normal_error)
        rows.append(row)
    grouped = {}
    for kind in ['strings', 'frame']:
        subset = [r for r in rows if r['kind'] == kind]
        grouped[kind] = dict(count=len(subset),
            point_tool_xyz_q10_q50_q90_m=quantiles([r['actual_point_tool_m'] for r in subset]),
            ball_tool_xyz_q10_q50_q90_m=quantiles([r['actual_ball_tool_m'] for r in subset]),
            absolute_tracking_error_q10_q50_q90_rad=quantiles([np.abs(r['command_minus_actual_arm_rad']) for r in subset]),
            snapshot_command_ball_xyz_q10_q50_q90_m=quantiles([r['snapshot_command_ball_tool_m'] for r in subset]),
            snapshot_command_center_shift_xyz_q10_q50_q90_m=quantiles([r['snapshot_command_center_shift_in_actual_tool_m'] for r in subset]),
            actual_point_rho_min_median_max=np.quantile([r['actual_point_ellipse_rho'] for r in subset], [0, .5, 1]).tolist(),
            actual_ball_rho_median=float(np.median([r['actual_ball_ellipse_rho'] for r in subset])),
            snapshot_command_ball_rho_median=float(np.median([r['snapshot_command_ball_ellipse_rho'] for r in subset])))
    assert grouped['strings']['count'] == 20 and grouped['frame']['count'] == 51
    (HERE / 'contacts.json').write_text(json.dumps(rows, indent=2, allow_nan=False)+'\n')
    with (HERE / 'contacts.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()} for row in rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(7, 7))
    theta = np.linspace(0, 2*np.pi, 400)
    axis.plot(semiaxes[0]*np.cos(theta), semiaxes[1]*np.sin(theta), 'k-', label='Declared strings ellipse')
    for kind, color in [('strings', '#0072B2'), ('frame', '#D55E00')]:
        points = np.asarray([row['actual_point_tool_m'] for row in rows if row['kind'] == kind])
        axis.scatter(points[:, 1], points[:, 2], s=30, c=color, label=f'{kind}: {len(points)} actual contacts')
    axis.scatter(*base['seed']['string_uv_m'], marker='x', s=130, c='black', label='Current teacher control point')
    axis.axhline(0, color='grey', linewidth=.5); axis.axvline(0, color='grey', linewidth=.5)
    axis.set(xlabel='Racket-local Y (m)', ylabel='Racket-local Z (m)',
        title='All 71 actual first contacts; frozen joint8 TRAIN128\nProjection on original racket plane; not incident state')
    axis.set_aspect('equal'); axis.legend(fontsize=8); figure.tight_layout()
    figure.savefig(HERE / 'contact_distribution.png', dpi=150); plt.close(figure)
    dependencies = [RUN/'completed.json', RUN/'INPUTS.json', RUN/'contacts.json', RUN/'episodes.json',
        RUN/'geometry.json', POLICY, Path(__file__).resolve(), ROOT/'quadsmash/control/full_racket_kinematics_v4.py',
        ROOT/'quadsmash/control/full_racket_geometry_v4.py', ROOT/'configs/full_racket_v4_r3.json',
        ROOT/'reports/G07_full_racket_v4/cem35_inner_strings_pose_train32_1150/metrics.json']
    report = dict(created_at=datetime.now(timezone.utc).isoformat(),
        scope='All 71 actual first racket contacts of one completed frozen TRAIN128 execution. Rigid-body identities and static command-pose geometry only; no new rollout or causal policy improvement.',
        source_run=str(RUN.relative_to(ROOT)), original_TRAIN_episodes=128, actual_contacts=len(rows),
        unchanged_string_semiaxes_yz_m=semiaxes.tolist(), teacher_string_uv_m=base['seed']['string_uv_m'],
        teacher_lateral_cartesian_gain=float(base['parameters'][18]),
        teacher_already_uses_live_FK_lateral_DLS=True, teacher_already_uses_velocity_lead_s=.06,
        grouped=grouped,
        checks=dict(max_source_URDF_string_center_error_m=max(r['source_URDF_center_error_m'] for r in rows),
            max_flange_to_string_center_error_m=max(r['flange_transform_error_m'] for r in rows),
            max_post_state_rigid_contact_velocity_error_mps=max(r['post_state_rigid_velocity_identity_error_mps'] for r in rows)),
        previous_inward_target_attempt=dict(run='cem35_inner_strings_pose_train32_1150',
            actual_candidate_TRAIN_episodes=512, airborne_strings_contacts=10, complete=0,
            conclusion='Static inward retarget was executed previously and did not establish improvement; do not repeat it as an untried method.'),
        next_hypothesis='Test bounded live ball/tool relative-state time-to-plane tangent centering, added to the existing residual actor. Preserve original geometry, all native actuators, 5rad/s target slew, and unchanged scoring. Use original TRAIN only before any formal test.',
        limitations=['Recorded contact poses/velocities are post PhysX contact states; incident velocity and collision impulse are not reconstructed.',
            'The snapshot commanded pose assumes instant joint tracking; it is not the pose that executed and does not establish a future contact outcome.',
            'The current teacher already includes FK/DLS/60ms lead. New feedback must address live interception rather than duplicate that solve.',
            'A failed earlier static retarget and this descriptive contact distribution do not prove any alternative will succeed.'],
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in dependencies},
        outputs={name: sha(HERE/name) for name in ['contacts.json', 'contacts.csv', 'contact_distribution.png']},
        formal_cases_read=False, formal_acceptance=False, goal_complete=False)
    (HERE/'REPORT.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(contacts=len(rows), grouped_counts={k:v['count'] for k,v in grouped.items()},
        teacher_uv=report['teacher_string_uv_m'], checks=report['checks'], formal_acceptance=False)), flush=True)


if __name__ == '__main__':
    main()
