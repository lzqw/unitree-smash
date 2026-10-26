#!/usr/bin/env python3
"""Build shared preparation proposals from completed native TRAIN evidence.

This is a CPU initializer, never a new dynamics or success measurement. Source
selection and the independent cold assessment are separate from these seeds.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import lsq_linear, least_squares
from scipy.spatial.transform import Rotation
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quadsmash.control.full_racket_prepose_branch_v4 import PreposeBranchControllerV4, validate_overrides
from quadsmash.control.full_racket_geometry_v4 import FullRacketGeometryV4
from quadsmash.control.full_racket_train_evaluation_v4 import training_cases, summarize_training
from quadsmash.evaluation_v4 import independent_episode


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source_path, evaluation, output, static_bank=None):
    assert not output.exists(), 'Use a fresh proposal path'
    read = lambda name: json.loads((evaluation/name).read_text())
    protocol_path = ROOT/'configs/g07_full_racket_eval_v4_r3.json'
    protocol = json.loads(protocol_path.read_text()); protocol['_root'] = str(ROOT)
    cases, indices = training_cases(ROOT, protocol)
    assert indices == list(range(128))
    job, inputs, metrics, rows = read('job.json'), read('INPUTS.json'), read('metrics.json'), read('episodes.json')
    assert job['status'] == 'SUCCEEDED' and job['charged_env_steps'] == read('completed.json')['env_steps']
    assert metrics['policy_sha256'] == inputs['policy_sha256'] == sha(source_path)
    assert metrics['protocol_sha256'] == inputs['protocol_sha256'] == sha(protocol_path)
    assert metrics['noise_level'] == 'none' and inputs['training_indices'] == indices
    summary = summarize_training(rows, cases, indices, protocol)
    assert all(metrics[key] == value for key, value in summary.items()) and summary['numerical_failures'] == 0
    source = torch.load(source_path, map_location='cpu', weights_only=False)
    assert source['native_execution'] and source['asset_sha256'] == protocol['asset']['sha256']
    assert sha(ROOT/protocol['asset']['path']) == protocol['asset']['sha256']
    assert all(sha(ROOT/name) == digest for name, digest in protocol['physical_configs'].items())
    model = FullRacketGeometryV4()
    controller = PreposeBranchControllerV4(source, {}, 2, 'cpu')
    uv = controller.reference.teacher.contact_offset[1:].numpy().copy()
    contacts = read('contacts.json')
    eligible = []
    for row in rows:
        if not independent_episode(row, protocol)['complete']:
            continue
        events = [event for event in contacts if event['env_id'] == row['env_id']
                  and event['episode_index'] == row['episode_index'] and event['first_racket_contact']
                  and event['string_contact'] and event['legal_initial_contact']]
        for event in events:
            q = np.asarray(event['joint_position_rad'][12:])
            mirror = q.copy()
            mirror[3:] = [(q[3]-np.pi+np.pi) % (2*np.pi)-np.pi,
                          -q[4], (q[5]-np.pi+np.pi) % (2*np.pi)-np.pi]
            if not (np.all(q >= model.bounds[:, 0]) and np.all(q <= model.bounds[:, 1])):
                continue
            target_center, target_rotation = model.forward(q, base_position=(0., 0., 0.))
            bounds = model.bounds.copy()
            bounds[4, 1] = -.05  # Choose the other wrist branch, within original physical limits.
            raw_mirror = mirror.copy()
            def pose_error(value):
                point, rotation = model.forward(value, base_position=(0., 0., 0.))
                return np.r_[point-target_center, .25*Rotation.from_matrix(target_rotation.T@rotation).as_rotvec()]
            initial_mirror = np.clip(mirror, bounds[:, 0]+1e-8, bounds[:, 1]-1e-8)
            fit = least_squares(pose_error, initial_mirror, bounds=(bounds[:, 0], bounds[:, 1]),
                                max_nfev=600, xtol=1e-12, ftol=1e-12, gtol=1e-12, x_scale='jac')
            mirror = fit.x
            center, rotation = model.forward(mirror, base_position=(0., 0., 0.))
            position_error = float(np.linalg.norm(center-target_center))
            rotation_error = float(np.rad2deg(Rotation.from_matrix(target_rotation.T@rotation).magnitude()))
            seed_check = dict(raw_mirror_joint_position_rad=raw_mirror.tolist(),
                minimum_raw_mirror_joint_bound_margin_rad=float(np.minimum(raw_mirror-model.bounds[:,0], model.bounds[:,1]-raw_mirror).min()),
                bounded_pose_solver_success=bool(fit.success), position_error_m=position_error, rotation_error_deg=rotation_error,
                original_physical_joint_bounds_retained=True, exact_equivalence_claimed=False,
                diagnostic_seed_position_bound_m=.005, diagnostic_seed_rotation_bound_deg=3., native_execution=False)
            if fit.success and position_error <= .005 and rotation_error <= 3.:
                eligible.append((row['jump']['recovery_s'], row, event, q, mirror, seed_check))
    assert eligible, 'No complete TRAIN contact with a bounded near-equivalent wrist seed; preserve this finding'
    _, row, event, q, mirror, branch_seed_check = max(eligible, key=lambda item: item[0])
    poses = np.asarray(event['body_link_poses_local'])
    root_rotation = Rotation.from_quat(poses[0, [4, 5, 6, 3]]).as_matrix()
    root_kwargs = dict(base_position=poses[0, :3], base_rpy=Rotation.from_matrix(root_rotation).as_euler('xyz'))
    center, rotation = model.forward(q, **root_kwargs)
    mirrored_center, mirrored_rotation = model.forward(mirror, **root_kwargs)
    actual_rotation = Rotation.from_quat(poses[-1, [4, 5, 6, 3]]).as_matrix()
    fk_error = float(np.linalg.norm(center-poses[-1, :3]))
    branch_position_error = float(np.linalg.norm(mirrored_center-center))
    branch_rotation_error = float(abs(mirrored_rotation-rotation).max())
    assert fk_error < 1e-4 and abs(rotation-actual_rotation).max() < 1e-4
    assert branch_position_error <= .005 and branch_seed_check['rotation_error_deg'] <= 3.
    velocity = (source['parameters'][26:32]*source['parameters'][7]).numpy().astype(float)
    assert abs(velocity).max() <= 5.
    mirror_velocity = velocity.copy(); mirror_velocity[4] *= -1
    gravity_body = root_rotation.T @ np.array([0., 0., -1.])
    projected_pitch = float(np.arctan2(gravity_body[0], -gravity_body[2]))
    solvers = []

    def solve_velocity(hit_q, desired, advance=0.):
        # Physical string surface, including its complete flange lever arm.
        local = np.r_[controller.reference.teacher.face_sign*model.half_extents[0], uv]
        kwargs = dict(root_kwargs, base_position=poses[0, :3]+[advance, 0., 0.])
        p, r = model.forward(hit_q, **kwargs); hit = p+r@local
        columns = []
        for axis in np.eye(6):
            ph, rh = model.forward(hit_q+axis*1e-6, **kwargs)
            pl, rl = model.forward(hit_q-axis*1e-6, **kwargs)
            columns.append((ph+rh@local-pl-rl@local)/2e-6)
        jacobian = np.stack(columns, axis=1)
        root_com = np.asarray(event['root_com_position_m'])+[advance, 0., 0.]
        body = np.asarray(event['root_com_linear_velocity_mps'])+np.cross(event['root_angular_velocity_rad_s'], hit-root_com)
        fit = lsq_linear(np.r_[jacobian, .04*np.eye(6)], np.r_[np.asarray(desired)-body, np.zeros(6)],
                         bounds=(-5., 5.), tol=1e-10, max_iter=300)
        assert fit.success and np.isfinite(fit.x).all() and abs(fit.x).max() <= 5.
        solvers.append(dict(hit_joint_position_rad=np.asarray(hit_q).tolist(), desired_point_velocity_world_mps=desired,
            hypothetical_walk_advance_m=advance, joint_velocity_seed_rad_s=fit.x.tolist(),
            native_source_body_twist_contribution_mps=body.tolist(), predicted_point_velocity_mps=(body+jacobian@fit.x).tolist(),
            new_native_execution=False, physical_velocity_tracking_proven=False))
        return fit.x.tolist()

    candidates = [dict(name='unchanged_measured_source', overrides={}),
        dict(name='measured_complete_contact_pose', overrides=dict(hit_joint_position_rad=q.tolist())),
        dict(name='bounded_near_equivalent_wrist_branch', overrides=dict(hit_joint_position_rad=mirror.tolist(),
            swing_joint_velocity_rad_s=mirror_velocity.tolist()))]
    for desired, advance in [([.8, 0., -.3], 0.), ([.8, 0., -1.2], 0.),
                             ([.8, 0., -.3], .1), ([.8, 0., -1.2], .1)]:
        candidates.append(dict(name=f'wrist_branch_vz{desired[2]}_walk{advance}', overrides=dict(
            hit_joint_position_rad=mirror.tolist(), swing_joint_velocity_rad_s=solve_velocity(mirror, desired, advance),
            walk_waypoint_delta_m=advance, desired_normal_world=[1., 0., 0.],
            desired_point_velocity_world_mps=desired, nominal_projected_pitch_rad=projected_pitch)))
    static_receipt = None
    if static_bank is not None:
        bank = json.loads(static_bank.read_text())
        found = [item for item in bank['candidates'] if item['pose_search_found']]
        assert found and not bank['native_execution']
        item = min(found, key=lambda entry: entry['target_rate_only_lower_bound_from_ground_s'])
        hit_q = item['joint_position_rad']; advance = item['requested_walk_advance_m']; desired = [.8, 0., -.3]
        candidates.append(dict(name='static_full_pose_seed_shared_all_cases', overrides=dict(
            hit_joint_position_rad=hit_q, swing_joint_velocity_rad_s=solve_velocity(hit_q, desired, advance),
            walk_waypoint_delta_m=advance, intercept_plane_x_m=item['ball_plane_x_m'],
            desired_normal_world=item['desired_normal_world'], desired_point_velocity_world_mps=desired,
            nominal_projected_pitch_rad=projected_pitch)))
        static_receipt = dict(path=str(static_bank), sha256=sha(static_bank), candidate_id=item['candidate_id'],
                              static_search_aim_uv_m=[0., .06], current_source_aim_uv_m=uv.tolist(),
                              exact_pose_not_claimed_for_changed_source_or_body_state=True)
    assert len(candidates) in (7, 8)
    for i, candidate in enumerate(candidates):
        candidate.update(candidate_id=i, overrides=validate_overrides(candidate['overrides']))
    result = dict(created_at=datetime.now(timezone.utc).isoformat(),
        source_policy=str(source_path), source_policy_sha256=sha(source_path), source_evaluation=str(evaluation),
        source_evaluation_sha256={name:sha(evaluation/name) for name in
            ['INPUTS.json','metrics.json','episodes.json','contacts.json','job.json','completed.json','public_and_actions_first2.npz']},
        source_original128_summary=summary, source_aim_uv_m=uv.tolist(),
        selected_TRAIN_contact=dict(env_id=event['env_id'],episode_index=event['episode_index'],
            event_index=event['event_index'],time_s=event['time_s'],
            choice='Longest measured stable recovery among complete TRAIN contacts with a bounded near-equivalent wrist seed.',
            shared_constants_for_all_cases=True,runtime_case_lookup=False),
        native_source_fk_center_error_m=fk_error, bounded_branch_center_error_m=branch_position_error,
        bounded_branch_rotation_max_error=branch_rotation_error, bounded_branch_seed_check=branch_seed_check, velocity_initializers=solvers,
        optional_static_pose_seed=static_receipt, candidates=candidates,
        hypothesis='Shorter wrist preparation path and measured high contact pose may improve legal hits and recovery.',
        planned_native_comparison='Each shared candidate gets all original128 normal ground cases, zero reset perturbation.',
        all_source_actors_and_normalizers_retained=True,original_physics_hardware_scores_and_ground_init_retained=True,
        native_execution=False, performance_inherited=False, formal_cases_read=False, formal_acceptance=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(candidates=len(candidates),selected_TRAIN=event['env_id'],
        bounded_branch_center_error_m=branch_position_error, output=str(output))), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--source-evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--static-bank', type=Path)
    args = parser.parse_args()
    build(args.source, args.source_evaluation, args.output, args.static_bank)
