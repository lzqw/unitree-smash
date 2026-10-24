#!/usr/bin/env python3
"""Audit actual contact velocities and propose bounded joint rates, without dynamics."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quadsmash.control.full_racket_kinematics_v4 import FullRacketKinematicsV4
from quadsmash.control.full_racket_geometry_v4 import FullRacketGeometryV4
from quadsmash.control.full_racket_contact_curriculum_v4 import usable_air_contact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['snapshot', 'run', 'checkpoint', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a fresh output directory')
    model = FullRacketKinematicsV4().double()
    limits = FullRacketGeometryV4().bounds
    warm = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    geometry_path = args.run/'geometry.json'
    geometry = json.loads(geometry_path.read_text())
    tool = geometry['body_names'].index('Racket')
    contacts_path, rows_path = [args.snapshot/name for name in ['contacts.partial.json', 'episodes.partial.json']]
    rows = {r['env_id']: r for r in json.loads(rows_path.read_text()) if r['episode_index'] == 0}
    contacts = [c for c in json.loads(contacts_path.read_text())
                if c['episode_index'] == 0 and c['first_racket_contact'] and c['string_contact']]
    desired_normal = np.array(warm['seed']['outward_contact_normal_world'])
    horizon = .08
    records, proposals, matrices, source_ids = [], [], [], []
    for contact in contacts:
        poses = np.array(contact['body_link_poses_local'])
        rotation = Rotation.from_quat(poses[0, [4, 5, 6, 3]]).as_matrix()
        tool_rotation = Rotation.from_quat(poses[tool, [4, 5, 6, 3]]).as_matrix()
        point = np.array(contact['point_m'])
        local = tool_rotation.T @ (point-poses[tool, :3])
        joints = np.array(contact['joint_position_rad'][12:])
        measured_qd = np.array(contact['joint_velocity_rad_s'][12:])
        with torch.inference_mode():
            center, normal, jp, jn = model(torch.tensor(joints[None]), torch.tensor(local))
        predicted_point = poses[0, :3]+rotation@center[0].numpy()
        jv, jnormal = [rotation@j[0].numpy() for j in [jp, jn]]
        normal = rotation@normal[0].numpy()
        omega = np.array(contact['root_angular_velocity_rad_s'])
        base = np.array(contact['root_com_linear_velocity_mps'])
        base += np.cross(omega, point-np.array(contact['root_com_position_m']))
        reconstructed = base+jv@measured_qd
        normal_base = np.cross(omega, normal)
        normal_rate = normal_base+jnormal@measured_qd
        native_normal_rate = np.cross(contact['flange_angular_velocity_rad_s'], contact['face_normal'])
        lower = np.maximum(-5., (limits[:, 0]-joints)/horizon)
        upper = np.minimum(5., (limits[:, 1]-joints)/horizon)
        eligible = contact['env_id'] in rows and usable_air_contact(rows[contact['env_id']])
        if eligible:
            matrices.append(dict(jv=jv, jn=jnormal, base=base, normal_base=normal_base,
                                 normal=normal, lower=lower, upper=upper))
            source_ids.append(contact['env_id'])
        record = dict(env_id=contact['env_id'], source_candidate=contact['env_id']//2,
            source_training_index=[1, 65][contact['env_id'] % 2], time_s=contact['time_s'],
            usable_air_contact_in_completed_row=eligible, point_position_error_m=float(np.linalg.norm(predicted_point-point)),
            measured_point_velocity_mps=contact['point_velocity_mps'], reconstructed_point_velocity_mps=reconstructed.tolist(),
            point_velocity_reconstruction_error_mps=float(np.linalg.norm(reconstructed-contact['point_velocity_mps'])),
            normal_rate_reconstruction_error_per_s=float(np.linalg.norm(normal_rate-native_normal_rate)),
            base_velocity_contribution_mps=base.tolist(), arm_velocity_contribution_mps=(jv@measured_qd).tolist(),
            reference_seed_predicted_velocity_mps=(base+jv@(warm['parameters'][26:32].numpy()*float(warm['parameters'][7]))).tolist(),
            joint_velocity_rad_s=measured_qd.tolist(), position_forecast_horizon_s=horizon,
            joint_rate_lower_rad_s=lower.tolist(), joint_rate_upper_rad_s=upper.tolist(), solutions=[])
        if np.any(upper < lower):
            record['solutions'].append(dict(feasible=False, reason='Actual state already beyond position-forecast rate bounds'))
        else:
            for lead_gain in [0., 4.]:
                for side_tolerance in [.25, .75]:
                    # These are proposal objectives, not task or hardware gates.
                    target_rate = lead_gain*(desired_normal-(desired_normal@normal)*normal)
                    A = np.vstack((jv[1], -jv[1], jv[2], -jv[2], jnormal, -jnormal))
                    b = np.r_[side_tolerance-base[1], side_tolerance+base[1],
                              -.2-base[2], 1.2+base[2],
                              target_rate+.75-normal_base, -target_rate+.75+normal_base]
                    fit = linprog(-jv[0], A_ub=A, b_ub=b, bounds=list(zip(lower, upper)), method='highs')
                    solution = dict(normal_turn_gain_per_s=lead_gain, lateral_speed_tolerance_mps=side_tolerance,
                                    feasible=bool(fit.success), status=int(fit.status))
                    if fit.success:
                        velocity = base+jv@fit.x
                        solution.update(joint_velocity_seed_rad_s=fit.x.tolist(),
                            predicted_point_velocity_mps=velocity.tolist(),
                            predicted_normal_rate_per_s=(normal_base+jnormal@fit.x).tolist())
                        if eligible:
                            proposals.append(dict(source_env_id=contact['env_id'], **solution))
                    record['solutions'].append(solution)
        records.append(record)
    assert len(records) >= 1
    assert len(matrices) >= 1
    sample = {key: np.stack([m[key] for m in matrices]) for key in matrices[0]}
    robust = []
    common_lower, common_upper = sample['lower'].max(0), sample['upper'].min(0)
    for lead_gain in [0., 4.]:
        for side_tolerance in [.75, 1.25]:
            for rate_tolerance in [.75, 1.5]:
                A, b = [], []
                for m in matrices:
                    target = lead_gain*(desired_normal-(desired_normal@m['normal'])*m['normal'])
                    task = np.vstack((m['jv'][1], -m['jv'][1], m['jv'][2], -m['jv'][2], m['jn'], -m['jn']))
                    bounds = np.r_[side_tolerance-m['base'][1], side_tolerance+m['base'][1],
                        -.1-m['base'][2], 1.5+m['base'][2],
                        target+rate_tolerance-m['normal_base'], -target+rate_tolerance+m['normal_base']]
                    A.extend(np.column_stack((task, np.zeros(len(task)))))
                    b.extend(bounds)
                    A.append(np.r_[-m['jv'][0], 1.])
                    b.append(m['base'][0])
                fit = linprog(np.r_[np.zeros(6), -1.], A_ub=A, b_ub=b,
                    bounds=[*zip(common_lower, common_upper), (None, None)], method='highs')
                if fit.success:
                    qd = fit.x[:6]
                    velocity = sample['base']+sample['jv']@qd
                    worst = int(velocity[:, 0].argmin())
                    robust.append(dict(source_env_id=source_ids[worst],
                        source_scope='One joint-rate direction jointly constrained over all prior usable contact states',
                        normal_turn_gain_per_s=lead_gain, lateral_speed_tolerance_mps=side_tolerance,
                        normal_rate_tolerance_per_s=rate_tolerance, feasible=True, status=int(fit.status),
                        joint_velocity_seed_rad_s=qd.tolist(), predicted_point_velocity_mps=velocity[worst].tolist()))
    for proposal in proposals+robust:
        qd = np.array(proposal['joint_velocity_seed_rad_s'])
        velocity = sample['base']+sample['jv']@qd
        rate = sample['normal_base']+sample['jn']@qd
        target_rate = 4.*(desired_normal-(sample['normal']@desired_normal)[:, None]*sample['normal'])
        violation = np.maximum(np.maximum(sample['lower']-qd, qd-sample['upper']), 0.)
        worst_forward = float(velocity[:, 0].min())
        maximum_side = float(np.abs(velocity[:, 1]).max())
        maximum_vertical_error = float(np.abs(velocity[:, 2]+.6).max())
        maximum_rate_error = float(np.linalg.norm(rate-target_rate, axis=1).max())
        proposal.update(worst_observed_state_forward_velocity_mps=worst_forward,
            maximum_observed_state_lateral_speed_mps=maximum_side,
            maximum_forecast_rate_bound_violation_rad_s=float(violation.max()),
            proposal_score=worst_forward-.25*maximum_side-.15*maximum_vertical_error
                           -.1*maximum_rate_error-2.*float(violation.max()))
    selected = []
    # Keep both consistency-oriented and faster rate directions. Constraining
    # every proposal to the worst saved state can discard the power direction
    # needed by the original full task.
    for family, pool, target_count in [('robust_prior_states', robust, 2),
                                        ('faster_single_state_proposal', proposals, 4)]:
        for proposal in sorted(pool, key=lambda p: p['proposal_score'], reverse=True):
            if proposal['maximum_forecast_rate_bound_violation_rad_s'] > 1e-6:
                continue
            qd = np.array(proposal['joint_velocity_seed_rad_s'])
            if all(np.linalg.norm(qd-np.array(p['joint_velocity_seed_rad_s'])) >= .4 for p in selected):
                selected.append(dict(proposal, selection_family=family))
            if len(selected) == target_count:
                break
    assert len(selected) == 4, 'Need four distinct measured-state velocity proposals'
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'observed_contact_kinematic_matrices.npz', **sample)
    paths = [contacts_path, rows_path, geometry_path, args.checkpoint, Path(__file__),
             ROOT/'quadsmash/control/full_racket_kinematics_v4.py']
    report = dict(created_at=datetime.now(timezone.utc).isoformat(),
        scope='Exact TRAIN contact-state kinematics and instantaneous bounded rate proposals. Root motion is held at its observed value. No torque, tracking, future pose, collision or recovery feasibility is certified.',
        actual_strings_contacts=len(records), usable_air_contacts_with_completed_rows=sum(r['usable_air_contact_in_completed_row'] for r in records),
        maximum_point_position_error_m=max(r['point_position_error_m'] for r in records),
        maximum_point_velocity_reconstruction_error_mps=max(r['point_velocity_reconstruction_error_mps'] for r in records),
        maximum_normal_rate_reconstruction_error_per_s=max(r['normal_rate_reconstruction_error_per_s'] for r in records),
        proposed_joint_rate_limit_rad_s=5., position_forecast_horizon_s=horizon,
        records=records, eligible_rate_proposals=proposals, robust_multi_state_rate_proposals=robust,
        selected_joint_rate_proposals=selected,
        selection_scope='Worst-state instantaneous kinematic proposal score over the eight prior usable TRAIN contacts; no changed-state outcome, torque or tracking feasibility.',
        new_dynamics_executed=False, formal_acceptance=False,
        sources={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (args.output/'REPORT.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ['actual_strings_contacts', 'usable_air_contacts_with_completed_rows',
        'maximum_point_position_error_m', 'maximum_point_velocity_reconstruction_error_mps',
        'maximum_normal_rate_reconstruction_error_per_s']}, indent=2), flush=True)
    print(json.dumps(dict(feasible_eligible_proposals=len(proposals),
        predicted_forward_speed_range_mps=([min(p['predicted_point_velocity_mps'][0] for p in proposals),
            max(p['predicted_point_velocity_mps'][0] for p in proposals)] if proposals else None))), flush=True)


if __name__ == '__main__':
    main()
