"""Fit shared bounded arm-rate directions to actual origin0 contact states."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation
import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from quadsmash.control.full_racket_kinematics_v4 import FullRacketKinematicsV4
from quadsmash.control.full_racket_geometry_v4 import FullRacketGeometryV4
from quadsmash.control.full_racket_origin0_residual_v4 import make_rally_controller


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    run = ROOT / 'reports/G07_full_racket_v4/gain_body8_best_fixed_train128_01'
    checkpoint_path = ROOT / 'reports/G07_full_racket_v4/gain_body8_sequential_n128_01/best_candidate.pt'
    source = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    controller = make_rally_controller(source, 2, 'cpu')
    reference = controller.reference_controller
    original_rate = reference.parameters[0, 26:32].numpy().copy() * float(reference.parameters[0, 7])
    rows = json.loads((run / 'episodes.json').read_text())
    contacts = json.loads((run / 'contacts.json').read_text())
    inputs = json.loads((run / 'INPUTS.json').read_text())
    assert inputs['policy_sha256'] == sha(checkpoint_path)
    assert inputs['training_indices'] == list(range(128)) and inputs['origin_spacing_m'] == 0.
    assert not any(inputs['reset_noise_amplitudes'].values())
    assert len(rows) == 128
    body_names = json.loads((run / 'geometry.json').read_text())['body_names']
    tool = body_names.index('Racket')
    model = FullRacketKinematicsV4().double()
    bounds = FullRacketGeometryV4().bounds
    samples, matrices, actual_contacts = [], [], []
    for row in rows:
        if not (row['legal_initial_contact'] and row['string_hit'] and row['airborne_string_hit']):
            continue
        event = next(e for e in contacts if e['env_id'] == row['env_id']
                     and e['episode_index'] == 0 and e['first_racket_contact'] and e['string_contact'])
        poses = np.asarray(event['body_link_poses_local'])
        rotation = Rotation.from_quat(poses[0, [4, 5, 6, 3]]).as_matrix()
        tool_rotation = Rotation.from_quat(poses[tool, [4, 5, 6, 3]]).as_matrix()
        point = np.asarray(event['point_m'])
        local = tool_rotation.T @ (point - poses[tool, :3])
        q = np.asarray(event['joint_position_rad'][12:])
        qd = np.asarray(event['joint_velocity_rad_s'][12:])
        with torch.inference_mode():
            center, normal, jv, jn = model(torch.tensor(q[None]), torch.tensor(local))
        jv, jn = rotation @ jv[0].numpy(), rotation @ jn[0].numpy()
        normal = rotation @ normal[0].numpy()
        omega = np.asarray(event['root_angular_velocity_rad_s'])
        base = np.asarray(event['root_com_linear_velocity_mps']) + np.cross(
            omega, point - np.asarray(event['root_com_position_m']))
        normal_base = np.cross(omega, normal)
        lower = np.maximum(-5., (bounds[:, 0] - q) / .08)
        upper = np.minimum(5., (bounds[:, 1] - q) / .08)
        assert np.all(lower <= upper)
        sample = dict(jv=jv, jn=jn, base=base, normal_base=normal_base, lower=lower, upper=upper)
        matrices.append(sample)
        actual_contacts.append(dict(episode=row, contact=event))
        samples.append(dict(source_training_index=row['source_training_index'],
            contact_time_s=event['time_s'], actual_point_velocity_mps=event['point_velocity_mps'],
            actual_normal_speed_mps=float(normal @ np.asarray(event['point_velocity_mps'])),
            actual_outgoing_velocity_mps=row['outgoing_velocity_mps'],
            actual_outgoing_speed_mps=float(np.linalg.norm(row['outgoing_velocity_mps'])),
            body_landing=row['jump']['body_landing'],
            position_reconstruction_error_m=float(np.linalg.norm(poses[0, :3] + rotation @ center[0].numpy() - point)),
            velocity_reconstruction_error_mps=float(np.linalg.norm(base + jv @ qd - event['point_velocity_mps'])),
            body_velocity_contribution_mps=base.tolist(), arm_velocity_contribution_mps=(jv @ qd).tolist(),
            actual_arm_joint_velocity_rad_s=qd.tolist(), arm_joint_position_rad=q.tolist(),
            original_joint_position_bounds_rad=bounds.tolist(),
            proposal_rate_lower_rad_s=lower.tolist(), proposal_rate_upper_rad_s=upper.tolist()))
    assert [sample['source_training_index'] for sample in samples] == [69, 100, 112, 123]
    assert max(sample['velocity_reconstruction_error_mps'] for sample in samples) < 1e-4
    shared_lower = np.max([sample['lower'] for sample in matrices], axis=0)
    shared_upper = np.min([sample['upper'] for sample in matrices], axis=0)
    assert np.all(shared_lower <= shared_upper)
    proposals = []
    for normal_tolerance in [1.5, 3., 6.]:
        constraints, values = [], []
        for sample in matrices:
            jv, jn, base, normal_base = [sample[key] for key in ['jv', 'jn', 'base', 'normal_base']]
            task = np.vstack((jv[1], -jv[1], jv[2], -jv[2], jn, -jn))
            limits = np.r_[.75 - base[1], .75 + base[1], .2 - base[2], 1.2 + base[2],
                           np.full(3, normal_tolerance) - normal_base,
                           np.full(3, normal_tolerance) + normal_base]
            constraints.extend(np.column_stack((task, np.zeros(len(task)))))
            values.extend(limits)
            constraints.append(np.r_[-jv[0], 1.])
            values.append(base[0])
        fit = linprog(np.r_[np.zeros(6), -1.], A_ub=constraints, b_ub=values,
                      bounds=[*zip(shared_lower, shared_upper), (None, None)], method='highs')
        assert fit.success
        qd = fit.x[:6]
        predicted = np.stack([sample['base'] + sample['jv'] @ qd for sample in matrices])
        proposals.append(dict(normal_rate_component_tolerance_per_s=normal_tolerance,
            shared_joint_rate_rad_s=qd.tolist(), predicted_point_velocity_by_TRAIN_order_mps=predicted.tolist(),
            minimum_predicted_forward_velocity_mps=float(predicted[:, 0].min()),
            LP_primal_constraint_max_violation=float(np.maximum(np.asarray(constraints) @ fit.x - values, 0.).max()),
            joint_rate_bound_max_violation=float(np.maximum(np.maximum(shared_lower - qd, qd - shared_upper), 0.).max())))
    candidates = [dict(candidate_id=0, name='unchanged_source', overrides={})]
    for proposal in proposals:
        for fraction in [.5, 1.]:
            rate = (1 - fraction) * original_rate + fraction * np.asarray(proposal['shared_joint_rate_rad_s'])
            assert np.max(np.abs(rate)) <= 5. + 1e-6
            candidates.append(dict(candidate_id=len(candidates),
                name=f"rate_tol{proposal['normal_rate_component_tolerance_per_s']:g}_blend{fraction:g}",
                overrides=dict(swing_joint_velocity_rad_s=rate.tolist()),
                contact_state_only_prediction_source=proposal['normal_rate_component_tolerance_per_s'],
                interpolation_fraction=fraction))
    for tilt in [.2, .35]:
        normal = np.array([1., 0., -tilt]); normal /= np.linalg.norm(normal)
        candidates.append(dict(candidate_id=len(candidates), name=f'fast_rate_with_normal_down{tilt:g}',
            overrides=dict(swing_joint_velocity_rad_s=proposals[-1]['shared_joint_rate_rad_s'],
                           desired_normal_world=normal.tolist()),
            normal_change_is_unexecuted_policy_target=True))
    files = [checkpoint_path, *(run / name for name in ['episodes.json', 'contacts.json', 'geometry.json', 'INPUTS.json']),
             ROOT / 'quadsmash/control/full_racket_kinematics_v4.py', ROOT / 'quadsmash/control/full_racket_geometry_v4.py',
             ROOT / 'quadsmash/control/piper_geometry.py', ROOT / 'configs/assembly_v2.json', Path(__file__)]
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(),
        scope='Actual contact-state reconstruction and shared bounded instantaneous rate directions; not a new motion, impact, tracking or recovery result.',
        actual_original128_checked=True, actual_legal_air_strings=4, actual_samples=samples,
        original_effective_swing_rate_rad_s=original_rate.tolist(),
        original_normal_world=reference.teacher.normal_world.tolist(),
        original_desired_point_velocity_world_mps=reference.teacher.velocity_world.tolist(),
        original_swing_half_window_s=float(reference.parameters[0, 19]),
        shared_rate_lower_rad_s=shared_lower.tolist(), shared_rate_upper_rad_s=shared_upper.tolist(),
        shared_instantaneous_proposals=proposals,
        position_forecast_horizon_s=.08, policy_rate_search_limit_rad_s=5.,
        proposal_only_constraints=dict(maximum_lateral_speed_mps=.75,
                                       vertical_speed_interval_mps=[-1.2, .2]),
        candidate_count=len(candidates), candidates=candidates,
        root_motion_held_at_measured_contact_state=True,
        blended_rates_not_claimed_to_satisfy_LP_at_all_states=True,
        candidate_joint_torque_tracking_collision_and_recovery_feasibility_proven=False,
        unchanged_physics_hardware_rules_target_and_acceptance=True,
        sources_sha256={str(path): sha(path) for path in files},
        native_execution=False, formal_acceptance=False)
    with (HERE / 'RESULT.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    with (HERE / 'ACTUAL_FOUR_CONTACTS.json').open('x') as stream:
        json.dump(actual_contacts, stream, indent=2)
        stream.write('\n')
    np.savez_compressed(HERE / 'contact_kinematic_matrices.npz',
                        **{key: np.stack([sample[key] for sample in matrices]) for key in matrices[0]})
    print(json.dumps(dict(actual_legal_air_strings=4, candidate_count=len(candidates),
                         original_forward_head_speeds_mps=[sample['actual_point_velocity_mps'][0] for sample in samples],
                         proposed_worst_state_forward_speeds_mps=[p['minimum_predicted_forward_velocity_mps'] for p in proposals])))


if __name__ == '__main__':
    main()
