"""Validate momentum-consistent hit velocities on actual native contacts.

All original128 rows are retained/recomputed. Optimization uses only actual
legal airborne contacts to diagnose local rate opportunities; no new rollout
or physical feasibility is claimed. No changes to any running controller.
"""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation

from free_floating import hit_state, linearization

HERE = Path(__file__).resolve().parent
MAIN = HERE.parents[2]
BASE = HERE.parent
MODEL_DIR = BASE/'centroidal_recovery_model_20261027'
ROOT = MAIN.parent/'unitree_smash_contact_recovery'
sys.path.insert(0, str(ROOT))
from quadsmash.control.full_racket_train_evaluation_v4 import training_cases
from quadsmash.control.full_racket_body_search_v4 import summarize_candidate
from quadsmash.evaluation_v4 import independent_episode


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(p):
    return json.loads(p.read_text())


def transform(rotation, vectors):
    return np.einsum('nij,nj->ni', rotation, vectors)


def optimize(item, index, qd, q, limits, rotation, mode):
    """Independent contact-state LP; position box uses an80ms linear preview.

    Point y/z and normal-rate constraints are auxiliary proposal constraints,
    not task or hardware changes. No torque/trajectory/contact feasibility.
    """
    if mode == 'measured_root_arm_only':
        matrix = item['fixed_root_hit_matrix'][index, :, 3:]
        normal_matrix = item['fixed_root_normal_rate_matrix'][index, :, 3:]
        # Subtract measured joint contribution; measured root stays fixed.
        offset = item['velocity'][index]-matrix@qd
        normal_offset = item['normal_rate'][index]-normal_matrix@qd
    else:
        matrix, offset = item['hit_jacobian'][index], item['hit_offset'][index]
        normal_matrix, normal_offset = item['normal_rate_jacobian'][index], item['normal_rate_offset'][index]
    matrix, offset = rotation@matrix, rotation@offset
    normal_matrix, normal_offset = rotation@normal_matrix, rotation@normal_offset
    cap = np.r_[np.full(12, 30.), np.full(6, 5.)]
    lower = np.maximum(-cap, (limits[:, 0]-q)/.08)
    upper = np.minimum(cap, (limits[:, 1]-q)/.08)
    if mode in ['measured_root_arm_only', 'conserved_momentum_arm_only']:
        if np.any((qd[:12]<lower[:12]) | (qd[:12]>upper[:12])):
            return dict(success=False, reason='Measured fixed leg rates violate original80ms preview box')
        lower[:12] = upper[:12] = qd[:12]
    else:
        change = {'conserved_momentum_leg_delta2': 2., 'conserved_momentum_leg_delta5': 5.}[mode]
        lower[:12] = np.maximum(lower[:12], qd[:12]-change)
        upper[:12] = np.minimum(upper[:12], qd[:12]+change)
    # A measured joint can be outside its soft preview interval. Do not
    # silently clip it or widen the position limits to obtain a solution.
    if np.any(lower > upper):
        return dict(success=False, reason='Original joint/velocity preview box infeasible')
    a = np.vstack((matrix[1], -matrix[1], matrix[2], -matrix[2], normal_matrix, -normal_matrix))
    b = np.r_[.75-offset[1], .75+offset[1], .2-offset[2], 1.2+offset[2],
              np.full(3, 3.)-normal_offset, np.full(3, 3.)+normal_offset]
    fit = linprog(-matrix[0], A_ub=a, b_ub=b, bounds=list(zip(lower, upper)), method='highs')
    if not fit.success:
        return dict(success=False, reason=fit.message)
    violation = max(float(np.maximum(a@fit.x-b, 0.).max()),
                    float(np.maximum(np.maximum(lower-fit.x, fit.x-upper), 0.).max()))
    assert violation < 1e-7
    return dict(success=True, point_velocity_world_m_s=(offset+matrix@fit.x).tolist(),
        normal_rate_world_per_s=(normal_offset+normal_matrix@fit.x).tolist(),
        joint_velocity_rad_s=fit.x.tolist(), LP_maximum_constraint_violation=violation,
        root_motion_conserved=mode!='measured_root_arm_only')


def main():
    spec = importlib.util.spec_from_file_location('validated_strike_model', MODEL_DIR/'known_model.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    check = load(MODEL_DIR/'ACTUAL_MODEL_CHECK.json')
    for name in ['known_model.py', 'MODEL.json']:
        assert sha(MODEL_DIR/name) == check['source_sha256'][str(MODEL_DIR/name)]
    model = module.KnownModel()
    assert all(sha(Path(p))==h for p,h in model.record['source_sha256'].items())
    protocol_path = ROOT/'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json'
    protocol = load(protocol_path)
    protocol['_root'] = str(ROOT)
    cases, indices = training_cases(ROOT, protocol)
    sources = [Path(__file__), HERE/'free_floating.py', MODEL_DIR/'known_model.py', MODEL_DIR/'MODEL.json',
               MODEL_DIR/'ACTUAL_MODEL_CHECK.json', protocol_path,
               ROOT/'quadsmash/evaluation_v4.py', ROOT/'quadsmash/control/full_racket_body_search_v4.py',
               ROOT/'quadsmash/control/full_racket_train_evaluation_v4.py',
               ROOT/'quadsmash/tasks/stand/coupled_env.py']
    snapshots = []
    for name, suffix in [('baseline0', '000'), ('early_lateral1', '001')]:
        run = BASE/'early_lateral7_n128_01'
        paths = [run/f'episodes_{suffix}.json', run/f'contacts_{suffix}.json',
                 run/f'initial_ground_states_{suffix}.json', run/'INPUTS.json', run/'geometry.json']
        hashes = {str(p):sha(p) for p in paths}
        rows, contacts, initial, inputs, geometry = [load(p) for p in paths]
        assert len(rows)==128 and inputs['training_indices']==list(range(128))
        assert inputs['origin_spacing_m']==0 and initial['all_clocks_zero']
        assert all(row['ground_init'] for row in rows)
        assert [row['source_training_index'] for row in rows]==list(range(128))
        assert [row['episode_index'] for row in rows]==initial['episode_ids']
        assert indices==list(range(128))
        assert [row['case_id'] for row in rows]==[case['case_id'] for case in cases]
        summary = summarize_candidate(rows, protocol)
        rescored = [independent_episode(row, protocol) for row in rows]
        assert summary['complete']==sum(x['complete'] for x in rescored)
        assert summary['airborne_contacts']==sum(x['airborne_contact'] for x in rescored)
        assert summary['conditional_recovery_count']==sum(x['conditional_recovery'] for x in rescored)
        assert geometry['body_names']==model.names
        selected = [row for row in rows if row['legal_initial_contact'] and row['string_hit'] and row['airborne_string_hit']]
        samples, events = [], []
        for row in selected:
            matches = [e for e in contacts if e['env_id']==row['env_id'] and e['episode_index']==row['episode_index']
                       and e['first_racket_contact'] and e['string_contact']]
            assert len(matches)==1
            events.append(matches[0])
        q, qd = [np.asarray([e[key] for e in events], dtype=float) for key in ['joint_position_rad', 'joint_velocity_rad_s']]
        poses = np.asarray([e['body_link_poses_local'] for e in events])
        root = module.rotation(poses[:, model.root, 3:])
        tool = model.indices['Racket']
        tool_rotation = module.rotation(poses[:, tool, 3:])
        point_world = np.asarray([e['point_m'] for e in events])
        local_point = transform(tool_rotation.transpose(0, 2, 1), point_world-poses[:, tool, :3])
        local_normal = transform(tool_rotation.transpose(0, 2, 1), np.asarray([e['face_normal'] for e in events]))
        omega_world = np.asarray([e['root_angular_velocity_rad_s'] for e in events])
        root_com_offset = np.asarray([e['root_com_position_m'] for e in events])-poses[:, model.root, :3]
        link_velocity_world = np.asarray([e['root_com_linear_velocity_mps'] for e in events])-np.cross(omega_world, root_com_offset)
        root_v = transform(root.transpose(0, 2, 1), link_velocity_world)
        root_w = transform(root.transpose(0, 2, 1), omega_world)
        item = linearization(model, q, local_point, local_normal, root_v, root_w, qd)
        reconstructed = poses[:, model.root, :3]+transform(root, item['point'])
        velocity = transform(root, item['velocity'])
        expected = np.asarray([e['point_velocity_mps'] for e in events])
        position_error = float(np.linalg.norm(reconstructed-point_world, axis=1).max())
        velocity_error = float(np.linalg.norm(velocity-expected, axis=1).max())
        assert position_error<2e-5 and velocity_error<1e-4
        exact = item['hit_offset']+transform(item['hit_jacobian'], qd)
        assert abs(exact-item['velocity']).max()<1e-11
        # Check eliminated velocities by full-body forward reconstruction for
        # deterministically varied18-joint rates; never applied to simulation.
        rng = np.random.default_rng(724201)
        random_qd = qd+rng.uniform(-2., 2., qd.shape)
        changed_w = item['root_angular_offset']+transform(item['root_angular_jacobian'], random_qd)
        changed_v = item['root_linear_offset']+transform(item['root_linear_jacobian'], random_qd)
        varied = model.forward(q, random_qd, changed_v, changed_w)
        _, varied_hit, _ = hit_state(model, varied, local_point, local_normal)
        h_error = float(abs(varied['angular_momentum']-item['state']['angular_momentum']).max())
        p_error = float(abs((varied['center_velocity']-item['state']['center_velocity'])*model.mass.sum()).max())
        affine_error = float(abs(varied_hit-item['hit_offset']-transform(item['hit_jacobian'], random_qd)).max())
        assert max(h_error, p_error, affine_error)<1e-10
        # Independent central kinematic differentiation of the moving root
        # and18joints, not just a second evaluation of the affine equations.
        epsilon = 1e-6
        points = []
        for sign in [-1., 1.]:
            shifted = model.forward(q+sign*epsilon*random_qd)
            shifted_point = hit_state(model, shifted, local_point, local_normal)[0]
            delta_rotation = Rotation.from_rotvec(sign*epsilon*changed_w).as_matrix()
            points.append(sign*epsilon*changed_v+transform(delta_rotation, shifted_point))
        finite_difference_error = float(abs((points[1]-points[0])/(2*epsilon)-varied_hit).max())
        assert finite_difference_error<1e-7
        limits = np.asarray(inputs['joint_limits_rad'])
        assert limits.shape==(18, 2)
        for i,(row,event) in enumerate(zip(selected,events)):
            modes = {mode:optimize(item, i, qd[i], q[i], limits, root[i], mode) for mode in
                ['measured_root_arm_only', 'conserved_momentum_arm_only',
                 'conserved_momentum_leg_delta2', 'conserved_momentum_leg_delta5']}
            normal = np.asarray(event['face_normal'])
            samples.append(dict(training_index=row['source_training_index'], episode_index=row['episode_index'],
                contact_clock_s=event['time_s'], actual_point_velocity_world_m_s=expected[i].tolist(),
                actual_normal_point_speed_m_s=float(normal@expected[i]),
                actual_outgoing_speed_m_s=float(np.linalg.norm(row['outgoing_velocity_mps'])),
                actual_contact_flight_recovery=row['contact_flight_recovery'],
                actual_target_landing=row['landed_target'], actual_shuttle_net_contact=row['shuttle_net_contact'],
                actual_net_blocked=row['net_blocked_fault'], instantaneous_rate_fits=modes))
        result = dict(name=name, original_episodes=128, actual_airborne_contacts=len(selected),
            actual_complete=summary['complete'],
            actual_contact_flight_recovery_count=summary['conditional_recovery_count'],
            actual_NF=summary['numerical_failures'],
            contact_position_max_error_m=position_error, contact_velocity_max_error_m_s=velocity_error,
            conserved_H_max_error_kg_m2_s=h_error, conserved_P_max_error_kg_m_s=p_error,
            perturbed_velocity_affine_max_error_m_s=affine_error,
            independent_central_kinematic_difference_max_error_m_s=finite_difference_error, samples=samples)
        assert all(sha(Path(p))==h for p,h in hashes.items())
        snapshots.append(result)
        sources.extend(paths)
    output = dict(checked_at=datetime.now(timezone.utc).isoformat(), runs=snapshots,
        scope='Actual contact reconstruction and instantaneous fixed-q conserved-momentum rate proposals only.',
        actuator_velocity_limits_rad_s=dict(legs=30., arms=5.),
        original_joint_position_limits_used=True, auxiliary_preview_horizon_s=.08,
        auxiliary_point_y_limit_m_s=.75, auxiliary_point_z_interval_m_s=[-1.2,.2],
        auxiliary_normal_rate_component_limit_per_s=3.,
        all_original128_case_orders_and_native_episode_ids_preserved=True,
        extra_force_or_simulator_state_write=False, native_execution=False,
        torque_tracking_or_new_contact_or_recovery_proven=False,
        formal_cases_read=False, formal_acceptance=False, goal_complete=False,
        source_sha256={str(p):sha(p) for p in sources})
    with (HERE/'ACTUAL_CHECK.json').open('x') as stream:
        stream.write(json.dumps(output, indent=2, allow_nan=False)+'\n')
    for run in snapshots:
        print(json.dumps({k:v for k,v in run.items() if k!='samples'}))
        for mode in run['samples'][0]['instantaneous_rate_fits']:
            fits = [s['instantaneous_rate_fits'][mode] for s in run['samples']]
            speeds = [f['point_velocity_world_m_s'][0] for f in fits if f['success']]
            print(json.dumps(dict(run=run['name'],mode=mode,successful_local_LP=len(speeds),
                total_actual_contacts=len(fits),forward_speeds_m_s=speeds)))


if __name__=='__main__':
    main()
