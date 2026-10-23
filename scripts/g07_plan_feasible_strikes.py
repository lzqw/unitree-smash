#!/usr/bin/env python3
"""CPU inverse flight/kinematic seeds for G07; no dynamic acceptance claims."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.optimize import differential_evolution, minimize, root

from quadsmash.control.piper_geometry import PiperGeometry
from quadsmash.physics.smash_feasibility import predict_flights
from quadsmash.physics.shuttlecock import ShuttlecockModel


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pose_description(model, result, *, face=False, **base):
    q = result.x[:6]
    center, rotation = model.forward(q, **base)
    contact, normal = model.contact_center(q, uv=result.x[6:8] if face else (0, 0), **base)
    jacobian = model.contact_jacobian(q, uv=result.x[6:8] if face else (0, 0), **base)
    return dict(arm_q_rad=q.tolist(), root=base, center_xyz_m=center.tolist(),
                contact_ball_center_xyz_m=contact.tolist(), normal_world=normal.tolist(),
                contact_uv_m=result.x[6:8].tolist() if face else [0, 0],
                box_top_world_m=float(center[2] + model.vertical_support(rotation)),
                root_height_m=float(base.get('base_position', (0, 0, .323))[2]),
                contact_jacobian_m_per_rad=jacobian.tolist(),
                limitation="kinematic pose only; joint tracking, collision and whole-body recovery unverified")


def make_search_seed(output):
    model = ShuttlecockModel()
    start, target, target_time = np.array([8.18, 0., 1.15]), np.array([1.4, 0., 1.65]), 1.35
    def arrival(v):
        p, velocity = start.copy(), np.array([v[0], 0., v[1]])
        for _ in range(round(target_time/.002)):
            p, velocity = model.integrate(p, velocity, .002)
        return p, velocity
    solution = root(lambda v: (arrival(v)[0] - target)[[0, 2]], [-6., 7.2])
    if not solution.success or np.linalg.norm(arrival(solution.x)[0] - target) > 1e-6:
        raise RuntimeError('Physical incoming launch solve did not converge')
    incoming = arrival(solution.x)[1]
    outgoing = np.array([7.5, 0., -3.])
    normal = outgoing - incoming
    normal /= np.linalg.norm(normal)
    normal_speed = (outgoing @ normal + .72*(incoming @ normal)) / 1.72
    prediction = predict_flights(target, outgoing, dt=.001)
    launch = dict(shuttle_start=start.tolist(), shuttle_velocity=[float(solution.x[0]), 0., float(solution.x[1])],
                  event_goal_time_s=target_time, event_goal_position=target.tolist(),
                  event_goal_normal=normal.tolist(), event_goal_velocity=(normal_speed*normal).tolist())
    (output/'next_search_launch.json').write_text(json.dumps(launch, indent=2)+'\n')
    seed = dict(kind='CPU design seed only, not executed in PhysX',
                incoming_velocity_at_target_mps=incoming.tolist(),
                desired_outgoing_velocity_mps=outgoing.tolist(), desired_normal_world=normal.tolist(),
                infinite_mass_racket_normal_speed_estimate_mps=float(normal_speed),
                normal_speed_estimate_limitation='not a finite-mass robot impulse solution',
                net_margin_m=float(prediction['net_margin'][0]),
                conservative_net_exit_margin_m=float(prediction['conservative_net_exit_margin'][0]),
                first_landing_xyz_m=prediction['landing_position'][0].tolist(),
                launch_solver='RK4 locked drag/gravity solve to [1.4,0,1.65] at 1.35 s',
                gpu_test_status='NOT_RUN_authorization_expired')
    (output/'next_search_seed.json').write_text(json.dumps(seed, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    model = PiperGeometry()
    bounds = model.bounds.tolist()
    center_opt = differential_evolution(lambda q: -model.forward(q)[0][2], bounds,
                                        seed=37, popsize=12, maxiter=120, tol=1e-8)
    poses = {'optimized_center_at_canonical_root': pose_description(model, center_opt)}
    # Keep all contacts 15 mm inside the declared blade edge. A high corner
    # alone does not supply a usable face or favorable collision normal.
    for forward_normal_min in (0., .7, .9):
        def loss(values):
            q, uv = values[:6], values[6:]
            contact, normal = model.contact_center(q, uv=uv)
            penalty = max(0, forward_normal_min - normal[0])**2 * 100
            penalty += max(0, abs(normal[1]) - .1)**2 * 100
            return -contact[2] + penalty
        result = differential_evolution(loss, bounds + [[-.095, .095]] * 2,
                                         seed=38, popsize=12, maxiter=200, tol=1e-8)
        def normals(values):
            return model.contact_center(values[:6], uv=values[6:])[1]
        refined = minimize(lambda values: -model.contact_center(values[:6], uv=values[6:])[0][2],
                           result.x, method='SLSQP', bounds=bounds + [[-.095, .095]] * 2,
                           constraints=[{'type': 'ineq', 'fun': lambda values: normals(values)[0] - forward_normal_min},
                                        {'type': 'ineq', 'fun': lambda values: .1 - abs(normals(values)[1])}],
                           options={'ftol': 1e-11, 'maxiter': 200})
        if not refined.success:
            raise RuntimeError(f'Constrained pose refinement failed: {refined.message}')
        result = refined
        poses[f'effective_face_normal_x_ge_{forward_normal_min}'] = pose_description(model, result, face=True)
    reference = json.loads((ROOT / 'reports/G07/full_arm_range03/metrics.json').read_text())
    reference_errors = []
    for row in reference['max_z_configurations']:
        position, _ = model.forward(row['arm_joint_position'])
        reference_errors.append(float(np.linalg.norm(position - row['racket_xyz'])))
    geometry = dict(
        method='pinned URDF forward kinematics and numerical pose optimization',
        coordinate_system='world x toward opponents, z up; local racket face normal +X',
        canonical_root_xyz=[0, 0, .323],
        box_half_extents_m=model.half_extents.tolist(),
        orientation_independent_box_half_diagonal_m=float(np.linalg.norm(model.half_extents)),
        exact_world_vertical_support='sum_i abs(R[2,i]) * half_extents[i]',
        previous_4096_sample_max_center_z_m=reference['position_max'][2],
        old_sample_urdf_fk_position_error_max_m=max(reference_errors),
        nominal_native_reset_error_m=float(np.linalg.norm(model.forward([0,1,-1,0,0,0])[0]
                                                          - [.284133315, -.00000090035, .646758080])),
        root_relative_triangle_distance_bound_m=model.unrestricted_reach_bound(),
        rigorous_bound_scope='distance from Go2 root only; does not bound dynamic root height',
        previous_1_535m_claim='withdrawn: finite sampling, finite jump search, and incorrect orientation-independent box extension',
        poses=poses,
        limitations=['Numerical pose maxima are found values, not certified global maxima.',
                     'URDF FK is cross-checked against old native poses; no new PhysX rollout executed.',
                     'No self-collision, torque, velocity, or landing-recovery feasibility certificate.'])
    (args.output/'geometry.json').write_text(json.dumps(geometry, indent=2)+'\n')
    # Solve the coupled flight problem: contact location, full outgoing vector,
    # net clearance and FIRST landing must simultaneously satisfy the task.
    tables, seeds = [], []
    for x_hit in (.3, .75, 1., 1.2, 1.3, 1.4, 1.45):
        heights, vxs, vzs = np.meshgrid(np.arange(1.35, 2.311, .02),
                                       np.arange(4., 14.01, .5), np.arange(-9., -.99, .5), indexing='ij')
        positions = np.column_stack((np.full(heights.size, x_hit), np.zeros(heights.size), heights.ravel()))
        velocities = np.column_stack((vxs.ravel(), np.zeros(heights.size), vzs.ravel()))
        prediction = predict_flights(positions, velocities, dt=.002)
        okay = prediction['predicted_legal_attack']
        speed = np.linalg.norm(velocities, axis=1)
        row = {'hit_x_m': x_hit, 'distance_to_net_m': 1.5-x_hit}
        for cap in (8.5, 10., 12., 14.):
            ids = np.flatnonzero(okay & (speed <= cap + 1e-10))
            if not len(ids):
                row[f'min_grid_height_at_speed_le_{cap:g}'] = None
                continue
            # Prefer some clearance, minimum required height, then lower speed.
            order = np.lexsort((speed[ids], positions[ids, 2]))
            best = ids[order[0]]
            row[f'min_grid_height_at_speed_le_{cap:g}'] = float(positions[best, 2])
            seeds.append(dict(speed_cap_mps=cap, contact_position_m=positions[best].tolist(),
                              outgoing_velocity_mps=velocities[best].tolist(),
                              net_margin_m=float(prediction['net_margin'][best]),
                              conservative_net_exit_margin_m=float(prediction['conservative_net_exit_margin'][best]),
                              net_crossing_xyz_m=prediction['net_position'][best].tolist(),
                              first_landing_xyz_m=prediction['landing_position'][best].tolist(),
                              first_landing_time_after_hit_s=float(prediction['landing_time'][best]),
                              kind='predicted free-flight target, not a PhysX contact result'))
        tables.append(row)
    flights = dict(method='batched RK4, locked quadratic drag and gravity, no bounce',
                   timestep_s=.002, height_grid_m=.02,
                   net_plane_x_m=1.5, net_top_m=1.524, sphere_radius_m=.025,
                   first_landing_target_x_m=[2,4], table=tables, seeds=seeds,
                   limitation='Necessary outgoing free-flight conditions only. Whole-body collision must produce these velocities.')
    (args.output/'flight_regions.json').write_text(json.dumps(flights, indent=2)+'\n')
    make_search_seed(args.output)
    manifest = dict(kind='CPU preparation, no GPU simulation, no policy acceptance',
                    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                    source_sha256={str(p.relative_to(ROOT)): digest(p) for p in
                        [model.urdf, ROOT/'configs/assembly_v2.json', ROOT/'configs/shuttle_physics_v2.json',
                         ROOT/'assets/quadsmash_standard_v2.usd', ROOT/'quadsmash/control/piper_geometry.py',
                         ROOT/'quadsmash/physics/smash_feasibility.py',
                         ROOT/'quadsmash/physics/smash_events.py',
                         ROOT/'quadsmash/tasks/stand/smash_env_v3.py',
                         ROOT/'scripts/search_g07_whole_body_v3.py', Path(__file__).resolve()]})
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'optimized_center_z_m': -center_opt.fun, 'flight_region_table': tables}, indent=2), flush=True)


if __name__ == '__main__':
    main()
