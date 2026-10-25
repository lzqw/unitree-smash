"""Read-only diagnosis of completed frozen TRAIN128 attacks.

Counterfactual velocities are free-flight calculations, never robot execution,
contact feasibility, a new success label, or formal evaluation performance.
"""
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from quadsmash.evaluation_v4 import independent_episode
from quadsmash.physics.court_v4 import top_height
from quadsmash.physics.shuttlecock import ShuttlecockModel, ShuttlecockParams


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flight(model, positions, velocities, court, radius, dt):
    """RK4 free flight; interpolate actual geometric crossing/first-ground planes."""
    positions = np.asarray(positions, dtype=float).copy()
    velocities = np.asarray(velocities, dtype=float).copy()
    count = len(positions)
    landing = np.full((count, 3), np.nan)
    crossing = np.full((count, 3), np.nan)
    landing_time = np.full(count, np.nan)
    # Require the entire sphere to have cleared the far face of the unchanged
    # net. This is a conservative no-touch flight check, not the native rule
    # verdict: touching the actual tape can still be a legal return.
    plane = court['net_plane_x_m'] + court['net_equivalent_thickness_m'] / 2 + radius
    for step in range(int(2 / dt)):
        before = positions.copy()
        positions, velocities = model.integrate(positions, velocities, dt)
        crossed = np.isnan(crossing[:, 0]) & (before[:, 0] < plane) & (positions[:, 0] >= plane)
        if crossed.any():
            f = (plane - before[crossed, 0]) / (positions[crossed, 0] - before[crossed, 0])
            crossing[crossed] = before[crossed] + f[:, None] * (positions[crossed] - before[crossed])
        landed = np.isnan(landing[:, 0]) & (before[:, 2] > radius) & (positions[:, 2] <= radius)
        if landed.any():
            f = (radius - before[landed, 2]) / (positions[landed, 2] - before[landed, 2])
            landing[landed] = before[landed] + f[:, None] * (positions[landed] - before[landed])
            landing_time[landed] = (step + f) * dt
        if np.isfinite(landing).all():
            break
    assert np.isfinite(landing).all() and np.isfinite(crossing).all()
    margin = crossing[:, 2] - radius - top_height(crossing[:, 1], court)
    return dict(first_landing=landing, crossing=crossing, conservative_no_touch_net_margin=margin,
                first_landing_elapsed_s=landing_time)


def target_solution(model, position, velocity, target_x, court, radius):
    """Rotate an outgoing velocity down, preserving its norm and horizontal azimuth."""
    speed = float(np.linalg.norm(velocity))
    azimuth = velocity[:2] / np.linalg.norm(velocity[:2])
    low = float(np.arctan2(-velocity[2], np.linalg.norm(velocity[:2])))
    high = np.deg2rad(70.)
    for _ in range(24):
        angle = (low + high) / 2
        trial = np.r_[speed * np.cos(angle) * azimuth, -speed * np.sin(angle)]
        result = flight(model, position[None], trial[None], court, radius, .0005)
        if result['first_landing'][0, 0] > target_x:
            low = angle
        else:
            high = angle
    angle = (low + high) / 2
    trial = np.r_[speed * np.cos(angle) * azimuth, -speed * np.sin(angle)]
    result = flight(model, position[None], trial[None], court, radius, .00025)
    original_angle = np.rad2deg(np.arctan2(-velocity[2], np.linalg.norm(velocity[:2])))
    return dict(target_landing_x_m=target_x, unchanged_speed_mps=speed,
        counterfactual_outgoing_velocity_mps=trial.tolist(), downward_angle_degrees=float(np.rad2deg(angle)),
        downward_angle_change_degrees=float(np.rad2deg(angle)-original_angle),
        first_landing_m=result['first_landing'][0].tolist(),
        conservative_no_touch_net_margin_m=float(result['conservative_no_touch_net_margin'][0]),
        robot_execution=False, achievable_racket_contact_not_verified=True)


def main():
    assert not (HERE/'REPORT.json').exists(), 'Preserve existing evidence'
    run = ROOT/'reports/G07_full_racket_v4/residual_ppo_centered_base8_n2048_probe8_fixed_train128_02'
    policy = ROOT/'reports/G07_full_racket_v4/residual_ppo_centered_base8_n2048_probe8_02/policy.pt'
    protocol_path = ROOT/'configs/g07_full_racket_eval_v4_r3.json'
    protocol = read(protocol_path)
    protocol['_root'] = str(ROOT)
    court_path = ROOT/'configs/court_net_v4.json'
    physics_path = ROOT/'configs/shuttle_physics_v2.json'
    court, physics = read(court_path), read(physics_path)
    config_path = ROOT/'configs/g07_full_racket_residual_ppo_centered_v4.json'
    config = read(config_path)
    assert read(run/'job.json')['status'] == 'SUCCEEDED'
    assert read(run/'INPUTS.json')['policy_sha256'] == sha(policy)
    rows, contacts = read(run/'episodes.json'), read(run/'contacts.json')
    assert len(rows) == 128 and [r['source_training_index'] for r in rows] == list(range(128))
    outcomes = [independent_episode(row, protocol) for row in rows]
    attacks = [i for i, (row, outcome) in enumerate(zip(rows, outcomes)) if
        outcome['airborne_contact'] and row['released'] and np.linalg.norm(row['outgoing_velocity_mps']) >= 8.
        and row['outgoing_velocity_mps'][0] > 0 and row['outgoing_velocity_mps'][2] <= -1.]
    assert attacks == [6, 7, 23]
    table = []
    for row, outcome in zip(rows, outcomes):
        velocity = np.asarray(row['outgoing_velocity_mps'])
        landing = np.asarray(row['first_landing_m'])
        speed = float(np.linalg.norm(velocity)) if np.isfinite(velocity).all() else None
        distance = None
        if np.isfinite(landing).all():
            dx = max(court['target_x_limits_m'][0]-landing[0], 0, landing[0]-court['target_x_limits_m'][1])
            dy = max(abs(landing[1])-court['target_y_half_width_m'], 0)
            distance = float(np.hypot(dx, dy))
        table.append(dict(TRAIN=row['source_training_index'], had_contact=row['had_contact'],
            legal_air=outcome['airborne_contact'], attack=row['source_training_index'] in attacks,
            outgoing_speed_mps=speed, outgoing_vz_mps=float(velocity[2]) if speed is not None else None,
            crossed_net=row['crossed_net'], landed_legal_court=row['landed_legal_court'],
            landed_target=row['landed_target'], target_distance_m=distance,
            conditional_recovery=outcome['conditional_recovery'], fault=outcome['fault'], complete=outcome['complete']))
    with (HERE/'ALL128_EVENT_BOTTLENECKS.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(table)
    params = ShuttlecockParams()
    assert abs(.5*params.air_density*params.drag_coefficient*params.reference_area/params.mass
               - physics['quadratic_drag_k_per_m']) < 1e-14
    assert params.mass == physics['mass_kg']
    model, radius = ShuttlecockModel(params), physics['collision_radius_m']
    evidence = []
    for index in attacks:
        row = rows[index]
        selected = [event for event in contacts if event['env_id'] == index]
        assert len(selected) == 1 and selected[0]['first_racket_contact'] and selected[0]['string_contact']
        event = selected[0]
        position = np.asarray(event['ball_position_m'])
        velocity = np.asarray(event['ball_velocity_mps'])
        assert np.array_equal(velocity, np.asarray(row['outgoing_velocity_mps']))
        measured_landing = np.asarray(row['first_landing_m'])
        baseline = flight(model, position[None], velocity[None], court, radius, .0005)
        refined = flight(model, position[None], velocity[None], court, radius, .00025)
        error = float(np.linalg.norm(baseline['first_landing'][0] - measured_landing))
        assert error < .025, 'Uncalibrated free-flight model; preserve failure instead of claiming a plan'
        convergence = float(np.linalg.norm(baseline['first_landing'][0]-refined['first_landing'][0]))
        assert convergence < 1e-5
        distance = float(measured_landing[0] - court['target_x_limits_m'][1])
        landing_bonus = config['joint_placement_reward']['weight'] * np.exp(
            -distance**2 / config['joint_placement_reward']['distance_scale_m']**2)
        delay_steps = int(np.ceil(row['first_landing_time_s']/.01) - np.ceil(row['release_time_s']/.01))
        decay = (config['runner']['algorithm']['gamma']*config['runner']['algorithm']['lam'])**delay_steps
        evidence.append(dict(TRAIN=index, actual_start='Native first string-contact event, post-contact ball center/velocity; not a restored robot state.',
            actual_event_time_s=event['time_s'], actual_ball_center_m=position.tolist(),
            actual_contact_point_m=row['contact_point_m'], actual_outgoing_velocity_mps=velocity.tolist(),
            actual_first_landing_m=measured_landing.tolist(), actual_net_margin_m=row['net_margin_m'],
            actual_qualified_recovery=outcomes[index]['conditional_recovery'],
            model_baseline_first_landing_m=baseline['first_landing'][0].tolist(),
            model_baseline_landing_error_m=error, rk4_dt_halving_difference_m=convergence,
            calculated_existing_placement_bonus= float(landing_bonus),
            calculated_bonus_scope='Recomputed from the actual first landing and unchanged configured kernel; not an observed paid reward log.',
            measured_release_to_landing_control_steps=delay_steps, gae_direct_kernel_decay_to_landing=float(decay),
            counterfactual_same_speed_solutions=[target_solution(model, position, velocity, x, court, radius)
                for x in [4., 3.9, 3.5]]))
    air = np.array([outcome['airborne_contact'] for outcome in outcomes])
    attack = np.array([i in attacks for i in range(128)])
    recovery = np.array([outcome['conditional_recovery'] for outcome in outcomes])
    crossed = np.array([row['crossed_net'] for row in rows])
    target = np.array([row['landed_target'] for row in rows])
    sources = [Path(__file__), policy, protocol_path, court_path, physics_path, config_path,
        ROOT/'quadsmash/physics/shuttlecock.py', ROOT/'quadsmash/physics/court_v4.py',
        ROOT/'quadsmash/evaluation_v4.py', ROOT/'quadsmash/control/measured_joint_placement_reward_v4.py',
        *[run/name for name in ['job.json','INPUTS.json','episodes.json','contacts.json','metrics.json']]]
    result = dict(created_at=datetime.now(timezone.utc).isoformat(),
        scope='Completed frozen centered8 original TRAIN128 diagnosis, with separately labelled CPU free-flight counterfactuals. No policy, source contract, cases, physics or acceptance changed.',
        actual_frozen_funnel=dict(episodes=128, had_contact=sum(row['had_contact']for row in rows),
            legal_air=int(air.sum()), attack=int(attack.sum()), air_and_cross=int((air&crossed).sum()),
            air_and_target=int((air&target).sum()), air_and_recovery=int(recovery.sum()),
            attack_and_cross=int((attack&crossed).sum()), attack_and_target=int((attack&target).sum()),
            attack_and_recovery=int((attack&recovery).sum()), full=sum(outcome['complete']for outcome in outcomes)),
        actual_attack_diagnostics=evidence,
        limits=['Outgoing velocity rotations are hypothetical ball-flight calculations; they do not show PiPER can produce them or preserve real contact and recovery.',
            'Positive conservative far-face sphere clearance is a no-touch free-flight check. Negative clearance is not a native fault verdict; legal tape touches are allowed by the actual rules.',
            'Baseline RK4 is checked against measured native first landings. Error includes native contact duration, sphere contact offset and collision detection; it is not a new native assessment.',
            'The GAE kernel is direct residual propagation only; a learned value function can bridge long delays. This is not proof of a training failure cause.',
            'All128 cases are retained; three attacks are a complete event subgroup, not a filtered assessment or formal A/B result.'],
        table_sha256=sha(HERE/'ALL128_EVENT_BOTTLENECKS.csv'),
        source_sha256={str(path.relative_to(ROOT)):sha(path) for path in sources},
        native_job_started=False, formal_cases_read=False, formal_acceptance=False, goal_complete=False)
    (HERE/'REPORT.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(funnel=result['actual_frozen_funnel'], attacks=[
        dict(TRAIN=e['TRAIN'], baseline_error_m=e['model_baseline_landing_error_m'],
            placement_bonus=e['calculated_existing_placement_bonus'], targets=e['counterfactual_same_speed_solutions'])
        for e in evidence]), indent=2))


if __name__ == '__main__':
    main()
