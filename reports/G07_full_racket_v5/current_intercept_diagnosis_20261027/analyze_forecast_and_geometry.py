"""Separate causal-ball forecast error from native racket tracking error.

Use all128 original cases from two sealed native executions. Interpolated
geometry is diagnostic only and never replaces native contact/score flags.
Only segments whose BOTH endpoints precede first racket contact are used.
"""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import torch

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = HERE.parents[2].parent/'unitree_smash_contact_recovery'
sys.path.insert(0, str(ROOT))
from quadsmash.control.initial_intercept import observed_initial_ball
from quadsmash.control.full_racket_train_evaluation_v4 import training_cases
from quadsmash.control.full_racket_body_search_v4 import summarize_candidate
from quadsmash.evaluation_v4 import independent_episode
from quadsmash.physics.full_racket_v4 import load_racket, string_prism_gap


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    a = np.asarray(values, dtype=float)
    assert a.size and np.isfinite(a).all()
    return dict(min=float(a.min()), median=float(np.median(a)), max=float(a.max()),
                p10=float(np.quantile(a, .1)), p90=float(np.quantile(a, .9)))


def integrate(position, velocity, duration, physics, max_dt):
    """RK4 flight with the original constant wind/quadratic drag."""
    position, velocity = position.copy(), velocity.copy()
    duration = np.asarray(duration)
    assert np.isfinite(duration).all() and (duration > 0).all()
    steps = int(np.ceil(duration.max()/max_dt))
    dt = duration[:, None]/steps
    wind = np.asarray(physics['wind_m_s'])
    drag = physics['quadratic_drag_k_per_m']
    def acceleration(v):
        relative = v-wind
        return np.array([0., 0., -9.81])-drag*np.linalg.norm(relative, axis=1, keepdims=True)*relative
    for _ in range(steps):
        a1 = acceleration(velocity)
        v2 = velocity+.5*dt*a1
        a2 = acceleration(v2)
        v3 = velocity+.5*dt*a2
        a3 = acceleration(v3)
        v4 = velocity+dt*a3
        a4 = acceleration(v4)
        position += dt*(velocity+2*v2+2*v3+v4)/6
        velocity += dt*(a1+2*a2+2*a3+a4)/6
    return position, velocity


def main():
    torch.set_num_threads(2)
    policy_path = BASE/'early_lateral7_n128_01/best_candidate.pt'
    expected = read(BASE/'early_lateral_completed_20261027/ACTUAL_INDEPENDENT128_CHECK.json')
    assert sha(policy_path) == expected['selected_policy_sha256']
    factory_path = BASE/'windowed_swing_20261027/window_controller.py'
    spec = importlib.util.spec_from_file_location('actual_intercept_source_factory', factory_path)
    factory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(factory)
    source = torch.load(policy_path, map_location='cpu', weights_only=False)
    controller = factory.make_source(source, 128, 'cpu')
    offset = controller.reference_controller.teacher.contact_offset.cpu().numpy().copy()
    racket_config = load_racket()
    physics_path = ROOT/'configs/shuttle_physics_v2.json'
    physics = read(physics_path)
    protocol_path = ROOT/'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json'
    protocol = read(protocol_path)
    protocol['_root'] = str(ROOT)
    cases, indices = training_cases(ROOT, protocol)
    assert indices == list(range(128))
    files = [Path(__file__), policy_path, factory_path, physics_path, protocol_path,
        ROOT/'quadsmash/control/initial_intercept.py', ROOT/'quadsmash/physics/full_racket_v4.py',
        ROOT/'configs/full_racket_v4_r3.json', ROOT/'quadsmash/evaluation_v4.py']
    reports = []
    for name, candidate in [('early_lateral7_best_cold128_01', 1), ('windowed_swing7_n128_01', 3)]:
        run = BASE/name
        suffix = f'{candidate:03d}'
        paths = [run/f'episodes_{suffix}.json', run/f'causal_plans_{suffix}.json',
                 run/f'all_first_episodes_{suffix}.npz', run/'geometry.json', run/'INPUTS.json',
                 run/f'initial_ground_states_{suffix}.json']
        hashes = {str(p): sha(p) for p in paths}
        files.extend(paths)
        rows, plans = [read(p) for p in paths[:2]]
        inputs, initial = read(paths[4]), read(paths[5])
        assert inputs['origin_spacing_m'] == 0 and inputs['training_indices'] == indices
        assert len(rows) == 128 and [x['case_id'] for x in rows] == [x['case_id'] for x in cases]
        assert [x['source_training_index'] for x in rows] == indices
        assert [x['episode_index'] for x in rows] == initial['episode_ids']
        assert initial['all_clocks_zero'] and initial['actual_environment_origins_m'] == [[0., 0., 0.]]*128
        assert all(x['ground_init'] and x['env_id'] == i for i, x in enumerate(rows))
        summary = summarize_candidate(rows, protocol)
        rescored = [independent_episode(x, protocol) for x in rows]
        assert summary['airborne_contacts'] == sum(x['airborne_contact'] for x in rescored)
        assert summary['complete'] == sum(x['complete'] for x in rescored)
        assert summary['conditional_recovery_count'] == sum(x['conditional_recovery'] for x in rescored)
        plan = plans[0]
        assert plan['env_ids'] == indices and all(plan['valid'])
        public = np.asarray(plan['used_public_observations'])
        assert public.shape == (128, 794) and (public[:, 773] < .5).all()
        assert (public[:, 67] > .5).all() and np.allclose(public[:, 775], plan['observation_clock_s'])
        p0, v0 = observed_initial_ball(public, [1.10, 0., 1.80], [1., 0., 0.])
        assert np.max(abs(p0-np.asarray(plan['initial_position_m']))) < 1e-12
        assert np.max(abs(v0-np.asarray(plan['initial_velocity_mps']))) < 1e-12
        data = np.load(paths[2])
        poses, balls, clocks, valid = [data[k] for k in ['link_poses', 'ball_position_m',
            'episode_clock_s', 'valid_first_episode']]
        assert poses.shape == (400, 128, 28, 7)
        tool = read(paths[3])['body_names'].index('Racket')
        actual_time, actual_ball = [], []
        for lane, row in enumerate(rows):
            first_contact = row['contact_time_s'] if np.isfinite(row['contact_time_s']) else np.inf
            eligible = np.flatnonzero(valid[:, lane] & (clocks[:, lane] >= 2.50)
                & (clocks[:, lane] < first_contact-.01))
            assert len(eligible), (name, lane, row['contact_time_s'])
            tick = eligible[np.argmin(abs(clocks[eligible, lane]-2.60))]
            assert abs(float(clocks[tick, lane])-2.60) < .012
            actual_time.append(clocks[tick, lane])
            actual_ball.append(balls[tick, lane])
        duration = np.asarray(actual_time)-public[:, 775]
        predicted, _ = integrate(p0, v0, duration, physics, .001)
        refined, _ = integrate(p0, v0, duration, physics, .0005)
        refinement = float(np.linalg.norm(predicted-refined, axis=1).max())
        assert refinement < 1e-7
        forecast_error = predicted-np.asarray(actual_ball)
        geometry_rows = []
        fraction = np.linspace(0., 1., 11)
        with torch.inference_mode():
            for lane, row in enumerate(rows):
                contact = row['contact_time_s']
                end = min(3.25, contact-.001) if np.isfinite(contact) else 3.25
                segments = np.flatnonzero(valid[:-1, lane] & valid[1:, lane]
                    & (clocks[:-1, lane] >= 2.3) & (clocks[1:, lane] <= end)
                    & (clocks[1:, lane] > clocks[:-1, lane]))
                assert len(segments)
                # Each interpolated segment is wholly before the first
                # native racket contact: no post-impact endpoint is used.
                a = poses[segments, lane, tool].astype(float)
                b = poses[segments+1, lane, tool].astype(float)
                times = clocks[segments, lane, None]+fraction*(clocks[segments+1, lane, None]-clocks[segments, lane, None])
                center = a[:, None, :3]+fraction[None, :, None]*(b-a)[:, None, :3]
                ball = balls[segments, lane, None]+fraction[None, :, None]*(balls[segments+1, lane]-balls[segments, lane])[:, None]
                qa, qb = a[:, 3:], b[:, 3:]
                qb = np.where((qa*qb).sum(1, keepdims=True) < 0, -qb, qb)
                quat = qa[:, None]+fraction[None, :, None]*(qb-qa)[:, None]
                quat /= np.linalg.norm(quat, axis=2, keepdims=True)
                times, center, ball, quat = times.ravel(), center.reshape(-1, 3), ball.reshape(-1, 3), quat.reshape(-1, 4)
                keep = (ball[:, 0] < 1.5) & (ball[:, 2] >= 1.3)
                assert keep.any(), (name, lane)
                times, center, ball, quat = times[keep], center[keep], ball[keep], quat[keep]
                gap = string_prism_gap(torch.as_tensor(ball), torch.as_tensor(center),
                    torch.as_tensor(quat), racket_config, physics['collision_radius_m']).numpy()
                index = int(np.argmin(gap))
                rotation = Rotation.from_quat(quat[index, [1, 2, 3, 0]])
                point = center[index]+rotation.apply(offset)
                geometry_rows.append(dict(TRAIN_index=lane, native_episode_id=row['episode_index'],
                    native_airborne_contact=bool(rescored[lane]['airborne_contact']),
                    native_any_racket_contact=row['had_contact'], native_frame_hit=row['frame_hit'],
                    native_contact_time_s=contact, approximate_closest_precontact_time_s=float(times[index]),
                    approximate_string_gap_m=float(gap[index]),
                    approximate_ball_minus_material_point_world_m=(ball[index]-point).tolist(),
                    approximate_ball_local_racket_m=rotation.inv().apply(ball[index]-center[index]).tolist(),
                    forecast_check_clock_s=float(actual_time[lane]), forecast_error_world_m=forecast_error[lane].tolist()))
        missed = [x for x in geometry_rows if not x['native_airborne_contact']]
        delta = np.asarray([x['approximate_ball_minus_material_point_world_m'] for x in missed])
        reports.append(dict(run=name, candidate_id=candidate, original128_summary=summary,
            source_policy_sha256=inputs['source_policy_sha256'],
            actual_air_TRAIN_indices=[x['TRAIN_index'] for x in geometry_rows if x['native_airborne_contact']],
            forecast_checked_all128_before_any_first_racket_contact=True,
            forecast_position_error_norm_m=stats(np.linalg.norm(forecast_error, axis=1)),
            forecast_absolute_axis_error_m={axis: stats(abs(forecast_error[:, i])) for i, axis in enumerate('xyz')},
            RK4_step_halving_max_position_difference_m=refinement,
            non_air_case_count=len(missed), non_air_signed_ball_minus_material_point_m={axis: stats(delta[:, i]) for i, axis in enumerate('xyz')},
            non_air_negative_y_count=int((delta[:, 1] < 0).sum()),
            non_air_absolute_y_over010m_count=int((abs(delta[:, 1]) > .10).sum()),
            closest_precontact_gap_m=stats([x['approximate_string_gap_m'] for x in missed]), rows=geometry_rows))
        assert all(sha(Path(p)) == h for p, h in hashes.items())
        print(json.dumps({k: v for k, v in reports[-1].items() if k != 'rows'}), flush=True)
    first, second = [set(r['actual_air_TRAIN_indices']) for r in reports]
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), results=reports,
        actual_air_gained_TRAIN_indices=sorted(second-first), actual_air_lost_TRAIN_indices=sorted(first-second),
        original_distinct_cases=128, native_executed_episodes_checked=256,
        geometry_interpolation_from50Hz_poses=True, interpolation_is_not_native_contact_evidence=True,
        interpolation_both_endpoints_before_first_native_racket_contact=True,
        new_native_execution=False, controller_changed=False, native_scores_changed=False,
        formal_cases_read=False, formal_acceptance=False, github_upload_performed=False,
        source_sha256={str(p): sha(p) for p in sorted(set(files))})
    with (HERE/'ACTUAL_FORECAST_AND_MISS_CHECK.json').open('x') as f:
        f.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(gained=result['actual_air_gained_TRAIN_indices'], lost=result['actual_air_lost_TRAIN_indices'])), flush=True)


if __name__ == '__main__':
    main()
