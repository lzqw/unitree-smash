"""Diagnose all original128 source misses using recorded native geometry.

The2ms normalized-quaternion interpolation is a geometric diagnostic only;
original native rows alone define actual contacts and success. No formal cases
or running source files are changed.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import torch

MAIN = Path(__file__).resolve().parents[3]
ROOT = MAIN.parent / 'unitree_smash_contact_recovery'
HERE = Path(__file__).resolve().parent
SOURCE = MAIN / 'reports/G07_full_racket_v5/teacher_contact_recovery_cold128_02'
POLICY = MAIN / 'reports/G07_full_racket_v4/gain_body8_sequential_n128_01/best_candidate.pt'
sys.path.insert(0, str(ROOT))
from quadsmash.control.full_racket_origin0_residual_v4 import make_rally_controller
from quadsmash.physics.full_racket_v4 import load_racket, string_prism_gap


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    a = np.asarray(values)
    return dict(min=float(a.min()), median=float(np.median(a)), max=float(a.max()),
                p10=float(np.quantile(a, .1)), p90=float(np.quantile(a, .9)))


def main():
    read = lambda name: json.loads((SOURCE / name).read_text())
    assert read('job.json')['status'] == 'SUCCEEDED'
    assert read('INPUTS.json')['training_indices'] == list(range(128))
    rows = read('episodes.json')
    assert len(rows) == 128
    model = load_racket()
    policy = torch.load(POLICY, map_location='cpu', weights_only=False)
    controller = make_rally_controller(policy, 128, 'cpu')
    reference = controller.reference_controller
    offset = reference.teacher.contact_offset.numpy().copy()
    tool = read('geometry.json')['body_names'].index('Racket')
    with np.load(SOURCE / 'all_first_episodes.npz') as data:
        poses, balls, clock, valid = [data[key].copy() for key in
            ['link_poses', 'ball_position_m', 'episode_clock_s', 'valid_first_episode']]
    assert poses.shape == (400, 128, 28, 7)
    results = []
    with torch.inference_mode():
        for lane, row in enumerate(rows):
            assert lane == row['env_id'] == row['source_training_index'] and row['episode_index'] == 0
            contact = row['contact_time_s']
            end = min(3.25, contact) if np.isfinite(contact) else 3.25
            indices = np.flatnonzero(valid[:-1, lane] & valid[1:, lane] &
                (clock[:-1, lane] >= 2.3) & (clock[:-1, lane] <= end) &
                (clock[1:, lane] > clock[:-1, lane]))
            assert len(indices)
            fraction = np.linspace(0., 1., 11)
            values = []
            for i in indices:
                time = clock[i, lane] + fraction * (clock[i + 1, lane] - clock[i, lane])
                center = poses[i, lane, tool, :3] + fraction[:, None] * (poses[i + 1, lane, tool, :3] - poses[i, lane, tool, :3])
                ball = balls[i, lane] + fraction[:, None] * (balls[i + 1, lane] - balls[i, lane])
                qa, qb = poses[i, lane, tool, 3:].astype(float), poses[i + 1, lane, tool, 3:].astype(float)
                if qa @ qb < 0:
                    qb = -qb
                quat = qa + fraction[:, None] * (qb - qa)
                quat /= np.linalg.norm(quat, axis=1, keepdims=True)
                keep = (time <= end + 1e-7) & (ball[:, 0] < 1.5) & (ball[:, 2] >= 1.3)
                if keep.any():
                    gap = string_prism_gap(torch.tensor(ball[keep], dtype=torch.float64),
                        torch.tensor(center[keep], dtype=torch.float64),
                        torch.tensor(quat[keep], dtype=torch.float64), model, .025).numpy()
                    for t, c, p, q, g in zip(time[keep], center[keep], ball[keep], quat[keep], gap):
                        hit_point = c + Rotation.from_quat(q[[1, 2, 3, 0]]).apply(offset)
                        values.append((float(g), float(t), c, p, hit_point, q))
            assert values
            gap, time, center, ball, point, quaternion = min(values, key=lambda item: item[0])
            results.append(dict(source_training_index=lane, native_airborne_string_hit=row['airborne_string_hit'],
                native_any_racket_contact=row['had_contact'], native_frame_hit=row['frame_hit'],
                native_contact_time_s=contact, geometric_closest_time_s=time,
                approximate_string_gap_m=gap, actual_recorded_interpolated_ball_m=ball.tolist(),
                actual_recorded_interpolated_string_center_m=center.tolist(),
                actual_recorded_interpolated_contact_target_point_m=point.tolist(),
                ball_minus_contact_point_m=(ball - point).tolist(),
                normal_world=Rotation.from_quat(quaternion[[1, 2, 3, 0]]).apply([1., 0., 0.]).tolist()))
    no_string = [row for row in results if not row['native_airborne_string_hit']]
    delta = np.array([row['ball_minus_contact_point_m'] for row in no_string])
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), original128_cases_checked=True,
        native_airborne_contacts=sum(row['native_airborne_string_hit'] for row in results),
        missed_air_string_cases=len(no_string),
        missed_signed_ball_minus_material_hit_point_m={axis: stats(delta[:, i]) for i, axis in enumerate('xyz')},
        missed_absolute_ball_minus_material_hit_point_m={axis: stats(abs(delta[:, i])) for i, axis in enumerate('xyz')},
        missed_ball_left_of_material_target=int(sum(delta[:, 1] < 0)),
        missed_lateral_error_over_010m=int(sum(abs(delta[:, 1]) > .10)),
        geometric_interpolation_seconds=.002, interpolated_states_not_native_substeps=True,
        actual_native_contact_flags_not_replaced_by_geometric_gap=True,
        material_contact_offset_in_racket_m=offset.tolist(),
        source_lateral_feedback_gate='Public ball present, begins at event-.25s, ramps over.12s, ends event+.03s.',
        source_sha256={str(path): sha(path) for path in
            [POLICY, SOURCE / 'all_first_episodes.npz', SOURCE / 'episodes.json', SOURCE / 'geometry.json',
             ROOT / 'configs/full_racket_v4_r3.json', ROOT / 'quadsmash/control/full_racket_teacher_v4.py', Path(__file__)]},
        rows=results, new_native_execution=False, formal_cases_read=False, formal_acceptance=False)
    serialized = json.dumps(result, indent=2) + '\n'
    with (HERE / 'ALL128_MISS_GEOMETRY.json').open('x') as stream:
        stream.write(serialized)
    print(json.dumps({key: value for key, value in result.items() if key not in ['rows', 'source_sha256']}))


if __name__ == '__main__':
    main()
