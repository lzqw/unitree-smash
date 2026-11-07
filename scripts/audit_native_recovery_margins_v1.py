"""Describe control-sampled recovery margins without replacing physics-rate scoring.

The 100 Hz public trace can identify which kinematic threshold is limiting.
It cannot prove uninterrupted physics-rate stability or an invariant region.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quadsmash.rl.episode_trace_v4 import unpack_public


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sampled_runs(mask, clock):
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
    return [dict(first_tick=int(s), last_tick=int(e), samples=int(e-s+1),
                 first_sample_s=float(clock[s]), last_sample_s=float(clock[e]),
                 sample_span_s=float(clock[e]-clock[s])) for s, e in zip(starts, ends)]


def inspect(run, case, start):
    read = lambda name: json.loads((run / name).read_text())
    inputs, metrics = read('INPUTS.json'), read('metrics.json')
    protocol_path = ROOT / 'configs/evaluation_protocol_v2.json'
    evaluation_path = ROOT / 'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json'
    if sha(evaluation_path) != inputs['protocol_sha256']:
        raise ValueError('Evaluation protocol differs from the executed experiment')
    snapshot = read('execution_source_hashes.json')
    checked_sources = [protocol_path, ROOT/'quadsmash/tasks/stand/observation_contract.py',
                       ROOT/'quadsmash/tasks/stand/jump_metrics.py',
                       ROOT/'quadsmash/tasks/stand/contact_flight_recovery_v5.py']
    for path in checked_sources:
        if snapshot.get(str(path.relative_to(ROOT))) != sha(path):
            raise ValueError('Metric or observation source differs from the native run')
    protocol = json.loads(protocol_path.read_text())['recovery']
    manifest = read('fixed_policy_traces/manifest.json')
    rows = [r for r in manifest['episodes'] if r['source_training_index'] == case]
    if len(rows) != 1:
        raise ValueError('Need one ground-start first-episode trace')
    row = rows[0]
    path = run / 'fixed_policy_traces' / row['data_file']
    if sha(path) != row['data_sha256']:
        raise ValueError('Native trace hash mismatch')
    with np.load(path, allow_pickle=False) as z:
        obs = unpack_public(z['packed_public']).astype(float)
    if len(obs) != row['samples'] or not 0 <= start < len(obs):
        raise ValueError('Invalid trace or requested diagnostic interval')
    clock = obs[:, 775]
    if not np.isfinite(obs).all() or not np.allclose(clock, np.arange(len(obs))*.01, atol=2e-6, rtol=0):
        raise ValueError('Need finite continuous native 100 Hz samples from ground')
    gravity = obs[:, 6:9].copy()
    gravity_norm = np.linalg.norm(gravity, axis=1)
    if np.max(abs(gravity_norm-1.)) > 1e-4:
        raise ValueError('Invalid projected unit gravity')
    gravity /= gravity_norm[:, None]
    roll = np.degrees(np.arctan2(-gravity[:, 1], -gravity[:, 2]))
    pitch = np.degrees(np.arcsin(np.clip(gravity[:, 0], -1., 1.)))
    # World horizontal speed is invariant to yaw. Remove the vertical
    # projection from the body-frame ROOT-LINK velocity in public[0:3].
    velocity = obs[:, :3]
    horizontal = np.sqrt(np.maximum(0., np.sum(velocity**2, axis=1)
                                     - np.sum(velocity*gravity, axis=1)**2))
    values = dict(max_abs_roll_pitch_deg=np.maximum(abs(roll), abs(pitch)),
        horizontal_root_speed_m_s=horizontal, angular_speed_rad_s=np.linalg.norm(obs[:, 3:6], axis=1),
        maximum_joint_speed_rad_s=np.max(abs(obs[:, 27:45]), axis=1),
        supporting_foot_count=(obs[:, 57:61] > .5).sum(axis=1))
    conditions = dict(
        attitude=values['max_abs_roll_pitch_deg'] <= protocol['maximum_abs_roll_pitch_deg'],
        horizontal_speed=horizontal <= protocol['maximum_horizontal_root_speed_m_s'],
        angular_speed=values['angular_speed_rad_s'] <= protocol['maximum_root_angular_speed_rad_s'],
        joint_speed=values['maximum_joint_speed_rad_s'] <= protocol['maximum_abs_joint_velocity_rad_s'],
        support=values['supporting_foot_count'] > 0)
    all_kinematic = np.logical_and.reduce(list(conditions.values()))
    selected = np.arange(len(obs)) >= start
    episodes = read('episodes.json')
    episode = next(r for r in episodes if r['source_training_index'] == case)
    last = {k: float(v[-1]) for k, v in values.items()}
    last.update(roll_deg=float(roll[-1]), pitch_deg=float(pitch[-1]),
                active_kinematic_failures=[k for k, v in conditions.items() if not v[-1]])
    summaries = []
    for name, begin in [('selected_interval', start), ('last_one_second', max(start, len(obs)-100))]:
        mask = np.arange(len(obs)) >= begin
        summaries.append(dict(name=name, first_tick=begin, last_tick=len(obs)-1,
            samples=int(mask.sum()),
            extrema={k: dict(min=float(v[mask].min()), max=float(v[mask].max())) for k, v in values.items()},
            failing_sample_counts={k: int((mask & ~v).sum()) for k, v in conditions.items()},
            sole_limiting_sample_counts={k: int((mask & ~v & np.logical_and.reduce(
                [w for j, w in conditions.items() if j != k])).sum()) for k, v in conditions.items()}))
    official = next(r for r in metrics['complete5_case_outcomes'] if r['training_index'] == case)
    source_files = [path, evaluation_path, run/'execution_source_hashes.json', run/'fixed_policy_traces/manifest.json',
        run/'INPUTS.json', run/'metrics.json', run/'episodes.json', Path(__file__),
        *checked_sources]
    return dict(run=str(run.resolve()), case=case, recovery_protocol=protocol,
        official_case_result=official, official_contact_flight=episode['contact_flight_recovery'],
        control_sampled_kinematic_runs=sampled_runs(selected & all_kinematic, clock),
        last_sample=last, intervals=summaries,
        thresholds_are_original=True, substitutes_for_physics_rate_score=False,
        guarantees_between_samples=False, new_native_execution=False,
        valid_expert_labels=False, source_sha256={str(p.resolve()): sha(p) for p in source_files})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, action='append', required=True)
    parser.add_argument('--case', type=int, required=True)
    parser.add_argument('--start-tick', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    if a.output.exists():
        raise ValueError('Preserve previous evidence')
    results = [inspect(run, a.case, a.start_tick) for run in a.run]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as stream:
        stream.write(json.dumps(dict(scope=__doc__, results=results), indent=2, allow_nan=False)+'\n')
    print(json.dumps([dict(run=r['run'], last_sample=r['last_sample'],
                          sampled_runs=r['control_sampled_kinematic_runs']) for r in results], indent=2))


if __name__ == '__main__':
    main()
