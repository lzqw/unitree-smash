"""Audit passive filtered force coverage, consistency and original execution."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_timed_impact_observation_v2 import audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['run', 'parent', 'bundle', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Fresh output required')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    dispatch_path = a.output.with_name('dispatch_audit.json')
    subprocess.run([sys.executable, str(ROOT / 'scripts/audit_native_experimental_recovery_v1.py'),
        '--run', str(a.run), '--bundle', str(a.bundle), '--output', str(dispatch_path)], check=True)
    read = lambda p: json.loads(Path(p).read_text())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    replay = audit(a.run.resolve(), a.parent.resolve())
    replay_path = a.output.with_name('passive_replay_audit.json')
    replay_path.write_text(json.dumps(replay, indent=2) + '\n')
    meta = read(a.run / 'filtered_contact_observation.json')
    if not meta['high_rate_complete'] or not meta['panel_complete'] or meta['lanes'] != [85, 87]:
        raise ValueError('Incomplete filtered observation')
    arrays = {}
    for stem in ['filtered_ground_panel', 'filtered_foot_pairs', 'foot_impact_observation']:
        path = a.run / (stem + '.npz')
        if stem != 'foot_impact_observation' and sha(path) != meta['data_sha256'][path.name]:
            raise ValueError('Filtered data hash mismatch')
        with np.load(path, allow_pickle=False) as z:
            arrays[stem] = {k: z[k].copy() for k in z.files}
        if any(not np.isfinite(v).all() for v in arrays[stem].values()):
            raise ValueError('Nonfinite observation')
    panel, pairs, original = [arrays[n] for n in ['filtered_ground_panel', 'filtered_foot_pairs', 'foot_impact_observation']]
    if (not np.array_equal(panel['physics_frame'], np.arange(160, 128001, 160))
            or not np.array_equal(pairs['physics_frame'], original['physics_frame'])
            or pairs['pair_buffer_invalid'].any()):
        raise ValueError('Filtered coverage or native buffer violation')
    ground = panel['ground_normal_force_world_n']
    net = panel['net_normal_force_world_n']
    if ground.shape != (800, 128, 4, 3) or net.shape != ground.shape:
        raise ValueError('Wrong panel force dimensions')
    first = panel['episode_index'] == 0
    unsupported = panel['observed_contacts'] & (np.linalg.norm(ground, axis=-1) < .01) & first[..., None]
    # Body origin is at most 2 mm from the sphere center; radius is 22 mm.
    # This conservative bound is a reporting bin, not a new control threshold.
    above = panel['foot_position_world'][..., 2] > .034
    rows = [dict(case=c, first_episode_sample_count=int(first[:, c].sum()),
        observed_without_ground_001n_foot_samples=int(unsupported[:, c].sum()),
        observed_without_ground_and_sphere_at_least_1cm_high_foot_samples=int((unsupported[:, c] & above[:, c]).sum()))
        for c in range(128)]
    alignment = []
    for tick in range(288, 301):
        hi = (tick - 288) * 160
        x = ground[tick - 1, meta['lanes']]
        y = pairs['pair_normal_force_world_n'][hi, :, :, 0]
        error = float(np.max(abs(x - y)))
        alignment.append(dict(tick=tick, error_n=error,
            passed=bool(np.allclose(x, y, atol=1e-3, rtol=1e-5))))
    normal = pairs['pair_normal_force_world_n']
    foot_pair_error = 0.
    for i in range(4):
        for j in range(i + 1, 4):
            a_filter = 1 + [k for k in range(4) if k != i].index(j)
            b_filter = 1 + [k for k in range(4) if k != j].index(i)
            foot_pair_error = max(foot_pair_error, float(np.max(abs(normal[:, :, i, a_filter] + normal[:, :, j, b_filter]))))
    residual = original['foot_force_world'] - normal.sum(3)
    # Pair views intentionally exclude other robot links and racket contacts.
    selected = []
    for frame in [46791, 46792, 47061, 47087, 47090]:
        i = int(np.flatnonzero(pairs['physics_frame'] == frame)[0])
        selected.append(dict(physics_frame=frame,
            observed_contacts=original['observed_contacts'][i, 0].tolist(),
            normal_by_foot_and_filter=normal[i, 0].tolist(),
            friction_by_foot_and_filter=pairs['pair_friction_force_world_n'][i, 0].tolist(),
            other_or_unexplained_normal_n=residual[i, 0].tolist()))
    dispatch = read(dispatch_path)
    pins = dict(replay['source_sha256'])
    for path in [Path(__file__), ROOT / 'scripts/native_ground_pair_observer_v1.py', dispatch_path,
            a.run / 'filtered_contact_observation.json', a.run / 'filtered_ground_panel.npz', a.run / 'filtered_foot_pairs.npz']:
        pins[str(path.resolve())] = sha(path)
    passed = replay['passed'] and all(x['passed'] for x in alignment)
    result = dict(dispatch, filtered_observation_passed=passed,
        passed_dispatch_integrity=dispatch['passed_dispatch_integrity'] and passed,
        passive_parent_replay=replay['passive_parent_replay'], panel_cases=rows,
        ground_view_alignment=alignment, foot_pair_normal_antisymmetry_max_n=foot_pair_error,
        maximum_other_or_unexplained_normal_n=float(np.max(abs(residual))),
        selected_case85_frames=selected, source_sha256=pins,
        note='Original scoring unchanged. Normal forces are separate from measured friction; no label automatically becomes an expert.')
    tmp = a.output.with_suffix('.tmp'); tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n'); tmp.replace(a.output)
    print(json.dumps({k: result[k] for k in ['complete5', 'filtered_observation_passed', 'passive_parent_replay',
        'foot_pair_normal_antisymmetry_max_n', 'maximum_other_or_unexplained_normal_n']}))
    if not passed:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
