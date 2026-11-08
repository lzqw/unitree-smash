"""Check observer passivity and attribute sampled normal forces near divergence."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np


def audit(run, baseline):
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    read = lambda p: json.loads(Path(p).read_text())
    for folder in [run, baseline]:
        job = read(folder / 'job.json')
        if job['status'] != 'SUCCEEDED' or job['exit_code'] != 0:
            raise ValueError('Completed native runs required')
    meta = read(run / 'body_pair_observation.json')
    data_path = run / 'body_pair_observation.npz'
    if sha(data_path) != meta['data_sha256']:
        raise ValueError('Changed pair-force data')
    requested, backend = meta['requested_filter_paths'], meta['backend_filter_paths']
    filters_ok = backend == requested or (len(backend) == 56 and all(x == requested for x in backend))
    replay = []
    pins = {str(p.resolve()): sha(p) for p in [Path(__file__), data_path,
        run / 'body_pair_observation.json', run / 'job.json', baseline / 'job.json']}
    for case in range(128):
        name = f'fixed_policy_traces/case_{case:03d}.npz'
        row = dict(case=case)
        for folder in [run, baseline]:
            pins[str((folder / name).resolve())] = sha(folder / name)
        with np.load(run / name, allow_pickle=False) as a, np.load(baseline / name, allow_pickle=False) as b:
            for key in ['packed_public', 'dispatched_actions']:
                row[key] = dict(shape_equal=a[key].shape == b[key].shape,
                    exactly_equal=np.array_equal(a[key], b[key]),
                    max_error=float(np.max(abs(a[key] - b[key]))) if a[key].shape == b[key].shape else None)
        replay.append(row)
    same = all(r[k]['exactly_equal'] for r in replay for k in ['packed_public', 'dispatched_actions'])
    events, around = [], []
    lane = lambda path: int(re.search(r'/env_(\d+)/', path).group(1)) if '/env_' in path else None
    with np.load(data_path, allow_pickle=False) as z:
        frames = z['physics_frame'].copy()
        if not np.array_equal(frames, np.arange(89600, 92001)):
            raise ValueError('Every physics frame in the declared window is required')
        force = z['normal_force_world_n']
        if force.shape != (2401, 56, 57, 3) or not np.isfinite(force).all():
            raise ValueError('Invalid pair-force matrix')
        norms = np.linalg.norm(force, axis=-1)
        if filters_ok:
            for i, j in np.argwhere(norms.max(0) > 1.):
                values = norms[:, i, j]; ids = np.flatnonzero(values > 1.)
                sensor, partner = meta['sensor_paths'][i], requested[j]
                events.append(dict(sensor=sensor, partner=partner,
                    inter_environment=lane(partner) is not None and lane(sensor) != lane(partner),
                    first_frame_over_1n=int(frames[ids[0]]), peak_normal_force_n=float(values.max()),
                    sample_count_over_1n=len(ids)))
            # The exact first divergent frame and its immediate neighbours.
            for frame in range(90568, 90573):
                f = norms[frame - 89600]
                pairs = [dict(sensor=meta['sensor_paths'][i], partner=requested[j],
                    normal_force_world_n=force[frame - 89600, i, j].tolist(),
                    norm_n=float(f[i, j])) for i, j in np.argwhere(f > 1e-6)]
                around.append(dict(physics_frame=frame, pairs=pairs))
            cross = [(i, j) for i, s in enumerate(meta['sensor_paths']) for j, p in enumerate(requested)
                if lane(p) is not None and lane(s) != lane(p)]
            cross_peak = max(float(norms[:, i, j].max()) for i, j in cross)
        else:
            cross_peak = None
    for path, expected in pins.items():
        if sha(path) != expected:
            raise ValueError('Evidence changed during audit: ' + path)
    return dict(scope=__doc__, observer_exact_replay=same, filter_paths_verified=filters_ok,
        complete_recording=meta['complete_window'], case_replay=replay,
        pairs_over_1n=events, divergence_frame_neighbourhood=around,
        maximum_inter_environment_normal_force_n=cross_peak, source_sha256=pins,
        physical_cause_identified=False, valid_expert_labels=False,
        limitations=['Pair-force interpretation requires exact replay and backend filter verification.',
            'Only normal forces for the specified bodies and 5.60–5.75 second window are measured.',
            'Same-frame forces do not establish causal order within the physics solver.',
            'Observer passivity does not replace the original strict cohort-isolation audit.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['run', 'baseline', 'output']:
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve existing evidence')
    result = audit(args.run, args.baseline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ['observer_exact_replay', 'filter_paths_verified',
        'complete_recording', 'maximum_inter_environment_normal_force_n']}))
    if not all(result[k] for k in ['observer_exact_replay', 'filter_paths_verified', 'complete_recording']):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
