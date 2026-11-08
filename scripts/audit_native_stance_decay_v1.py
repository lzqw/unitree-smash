"""Read-only audit of support loss after a state-qualified stance capture.

No counterfactual control, acceptance changes, or native success claims.
The physical-state record precedes the command at the same control tick.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pinocchio as pin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cases', type=int, nargs='+', required=True)
    a = p.parse_args()
    read = lambda f: json.loads(f.read_text())
    if a.output.exists() or read(a.run / 'job.json')['status'] != 'SUCCEEDED':
        raise ValueError('Fresh output and completed native evidence required')
    model = LoadedPinocchioModelV1()
    with np.load(a.run / 'pure_control_commands.npz') as z:
        arrays = {k: z[k].copy() for k in z.files}
    with np.load(a.run / 'pure_control_ground_forces.npz') as z:
        peaks = z['interval_peak_normal_n'].copy()
    names = read(a.run / 'pure_control_ground_forces.json')['body_names']
    nonfoot = [i for i, name in enumerate(names) if not name.endswith('_foot')]
    diagnostics = [json.loads(s) for s in
                   (a.run / 'pure_control_diagnostics.jsonl').read_text().splitlines()]
    if len(diagnostics) != len(arrays['tick']):
        raise ValueError('Unaligned state and command evidence')
    output = []
    for case in a.cases:
        ids = np.flatnonzero(arrays['lane'] == case)
        if not len(ids):
            raise ValueError('No causal ownership records for case')
        start = int(arrays['tick'][ids[0]])
        touched = np.any(peaks[start:, case][:, nonfoot] > 0., axis=1)
        end = start + int(np.flatnonzero(touched)[0]) if touched.any() else 800
        events = []
        for index in ids:
            index = int(index)
            tick = int(arrays['tick'][index])
            row = diagnostics[index]
            if (row['lane'], row['tick']) != (case, tick):
                raise ValueError('Mismatched native record')
            if tick > end:
                continue
            d = row['diagnostic']
            if not (d.get('terminal_candidate_now') or tick in (end - 1, end)):
                continue
            q, v = arrays['q'][index], arrays['v'][index]
            rotation = pin.Quaternion(q[3:7]).matrix()
            events.append(dict(tick=tick, state_precedes_command=True,
                stance_captured_now=d.get('terminal_candidate_now', False),
                stance_active=d['terminal_stance_active'],
                source=d['source'], accepted=d['solver_accepted'], rejection=d['rejection'],
                roll_pitch_rad=[float(np.arctan2(rotation[2, 1], rotation[2, 2])),
                    float(np.arcsin(np.clip(-rotation[2, 0], -1., 1.)))],
                velocity_world_m_s=(rotation @ v[:3]).tolist(),
                angular_speed_rad_s=float(np.linalg.norm(v[3:6])),
                max_joint_speed_rad_s=float(np.max(abs(v[model.v_indices]))),
                contacts=row['phase']['ground_contacts'],
                sampled_stable_s=row['phase']['control_sample_stable_duration_s']))
        output.append(dict(case=case, first_nonfoot_positive_force_tick=end if end < 800 else None,
            events=events))
    files = [Path(__file__)] + [a.run / n for n in ['job.json', 'pure_control_commands.npz',
        'pure_control_diagnostics.jsonl', 'pure_control_ground_forces.npz', 'pure_control_ground_forces.json']]
    files += [Path(m.__file__) for n, m in list(sys.modules.items())
              if n.startswith('scripts.') and getattr(m, '__file__', None)]
    data = dict(results=output, source_sha256={str(f.resolve()): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in files}, native_rerun=False, causality_proven=False, acceptance_changed=False)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = a.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(a.output)
    print(json.dumps(output))


if __name__ == '__main__':
    main()
