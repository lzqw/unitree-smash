"""Compare authored foot-sphere separation with recorded normal-force pairs.

This is retrospective geometry/force evidence, not a contact-pair measurement
or an online controller. Opposite forces alone do not identify collider pairs.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pinocchio as pin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.check_constrained_recovery_suffix_v1 import source_context
from scripts.search_preserved_early_support_v1 import initialize
from scripts.search_landing_support_tail_v3 import _WORKER
from scripts.native_recovery_query_bank_v1 import state_from_sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['plan', 'run', 'prediction', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Fresh output required')
    read = lambda p: json.loads(Path(p).read_text())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    prediction = read(args.prediction)
    pins = dict(prediction['source_sha256'])
    for p, h in pins.items():
        if sha(p) != h:
            raise ValueError('Changed evidence ' + p)
    setup = read(args.plan)
    source = read(setup['source'])
    *_, context = source_context(source['configuration']['source'])
    initialize(*context)
    model = _WORKER['controller'].qp.model
    spheres = _WORKER['spheres']
    meta = read(args.run / 'foot_impact_observation.json')
    with np.load(args.run / 'foot_impact_observation.npz', allow_pickle=False) as z:
        data = {k: z[k].copy() for k in z.files}
    predictions = {(r['case'], r['physics_frame']): r for r in prediction['rows']}
    records = []
    summaries = {}
    for lane, case in enumerate(meta['lanes']):
        minimum = np.inf
        close_count = force_count = 0
        for i, frame in enumerate(data['physics_frame']):
            sample = {k: v[i, lane] for k, v in data.items() if k != 'physics_frame'}
            state = state_from_sample(model, sample, meta['origins_world'][lane])
            pin.forwardKinematics(model.model, model.data, state.q)
            pin.updateFramePlacements(model.model, model.data)
            centers = np.array([model.data.oMf[model.body_frames[s['body']]].act(
                np.asarray(s['center_body_m'])) for s in spheres])
            forces = data['foot_force_world'][i, lane]
            pairs = []
            for a, b in itertools.combinations(range(4), 2):
                delta = centers[a] - centers[b]
                distance = float(np.linalg.norm(delta))
                gap = distance - spheres[a]['radius_m'] - spheres[b]['radius_m']
                minimum = min(minimum, gap)
                horizontal_a, horizontal_b = forces[a, :2], forces[b, :2]
                strength = min(float(np.linalg.norm(horizontal_a)), float(np.linalg.norm(horizontal_b)))
                mismatch = float(np.linalg.norm(horizontal_a + horizontal_b))
                opposite = strength > 1. and mismatch < .01 * strength
                close = gap < .001  # reporting bin only, never a native contact threshold
                close_count += int(close)
                force_count += int(opposite)
                if close or opposite:
                    normal = delta / distance if distance > 1e-12 else np.zeros(3)
                    horizontal_normal = normal[:2]
                    denom = np.linalg.norm(horizontal_normal) * np.linalg.norm(horizontal_a)
                    alignment = float(horizontal_normal @ horizontal_a / denom) if denom > 1e-12 else None
                    pairs.append(dict(feet=[model.feet[a], model.feet[b]], gap_m=gap,
                        normal_from_second_to_first=normal.tolist(),
                        forces_world_n=[forces[a].tolist(), forces[b].tolist()],
                        equal_opposite_horizontal=opposite, horizontal_mismatch_n=mismatch,
                        horizontal_normal_alignment=alignment))
            previous = predictions.get((case, int(frame) - 1), {})
            if pairs or previous.get('maximum_joint_velocity_error_rad_s', 0.) > 1.:
                records.append(dict(case=case, physics_frame=int(frame),
                    time_s=float(frame * meta['physics_dt_s']), pairs=pairs,
                    preceding_joint_velocity_error_rad_s=previous.get('maximum_joint_velocity_error_rad_s')))
        summaries[str(case)] = dict(frames=len(data['physics_frame']),
            minimum_pair_gap_m=float(minimum), close_pair_frame_count=close_count,
            opposite_horizontal_pair_frame_count=force_count)
    for p in [args.plan, args.prediction, args.run / 'foot_impact_observation.npz', Path(__file__).resolve()]:
        pins[str(p.resolve())] = sha(p)
    for p, h in pins.items():
        if sha(p) != h:
            raise ValueError('Changed evidence during analysis ' + p)
    result = dict(scope=__doc__, summaries=summaries, records=records, source_sha256=pins,
        controller_changed=False, new_native_execution=False, valid_expert_labels=False,
        limitations=['Sensor net_forces_w contains normal forces from all contacts, not tangential force.',
            'Authored collider proximity plus normal-force alignment supports a pair hypothesis; no native pair IDs were recorded.',
            'This does not attribute all landing failures or predict a control improvement.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix('.tmp')
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    tmp.replace(args.output)
    print(json.dumps(summaries))


if __name__ == '__main__':
    main()
