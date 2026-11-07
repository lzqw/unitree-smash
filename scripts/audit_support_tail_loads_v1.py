"""Audit actual computed model loads, rather than treating four flags as load.

Replays the last ten saved fine-model intervals with an observational hook.
These are predicted impulses; no native contact-force measurement is implied.
"""
import argparse
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pinocchio as pin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.search_landing_support_tail_v3 import initialize_worker, _WORKER


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve prior evidence')
    read = lambda p: json.loads(Path(p).read_text())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    source = read(args.source)
    hashes = dict(source['source_sha256'])
    hashes[str(args.source.resolve())] = sha(args.source)
    hashes[str(Path(__file__).resolve())] = sha(__file__)
    for filename, expected in hashes.items():
        if sha(filename) != expected:
            raise ValueError('Changed source ' + filename)
    if not source['any_model_feasible'] or not source['replay_reproduced']:
        raise ValueError('Reproduced fine candidate required')
    tail = read(source['configuration']['source'])
    parent = read(tail['configuration']['source_search'])
    cfg = parent['configuration']
    query = read(cfg['query'])
    manifest = read(Path(cfg['bundle'])/'manifest.json')
    initialize_worker(manifest['configuration'], query, source['fixed_early_knots_rad'], cfg['sphere_audit'])
    w = _WORKER
    model = w['controller'].qp.model
    interface = w['controller'].qp.interface
    replay = source['independent_replay']
    if replay['substeps'] != 160 or len(replay['trajectory']) != 520:
        raise ValueError('Original complete fine trajectory required')
    predictor = importlib.import_module('scripts.predict_frictional_spheres_v1')
    original = predictor.frictional_velocity_step
    impulses = []
    def observe(*a, **kw):
        result = original(*a, **kw)
        impulses.append(result['impulse'].reshape(4, 3).copy())
        return result
    maximum_error = 0.
    predictor.frictional_velocity_step = observe
    try:
        for index in range(510, 520):
            previous, expected = replay['trajectory'][index-1:index+1]
            state = copy.deepcopy(w['initial'])
            state.q = np.asarray(previous['q']); state.v = np.asarray(previous['v'])
            state.joint_position = state.q[model.q_indices].copy()
            state.joint_velocity = state.v[model.v_indices].copy()
            state.root_position_world = state.q[:3].copy()
            state.root_rotation_world = pin.Quaternion(state.q[3:7]).matrix()
            state.contacts = np.asarray(previous['contacts'], bool)
            state.previous_command_target = np.asarray(previous['target'])
            predicted = predictor.predict_frictional_spheres(model, interface, state,
                np.asarray(expected['target']), w['spheres'], geometry=w['geometry'], substeps=160)
            for key in ['q', 'v']:
                maximum_error = max(maximum_error, float(np.max(abs(getattr(predicted['state'], key)-expected[key]))))
    finally:
        predictor.frictional_velocity_step = original
    if maximum_error > 1e-12 or len(impulses) != 1600:
        raise ValueError('Load observation changed the saved fine trajectory')
    forces = np.asarray(impulses)[:, :, 2]/(.01/160)
    for filename, expected in hashes.items():
        if sha(filename) != expected:
            raise ValueError('Changed source during load audit ' + filename)
    output = dict(scope=__doc__, case=3, control_ticks=[790, 799], micro_samples=len(forces),
        foot_order=model.feet, mean_normal_force_n=forces.mean(0).tolist(),
        minimum_normal_force_n=forces.min(0).tolist(), maximum_normal_force_n=forces.max(0).tolist(),
        fraction_at_least_5n=(forces>=5.).mean(0).tolist(), maximum_replay_state_difference=maximum_error,
        source_sha256=hashes, native_execution=False, valid_expert_labels=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in output.items() if k != 'source_sha256'}))


if __name__ == '__main__':
    main()
