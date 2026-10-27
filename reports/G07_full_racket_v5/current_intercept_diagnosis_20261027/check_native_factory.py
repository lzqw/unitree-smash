"""Bind the unchanged target core to two real sources without inventing future policies."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2].parent/'unitree_smash_contact_recovery'
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2)
    path = HERE/'native_factory.py'
    spec = importlib.util.spec_from_file_location('checked_shared_reference_native_factory', path)
    factory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(factory)
    cpu = read(HERE/'CPU_SHARED_TARGET_CHECK.json')
    assert all(sha(Path(p)) == h for p, h in cpu['source_sha256'].items())
    files = [Path(__file__), path, HERE/'shared_target_controller.py', HERE/'CPU_SHARED_TARGET_CHECK.json']
    results = []
    for name, suffix, policy_path in [
        ('early_lateral7_best_cold128_01', '001', HERE.parent/'early_lateral7_n128_01/best_candidate.pt'),
        ('windowed_swing7_n128_01', '003', HERE.parent/'windowed_swing7_n128_01/candidate_003.pt')]:
        run = HERE.parent/name
        inputs_path, data_path = run/'INPUTS.json', run/f'public_and_actions_{suffix}.npz'
        inputs = read(inputs_path)
        defaults, limits = [torch.tensor(inputs[k], dtype=torch.float32) for k in
                            ['default_joint_position_rad', 'joint_limits_rad']]
        source = torch.load(policy_path, map_location='cpu', weights_only=False)
        with np.load(data_path) as data:
            public, native, valid = [data[k].copy() for k in ['observations', 'actions', 'valid_first_episode']]
        files.extend([policy_path, inputs_path, data_path])
        for candidate in [factory.candidates()[0], factory.candidates()[3]]:
            controller = factory.make_controller(source, 6, 'cpu', bias_world_m=candidate['bias_world_m'])
            digest = hashlib.sha256()
            previous = None
            maximum = 0.
            for value, actual, keep in zip(public, native, valid):
                if previous is not None:
                    ids = torch.tensor(np.flatnonzero(value[:, 775] < previous-1e-6))
                    if len(ids):
                        controller.reset(ids)
                action = controller.action({'policy': torch.from_numpy(value)}, defaults, limits)
                digest.update(action.numpy().tobytes())
                if keep.any():
                    maximum = max(maximum, float(abs(action.numpy()[keep]-actual[keep]).max()))
                previous = value[:, 775].copy()
            original = next(r for r in cpu['results'] if r['run'] == name
                            and r['candidate']['candidate_id'] == candidate['candidate_id'])
            assert digest.hexdigest() == original['command_stream_sha256']
            if candidate['candidate_id'] == 0:
                assert maximum <= .0005
            result = dict(run=name, candidate_id=candidate['candidate_id'], actual_public_rows=4800,
                valid_first_episode_rows=int(valid.sum()), checked_core_command_stream_bitwise_identical=True,
                actual_baseline_GPU_max_error=maximum if candidate['candidate_id'] == 0 else None)
            results.append(result)
            print(json.dumps(result), flush=True)
    record = dict(checked_at=datetime.now(timezone.utc).isoformat(), results=results,
        factory_sha256=sha(path), future_actual_source_branches_pending_real_source_admission_and_replay=True,
        source_sha256={str(p): sha(p) for p in sorted(set(files))},
        no_future_checkpoint_fabricated=True, new_native_execution=False, formal_acceptance=False)
    with (HERE/'NATIVE_FACTORY_CPU_CHECK.json').open('x') as f:
        f.write(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__':
    main()
