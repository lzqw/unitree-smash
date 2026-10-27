"""Repeat every portable proposal on the separately downloaded real public data."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root', type=Path, required=True)
    parser.add_argument('--package-root', type=Path, required=True)
    parser.add_argument('--public-data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, packet, public_path = args.runtime_root.resolve(), args.package_root.resolve(), args.public_data.resolve()
    assert all(str(path).startswith('/tmp/') for path in [root, packet, public_path])
    os.chdir(root)
    original_reads = []

    def forbid_original(event, values):
        if event != 'open' or not values or not isinstance(values[0], (str, bytes)):
            return
        name = values[0].decode() if isinstance(values[0], bytes) else values[0]
        mode = values[1] if len(values) > 1 else None
        if (mode is None or isinstance(mode, str) and ('r' in mode or '+' in mode)) and str(Path(name).absolute()).startswith('/home/autodrive/LZQW/'):
            original_reads.append(name)
            raise RuntimeError('Original live project read rejected')

    sys.addaudithook(forbid_original)
    sys.path.insert(0, str(root))
    controller_dir = packet / 'controllers/G07_full_racket_v5/recovery_motion_prior_20261027'
    spec = importlib.util.spec_from_file_location('downloaded_prior_io', controller_dir / 'proposal_io.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    family_path = packet / 'policy/recovery_prior_proposals.pt'
    family = torch.load(family_path, map_location='cpu', weights_only=False)
    assert family['native_execution'] is False
    commands = json.loads((packet / 'reference/native_command_inputs.json').read_text())
    default, limits = [torch.tensor(commands[key], dtype=torch.float32) for key in
                       ['default_joint_position_rad', 'joint_limits_rad']]
    with np.load(public_path) as data:
        public, actions, valid = [data[key].copy() for key in ['observations', 'actions', 'valid_first_episode']]
        indices = data['training_indices'].tolist()
    assert public.shape == (800, 6, 794) and indices == [0, 29, 69, 100, 112, 123]
    expected, results = [], []
    with torch.inference_mode():
        for candidate in family['candidates']:
            controller = module.make_controller(family, candidate['candidate_id'], 6, 'cpu')
            native_error = source_difference = slew = 0.
            previous = None
            digest = hashlib.sha256()
            for tick, (values, native, active) in enumerate(zip(public, actions, valid)):
                result = controller.action({'policy': torch.from_numpy(values)}, default, limits)
                assert torch.isfinite(result).all() and result.shape == (6, 18) and (abs(result) <= 1).all()
                digest.update(result.numpy().tobytes())
                native_error = max(native_error, float(abs(result.numpy()[active] - native[active]).max()))
                if candidate['candidate_id'] == 0:
                    expected.append(result.clone())
                else:
                    difference = abs(result - expected[tick])
                    source_difference = max(source_difference, float(difference.max()))
                    if candidate['group'] == 'legs':
                        assert not difference[:, 12:].any()
                    elif candidate['group'] == 'arms':
                        assert not difference[:, :12].any()
                target = limits[:, 0] + .5 * (result + 1) * (limits[:, 1] - limits[:, 0])
                if previous is not None:
                    slew = max(slew, float(abs(target[:, 12:] - previous[:, 12:]).max()) / .01)
                previous = target.clone()
            assert slew <= 5.0005
            if candidate['candidate_id'] == 0:
                assert native_error <= 5e-4
            else:
                assert source_difference > 0
            row = dict(candidate_id=candidate['candidate_id'], real_public_rows=int(valid.sum()),
                maximum_original_GPU_action_difference=native_error, maximum_source_action_difference=source_difference,
                maximum_arm_target_rate_rad_s=slew, replayed_action_stream_sha256=digest.hexdigest())
            results.append(row)
            print(json.dumps(row), flush=True)
    assert len(results) == 9 and not original_reads
    record = dict(checked_at=datetime.now(timezone.utc).isoformat(), downloaded_packet=str(packet),
        restored_runtime=str(root), downloaded_public_dataset=str(public_path),
        all9_portable_candidates_replayed=True, actual_public_rows_per_candidate=4800, results=results,
        public_dataset_sha256=hashlib.sha256(public_path.read_bytes()).hexdigest(),
        family_sha256=hashlib.sha256(family_path.read_bytes()).hexdigest(),
        original_live_project_Python_reads_rejected=True, observed_original_project_Python_reads=original_reads,
        program_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        proposed_family_native_execution=False, no_new_closed_loop_outcome_claimed=True,
        formal_cases_read=False, formal_acceptance=False, goal_complete=False)
    with args.output.open('x') as stream:
        stream.write(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
