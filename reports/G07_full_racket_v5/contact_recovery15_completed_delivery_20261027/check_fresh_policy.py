"""Replay the new frozen policy from downloaded data and restored runtime only."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, data = args.runtime_root.resolve(), args.data_root.resolve()
    assert str(root).startswith('/tmp/') and str(data).startswith('/tmp/')
    assert not args.output.exists()
    # The original walking manifest intentionally uses paths relative to its
    # runtime root; select that restored root before constructing any actor.
    os.chdir(root)
    original_reads = []

    def forbid_original_reads(event, values):
        if event != 'open' or not values or not isinstance(values[0], (str, bytes)):
            return
        name = values[0].decode() if isinstance(values[0], bytes) else values[0]
        mode = values[1] if len(values) > 1 else None
        read_mode = mode is None or (isinstance(mode, str) and ('r' in mode or '+' in mode))
        if read_mode and str(Path(name).absolute()).startswith('/home/autodrive/LZQW/'):
            original_reads.append(name)
            raise RuntimeError('Original live project read rejected during fresh CPU check')

    sys.addaudithook(forbid_original_reads)
    sys.path.insert(0, str(root))
    from quadsmash.control.full_racket_origin0_residual_v4 import make_rally_controller
    from quadsmash.control.full_racket_train_evaluation_v4 import training_cases, summarize_training
    read = lambda path: json.loads(path.read_text())
    cold, train = data / 'cold128', data / 'training15'
    for run in [cold, train]:
        for name, digest in read(run / 'execution_source_hashes.json').items():
            assert sha(root / name) == digest
    protocol_path = root / 'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json'
    protocol = read(protocol_path)
    protocol['_root'] = str(root)
    cases, indices = training_cases(root, protocol)
    summary = summarize_training(read(cold / 'episodes.json'), cases, indices, protocol)
    assert all(read(cold / 'metrics.json')[key] == value for key, value in summary.items())
    source = torch.load(train / 'policy.pt', map_location='cpu', weights_only=False)
    assert source['protocol_sha256'] == sha(protocol_path)
    assert read(cold / 'INPUTS.json')['policy_sha256'] == sha(train / 'policy.pt')
    metadata = read(cold / 'native_command_inputs.json')
    default, limits = [torch.tensor(metadata[name], dtype=torch.float32) for name in
        ['default_joint_position_rad', 'joint_limits_rad']]
    controller = make_rally_controller(source, 2, 'cpu')
    with np.load(cold / 'public_and_actions_first2.npz') as arrays:
        observations, actions, valid = [arrays[key].copy() for key in ['observations', 'actions', 'valid_first_episode']]
    maximum, previous = 0., None
    with torch.inference_mode():
        for values, native, keep in zip(observations, actions, valid):
            public = torch.from_numpy(values)
            if previous is not None:
                ids = torch.where(public[:, 775] < previous - 1e-5)[0]
                if len(ids):
                    controller.reset(ids)
            result = controller.action({'policy': public}, default, limits).numpy()
            if keep.any():
                maximum = max(maximum, float(abs(result[keep] - native[keep]).max()))
            previous = public[:, 775].clone()
    assert maximum <= 5e-4 and not original_reads
    module = sys.modules[make_rally_controller.__module__].__file__
    assert Path(module).is_relative_to(root)
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), restored_runtime=str(root),
        fresh_downloaded_data=str(data), actual_imported_module=module, actual_runtime_cwd=str(Path.cwd()),
        original128_summary=summary, actual_public_rows=int(valid.sum()),
        CPU_vs_actual_GPU_max_action_error=maximum, original_action_tolerance=5e-4,
        all394_execution_sources_match_restored_companion=True,
        original_live_project_reads_rejected=True, observed_original_live_project_reads=original_reads,
        policy_sha256=sha(train / 'policy.pt'), program_sha256=sha(Path(__file__)),
        native_reexecution=False, formal_cases_read=False, formal_acceptance=False, goal_complete=False)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(actual_public_rows=int(valid.sum()), max_action_error=maximum,
                         original_live_project_reads=len(original_reads))))


if __name__ == '__main__':
    main()
