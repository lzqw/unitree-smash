"""CPU replay a downloaded proposal using only its packet and restored runtime."""
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
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    runtime, packet = args.runtime_root.resolve(), args.package_root.resolve()
    assert str(runtime).startswith('/tmp/') and str(packet).startswith('/tmp/')
    assert not args.output.exists()
    os.chdir(runtime)
    reads = []

    def reject_original(event, values):
        if event != 'open' or not values or not isinstance(values[0], (str, bytes)):
            return
        name = values[0].decode() if isinstance(values[0], bytes) else values[0]
        mode = values[1] if len(values) > 1 else None
        if (mode is None or isinstance(mode, str) and ('r' in mode or '+' in mode)) and str(Path(name).absolute()).startswith('/home/autodrive/LZQW/'):
            reads.append(name)
            raise RuntimeError('Original project read rejected during portable replay')

    sys.addaudithook(reject_original)
    sys.path.insert(0, str(runtime))
    from quadsmash.rl.episode_trace_v4 import unpack_public
    family_path = packet / 'policy/recovery_prior_proposals.pt'
    family = torch.load(family_path, map_location='cpu', weights_only=False)
    assert family['native_execution'] is False and family['formal_acceptance'] is False
    controller_dir = packet / 'controllers/G07_full_racket_v5/recovery_motion_prior_20261027'
    spec = importlib.util.spec_from_file_location('downloaded_recovery_proposal_io', controller_dir / 'proposal_io.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    controller = module.make_controller(family, 1, 1, 'cpu')
    with np.load(packet / 'reference/actual_recovered_trace.npz') as data:
        public = unpack_public(data['packed_public'])
        requests, actions = data['residual_actions'].copy(), data['native_actions'].copy()
    commands = json.loads((packet / 'reference/native_command_inputs.json').read_text())
    default, limits = [torch.tensor(commands[key], dtype=torch.float32) for key in
        ['default_joint_position_rad', 'joint_limits_rad']]
    maximum = 0.
    with torch.inference_mode():
        for values, request, actual in zip(public, requests, actions):
            result = controller.action({'policy': torch.from_numpy(values[None])}, default, limits)
            assert np.array_equal(controller.last_request.numpy()[0], request)
            maximum = max(maximum, float(abs(result.numpy()[0] - actual).max()))
    assert len(public) == 800 and maximum <= 5e-4 and not reads
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), restored_runtime=str(runtime),
        downloaded_packet=str(packet), source_policy_native_execution_preserved=True,
        proposed_family_native_execution=False, actual_public_rows=800,
        all800_lookup_requests_match_real_recorded_requests_bitwise=True,
        CPU_vs_actual_recovered_GPU_command_error=maximum, original_action_tolerance=5e-4,
        original_project_Python_reads_rejected=True, observed_original_project_Python_reads=reads,
        family_sha256=hashlib.sha256(family_path.read_bytes()).hexdigest(),
        program_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        imported_core=str(Path(sys.modules[unpack_public.__module__].__file__).resolve()),
        no_new_physical_recovery_or_smash_claimed=True, native_reexecution=False,
        formal_cases_read=False, formal_acceptance=False, goal_complete=False)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ['actual_public_rows', 'CPU_vs_actual_recovered_GPU_command_error',
        'proposed_family_native_execution', 'observed_original_project_Python_reads']}))


if __name__ == '__main__':
    main()
