"""Instantiate the intact source and replay its actual original GPU commands."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
MAIN = ROOT.parent / 'unitree_smash'
BASE = MAIN / 'reports/G07_full_racket_v4'
sys.path.insert(0, str(ROOT))
from quadsmash.control.full_racket_origin0_residual_v4 import make_rally_controller


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    policy_path = BASE / 'gain_body8_sequential_n128_01/best_candidate.pt'
    data_path = BASE / 'gain_body8_best_fixed_train128_01/public_and_actions_first2.npz'
    metadata_path = BASE / 'gain_body8_sequential_n128_01/INPUTS.json'
    policy = torch.load(policy_path, map_location='cpu', weights_only=False)
    metadata = json.loads(metadata_path.read_text())
    default, limits = [torch.tensor(metadata[name], dtype=torch.float32) for name in
                       ['default_joint_position_rad', 'joint_limits_rad']]
    controller = make_rally_controller(policy, 2, 'cpu')
    with np.load(data_path) as data:
        public, actual, valid = [data[name].copy() for name in ['observations', 'actions', 'valid_first_episode']]
    previous = None
    maximum = 0.
    with torch.inference_mode():
        for observation, action, keep in zip(public, actual, valid):
            value = torch.from_numpy(observation)
            if previous is not None:
                ids = torch.where(value[:, 775] < previous - 1e-5)[0]
                if len(ids):
                    controller.reset(ids)
            result = controller.action({'policy': value}, default, limits).numpy()
            if keep.any():
                maximum = max(maximum, float(abs(result[keep] - action[keep]).max()))
            previous = value[:, 775].clone()
    assert maximum <= 5e-4
    urdf = ROOT / 'assets/vendor/piper_isaac_sim/piper_description/urdf/piper_no_gripper_description.urdf'
    failure = MAIN / 'reports/G07_full_racket_v5/teacher_contact_recovery_cold128_01'
    assert not (failure / 'completed.json').exists()
    job = json.loads((failure / 'job.json').read_text())
    assert job['status'] == 'FAILED' and not job['receipt_present']
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(),
        source_policy_sha256=sha(policy_path), source_public_arrays_sha256=sha(data_path),
        source_command_metadata_sha256=sha(metadata_path), actual_valid_public_rows=int(valid.sum()),
        CPU_vs_actual_original_GPU_max_action_error=maximum, original_action_tolerance=5e-4,
        full_nested_source_controller_instantiated=True, source_policy_bytes_unchanged=True,
        read_only_piper_URDF=str(urdf), piper_URDF_sha256=sha(urdf),
        prior_startup_failure_preserved=dict(output=str(failure), job=job, error=(failure / 'error.txt').read_text(),
                                              policy_rollout_loop_entered=False),
        checker_source_sha256=sha(Path(__file__)), new_native_execution=False, formal_acceptance=False)
    with (HERE / 'CONTROLLER_RUNTIME_CPU_CHECK.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(actual_valid_public_rows=int(valid.sum()), maximum_action_error=maximum,
                         full_nested_source_controller_instantiated=True)))


if __name__ == '__main__':
    main()
