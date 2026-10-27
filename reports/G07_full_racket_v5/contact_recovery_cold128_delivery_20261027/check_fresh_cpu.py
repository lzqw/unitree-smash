"""Verify the restored inputs and replay real CPU commands from the fresh tree."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--restored-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prefix = args.restored_root.resolve()

    def relocated(original):
        original = Path(original)
        assert original.is_absolute()
        return prefix.joinpath(*original.parts[1:])

    project = '/home/autodrive/LZQW/unitree_smash'
    root = relocated(project + '_contact_recovery')
    main_root = relocated(project)
    original_project_reads = []

    def audit(event, arguments):
        if event != 'open' or not isinstance(arguments[0], (str, bytes)):
            return
        name = Path(os.fsdecode(arguments[0])).absolute()
        mode = arguments[1]
        if str(name).startswith('/home/autodrive/LZQW/') and not (isinstance(mode, str) and any(c in mode for c in 'wax')):
            original_project_reads.append(str(name))
            raise RuntimeError('Fresh CPU check attempted to read the live project: ' + str(name))

    sys.addaudithook(audit)
    sys.path.insert(0, str(root))
    os.chdir(root)
    import numpy as np
    import torch
    from quadsmash.rl.contact_recovery_source_v5 import verify_recovery_source
    from quadsmash.control.full_racket_origin0_residual_v4 import make_rally_controller
    from quadsmash.tasks.stand.contact_recovery_origin0_v5 import PROTOCOL

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def read(path):
        return json.loads(path.read_text())

    queue = main_root / 'reports/G07_full_racket_v5/contact_recovery_queue02'
    expected = read(queue / 'PLAN.json')['source_sha256']
    assert len(expected) == 931
    for name, digest in expected.items():
        assert sha(relocated(name)) == digest, name
    receipt = read(main_root / 'reports/G07_full_racket_v4/origin_precision_after_aim_queue01/PRECISION_ADMISSION.json')
    for name, digest in receipt['source_sha256'].items():
        assert sha(relocated(name)) == digest
    for name in ['baseline_run', 'compact_run']:
        job = read(relocated(receipt[name]) / 'job.json')
        completion = read(relocated(receipt[name]) / 'completed.json')
        assert job['status'] == 'SUCCEEDED' and job['charged_env_steps'] == completion['env_steps'] == 6400
    for layer in read(root / 'configs/full_racket_usd_dependencies_v4.json')['layers']:
        name = Path(layer['path'])
        path = relocated(name) if name.is_absolute() else root / name
        assert path.stat().st_size == layer['bytes'] and sha(path) == layer['sha256']
    policy = main_root / 'reports/G07_full_racket_v4/gain_body8_sequential_n128_01/best_candidate.pt'
    run = main_root / 'reports/G07_full_racket_v5/teacher_contact_recovery_cold128_02'
    checkpoint = torch.load(policy, map_location='cpu', weights_only=False)
    verification = verify_recovery_source(policy, checkpoint, run, root, sha(root / PROTOCOL))
    summary = verification['source_original_TRAIN128_summary']
    assert (summary['complete'], summary['airborne_contacts'], summary['conditional_recovery_count'],
            summary['numerical_failures']) == (0, 4, 0, 0)
    controller = make_rally_controller(checkpoint, 2, 'cpu')
    metadata = read(run / 'native_command_inputs.json')
    default, limits = [torch.tensor(metadata[name], dtype=torch.float32) for name in
                       ['default_joint_position_rad', 'joint_limits_rad']]
    with np.load(run / 'public_and_actions_first2.npz') as data:
        public, actions, valid = [data[name].copy() for name in ['observations', 'actions', 'valid_first_episode']]
    maximum = 0.
    previous = None
    with torch.inference_mode():
        for values, recorded, keep in zip(public, actions, valid):
            observation = torch.from_numpy(values)
            if previous is not None:
                reset = torch.where(observation[:, 775] < previous - 1e-5)[0]
                if len(reset):
                    controller.reset(reset)
            actual = controller.action({'policy': observation}, default, limits).numpy()
            if keep.any():
                maximum = max(maximum, float(abs(actual[keep] - recorded[keep]).max()))
            previous = observation[:, 775].clone()
    assert maximum <= 5e-4 and not original_project_reads
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), fresh_tree=str(prefix),
        all931_held_inputs_verified_from_restored_tree=True,
        mapped_precision_inputs_and_probe_receipts_verified=True, all10_USD_source_layers_verified=True,
        independent_revised128_summary=summary, actual_public_rows=int(valid.sum()),
        CPU_vs_actual_GPU_max_action_error=maximum, original_action_tolerance=5e-4,
        source_policy_sha256=sha(policy), protocol_sha256=sha(root / PROTOCOL),
        imported_controller_module=str(sys.modules['quadsmash.control.full_racket_origin0_residual_v4'].__file__),
        Python_reads_from_original_live_project=original_project_reads,
        original_live_project_reads_were_rejected=True,
        source_admission_recomputed_in_fresh_tree=True,
        native_execution=False, full_original_precision_function_under_relocation_executed=False,
        external_pinned_Isaac_Python_environment_reused=True, formal_cases_read=False, formal_acceptance=False)
    assert not args.output.exists()
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(all931_held_inputs_verified=True, actual_public_rows=int(valid.sum()),
                         maximum_action_error=maximum, live_project_reads=original_project_reads,
                         native_execution=False)))


if __name__ == '__main__':
    main()
