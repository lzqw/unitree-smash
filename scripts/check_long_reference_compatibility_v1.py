"""Compare unchanged short-reference command mappings before native rollout."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.pure_control_trajectory_recovery_v9 import PureControlTrajectoryRecoveryV9
from scripts.pure_control_long_trajectory_recovery_v11 import PureControlLongTrajectoryRecoveryV11
from scripts.whole_body_support_qp_v1 import PositionPDInterface


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['original-configuration', 'new-configuration', 'fixture-run', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Fresh compatibility evidence required')
    read = lambda p: json.loads(Path(p).read_text())
    old_cfg, new_cfg = read(args.original_configuration), read(args.new_configuration)
    nominal = read(args.fixture_run / 'native_command_inputs.json')['default_joint_position_rad']
    original = lambda: PureControlTrajectoryRecoveryV9(PositionPDInterface(**old_cfg['interface']), nominal, old_cfg['model_reference_library'])
    modified = lambda: PureControlLongTrajectoryRecoveryV11(PositionPDInterface(**new_cfg['interface']), nominal, new_cfg['model_reference_library'])
    library = modified().library
    rows = []
    for source in library.records:
        if len(source['trajectory']) != 60:
            continue
        old, new = original(), modified()
        errors = []
        for k in [-1, 29, 59]:
            before = library.state(source, k)
            after = library.state(source, k)
            phase = 'BUFFER_LOAD' if before.contacts.any() else 'AIR_PREPARE'
            u, first = old.command(before, phase, 100. + (k + 1) * .01)
            v, second = new.command(after, phase, 100. + (k + 1) * .01)
            error = float(np.max(abs(u - v)))
            if error > 1e-12 or first['source'] != second['source']:
                raise ValueError('Unchanged reference behavior drifted: ' + str((source['source_case'], k, error)))
            errors.append(error)
        rows.append(dict(source_key=source['source_key'], maximum_command_error_rad=max(errors)))
    pins = [Path(__file__), args.original_configuration, args.new_configuration,
            Path(old_cfg['model_reference_library']), Path(new_cfg['model_reference_library'])]
    result = dict(passed=True, unchanged_references=len(rows), command_comparisons=3 * len(rows),
        records=rows, native_success=False,
        scope='Same recorded inputs and command times; does not prove identical closed-loop physical trajectories.',
        source_sha256={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in pins})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['records', 'source_sha256']}))


if __name__ == '__main__':
    main()
