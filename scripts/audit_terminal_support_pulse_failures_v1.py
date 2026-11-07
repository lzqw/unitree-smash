"""Reproduce failed feedback pulses without altering the declared model or limits."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import stress_terminal_support_commands_v1 as stress
from scripts import predict_frictional_spheres_v1 as predictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['bundle', 'warm-plan', 'validation', 'sphere-audit', 'pulses', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve prior audit evidence')
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    evidence = json.loads(args.pulses.read_text())
    hashes = dict(evidence['source_sha256'])
    hashes[str(args.pulses.resolve())] = sha(args.pulses)
    hashes[str(Path(__file__).resolve())] = sha(__file__)
    for path, digest in hashes.items():
        if sha(path) != digest:
            raise ValueError('Changed input: ' + path)
    stress.initialize(args.bundle, args.warm_plan, args.validation, args.sphere_audit)
    controller, policy, initial, _, spheres, geometry, _ = stress.CONTEXT
    model = controller.qp.model
    interface = controller.qp.interface
    microsteps = []
    original_step = predictor.frictional_velocity_step

    def observed_step(*arguments, **keywords):
        result = original_step(*arguments, **keywords)
        speed = np.abs(result['velocity'][model.v_indices])
        worst = int(np.argmax(speed - interface.velocity))
        microsteps.append(dict(micro=len(microsteps), joint_index=worst,
                              speed_rad_s=float(speed[worst]),
                              limit_rad_s=float(interface.velocity[worst]),
                              violation_rad_s=float(speed[worst] - interface.velocity[worst])))
        return result

    # Observational wrapper only: the original solver result is returned unchanged.
    predictor.frictional_velocity_step = observed_step
    rows = []
    for source in evidence['rows']:
        task = source['task']
        if not task['feedback'] or source['failure'] is None:
            continue
        state = copy.deepcopy(initial)
        for offset in range(source['failure']['offset'] + 1):
            target, _ = policy.action(offset, state, state.previous_command_target)
            if task['start'] <= offset < task['end']:
                target[task['joint_start']:task['joint_end']] += task['amplitude_rad']
            target = np.clip(target, interface.limits[:, 0], interface.limits[:, 1])
            target[12:] = np.clip(target[12:], state.previous_command_target[12:] - .1,
                                 state.previous_command_target[12:] + .1)
            microsteps.clear()
            item = stress.predict_frictional_spheres(model, interface, state, target,
                                                     spheres, geometry=geometry)
            if offset == source['failure']['offset']:
                for key, value in source['failure']['violations'].items():
                    if abs(item['diagnostics'][key] - value) > 1e-10:
                        raise ValueError('Failed pulse did not reproduce')
                following = item['state']
                speed = np.abs(following.joint_velocity)
                ratio = speed / interface.velocity
                worst = int(np.argmax(ratio))
                rows.append(dict(task=task, failure=source['failure'],
                    maximum_microstep_velocity_excess=max(microsteps, key=lambda row: row['violation_rad_s']),
                    first_ground_violation=item['first_ground_violation'],
                    control_boundary_max_speed_ratio=float(ratio[worst]),
                    control_boundary_worst_joint_index=worst,
                    control_boundary_worst_joint_speed=float(speed[worst]),
                    control_boundary_worst_joint_limit=float(interface.velocity[worst]),
                    joint_velocities=following.joint_velocity.tolist(),
                    interpretation='Worst boundary joint need not be worst at internal microstep; path diagnostics retain microstep maxima.'))
            state = item['state']
            state.previous_command_target = target.copy()
    for path, digest in hashes.items():
        if sha(path) != digest:
            raise ValueError('Input changed during audit: ' + path)
    result = dict(rows=rows, failures_reproduced=len(rows), source_sha256=hashes,
                  native_execution=False, valid_expert_labels=False)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'source_sha256'}))


if __name__ == '__main__':
    main()
