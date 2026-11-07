"""Bounded command-space search for arm-speed margin in a coupled recovery prefix.

Original full-episode checks and actuator bounds are unchanged. This searches a
local model candidate, never a native expert label or a certified safe region.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import stress_terminal_support_commands_v1 as stress
from scripts import predict_frictional_spheres_v1 as predictor

BASE = None
PEAK = 0.0


def initialize(*paths):
    global BASE
    stress.initialize(*paths)
    BASE = stress.CONTEXT
    model = BASE[0].qp.model
    original_step = predictor.frictional_velocity_step

    def observed_step(*args, **kwargs):
        global PEAK
        result = original_step(*args, **kwargs)
        PEAK = max(PEAK, float(abs(result['velocity'][model.v_indices[13]])))
        return result

    predictor.frictional_velocity_step = observed_step


class ArmPrefixCandidate:
    def __init__(self, policy, interface, amplitude):
        self.policy = policy
        self.interface = interface
        self.amplitude = amplitude
        self.max_applied = 0.0

    def action(self, offset, state, previous):
        command, mode = self.policy.action(offset, state, previous)
        original = command[13]
        if 15 < offset < 45:
            command[13] += self.amplitude * np.sin(np.pi * (offset - 15) / 30.)
        command = np.clip(command, self.interface.limits[:, 0], self.interface.limits[:, 1])
        command[12:] = np.clip(command[12:], np.asarray(previous)[12:] - .01*self.interface.arm_rate,
                               np.asarray(previous)[12:] + .01*self.interface.arm_rate)
        self.max_applied = max(self.max_applied, float(abs(command[13] - original)))
        return command, mode


def evaluate(task):
    global PEAK
    PEAK = 0.0
    controller, policy, *rest = BASE
    candidate = ArmPrefixCandidate(policy, controller.qp.interface, task['arm_correction_rad'])
    stress.CONTEXT = (controller, candidate, *rest)
    row = stress.evaluate(task)
    row['joint2_peak_rad_s'] = PEAK
    row['joint2_limit_rad_s'] = float(controller.qp.interface.velocity[13])
    row['joint2_command_correction_max_rad'] = candidate.max_applied
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['bundle', 'warm-plan', 'validation', 'sphere-audit', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--amplitudes', type=float, nargs='+', default=[0., -.001, -.003, -.006, -.01])
    parser.add_argument('--all-pulses', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve previous experiment')
    if any(not np.isfinite(value) or abs(value) > .02 for value in args.amplitudes):
        raise ValueError('Local command search limited to 0.02 rad')
    paths = tuple(str(path) for path in [args.bundle, args.warm_plan, args.validation, args.sphere_audit])
    initialize(*paths)
    files = [Path(__file__), args.bundle/'manifest.json', args.warm_plan, args.validation, args.sphere_audit]
    files += [Path(value.__file__) for key,value in list(sys.modules.items())
              if key.startswith('scripts.') and getattr(value, '__file__', None)]
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    hashes = {str(path.resolve()): digest(path) for path in files}
    tasks = []
    for correction in args.amplitudes:
        tasks.append(dict(name='nominal', feedback=True, start=0, end=0, joint_start=0,
                          joint_end=4, amplitude_rad=0., arm_correction_rad=correction))
        groups = [('prefix_hips', 0, 10, 0, 4)]
        if args.all_pulses:
            groups.append(('terminal_calves', 80, 100, 8, 12))
        for name,start,end,j0,j1 in groups:
            for pulse in ([-.02,-.005,.005,.02] if args.all_pulses else [-.02,-.005]):
                tasks.append(dict(name=name, feedback=True, start=start, end=end,
                                  joint_start=j0,joint_end=j1,amplitude_rad=pulse,arm_correction_rad=correction))
    rows = []
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize, initargs=paths) as pool:
        for row in pool.map(evaluate, tasks):
            rows.append(row)
            print(json.dumps(row), flush=True)
            temporary = args.output.with_suffix('.partial.tmp')
            temporary.write_text(json.dumps(dict(rows=rows, complete=False), indent=2)+'\n')
            temporary.replace(args.output.with_suffix('.partial.json'))
    for path, expected in hashes.items():
        if digest(path) != expected:
            raise ValueError('Input changed: '+path)
    output = dict(scope=__doc__, rows=rows, complete=True, source_sha256=hashes,
                  correction_shape=dict(joint_index=13, start=15, end=45, profile='half_sine'),
                  wall_seconds=time.monotonic()-started, native_execution=False,
                  valid_expert_labels=False, generalization_proven=False)
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    temporary.replace(args.output)


if __name__ == '__main__':
    main()
