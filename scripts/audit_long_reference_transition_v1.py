"""Record the unchanged controller's reference-to-support transition.

Replays from recorded release states in the diagnostic model; never restores
a model terminal state into Isaac. Instrumentation changes no control output.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.audit_recovery_long_reference_fine_v8 as audit
from scripts.audit_joint_sequence_proposal_v1 import plain


def inspect(configuration, nominal, index, deadline):
    original = audit.PureControlLongTrajectoryRecoveryV11
    captured = []

    class RecordingController(original):
        def command(self, state, phase, time_s):
            target, diagnostic = super().command(state, phase, time_s)
            step = len(captured)
            captured.append(dict(step=step, q=state.q.copy(), v=state.v.copy(),
                previous=state.previous_command_target.copy(), target=target.copy(),
                contacts=state.contacts.copy(), phase=phase, diagnostic=diagnostic))
            return target, diagnostic

    audit.PureControlLongTrajectoryRecoveryV11 = RecordingController
    try:
        result = audit.evaluate(configuration, nominal, index, 'predictive_selection', deadline, 260)
    finally:
        audit.PureControlLongTrajectoryRecoveryV11 = original
    transitions = []
    for row in captured:
        if 145 <= row['step'] <= 180:
            delta = row['target'] - row['previous']
            transitions.append(dict(step=row['step'], source=row['diagnostic']['source'],
                solver_accepted=row['diagnostic']['solver_accepted'],
                maximum_command_change_rad=float(np.max(abs(delta))),
                maximum_leg_change_rad=float(np.max(abs(delta[:12]))),
                maximum_arm_change_rad=float(np.max(abs(delta[12:]))),
                support_feet=int(row['contacts'].sum()),
                joint_speed_rad_s=float(np.max(abs(row['v'][6:])))))
    result.update(transition_summary=transitions,
        command_trace=[r for r in captured if r['step'] >= 145],
        diagnostic_horizon_s=2.6, instrumentation_only=True,
        not_full_episode_evaluation=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['configuration', 'fixture-run', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--cases', type=int, nargs='+', required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Fresh diagnostic output required')
    read = lambda p: json.loads(Path(p).read_text())
    cfg = read(args.configuration)
    nominal = read(args.fixture_run / 'native_command_inputs.json')['default_joint_position_rad']
    library = read(cfg['model_reference_library'])
    indices = [i for i, r in enumerate(library['references']) if r['source_case'] in args.cases]
    if len(indices) != len(set(args.cases)):
        raise ValueError('Requested sources absent or duplicated')
    args.output.mkdir(parents=True)

    def write(name, value):
        path = args.output / name
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + '\n')
        temp.replace(path)

    files = [Path(__file__), args.configuration, Path(cfg['model_reference_library']),
             args.fixture_run / 'native_command_inputs.json']
    files += [Path(m.__file__) for n, m in list(sys.modules.items())
              if n.startswith('scripts.') and getattr(m, '__file__', None)]
    pins = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    write('inputs.json', dict(source_sha256=pins, cases=args.cases, max_seconds=900,
        workers=min(6, len(indices)), instrumentation_only=True, training=False))
    start = time.monotonic()
    results = []
    with ProcessPoolExecutor(max_workers=min(6, len(indices)), mp_context=mp.get_context('spawn')) as pool:
        futures = [pool.submit(inspect, cfg, nominal, i, start + 900.) for i in indices]
        for future in as_completed(futures):
            result = future.result()
            write('case_%03d.json' % result['source_case'], result)
            results.append(dict(source_case=result['source_case'], steps=result['steps'],
                failure=result['failure'], transition_summary=result['transition_summary']))
            write('status.json', dict(state='RUNNING', completed=len(results), total=len(indices)))
    for name, expected in pins.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError('Diagnostic input changed: ' + name)
    summary = dict(state='COMPLETED', wall_seconds=time.monotonic()-start,
        results=results, native_success=False, valid_expert_labels=False)
    write('result.json', summary)
    write('status.json', dict(state='COMPLETED', completed=len(results), total=len(indices)))
    print(json.dumps(plain(summary)), flush=True)


if __name__ == '__main__':
    main()
