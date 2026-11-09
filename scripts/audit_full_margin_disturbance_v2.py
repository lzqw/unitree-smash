"""Paired model-only disturbance tests of full nominal landing sequences.

Both plans start at the same recorded release. A declared artificial angular
velocity change at tick 300 probes sensitivity; it is not a native reset,
physical task episode, or evidence of an external-force recovery method.
Compare fixed commands with the existing V15 observed-state candidate selector.
The predictor and plant share a model, so native robustness remains unproved.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.pure_control_stance_foot_reference_v15 import PureControlStanceFootReferenceV15
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.audit_actual_handoff_model_v1 import state_from_record
from scripts.audit_settled_stance_model_v2 import ForceRecorder
from scripts.event_recovery_manager_v1 import EventRecoveryManagerV1, RecoveryPhaseMemory
from scripts.predict_self_contact_spheres_v2 import predict_self_contact_spheres
from scripts.retry_budgeted_settling_feedback_v1 import sha, write


def evaluate(cfg, nominal, path, label, mode, pitch_delta, deadline):
    data = json.loads(Path(path).read_text())
    record, reference = data['initial_actual_state'], data['trajectory']
    c = PureControlStanceFootReferenceV15(PositionPDInterface(**cfg['interface']),
        nominal, cfg['model_reference_library'])
    state = state_from_record(c.model, record)
    phase = record['phase']
    c.chosen = dict(query=dict(q=record['q'], v=record['v'],
        contacts=phase['ground_contacts'], target=record['previous']),
        trajectory=reference, source_key=sha(path), native_success=False)
    c.selection_done = True
    c.selection_score = None
    manager = EventRecoveryManagerV1()
    manager.memory = RecoveryPhaseMemory(episode=0, last_time_s=record['time_s'],
        phase=phase['phase'], release_time_s=phase['release_time_s'],
        first_ground_time_s=phase['first_ground_time_s'],
        settled_duration_s=phase['phase_settled_duration_s'],
        final_stable_duration_s=0., ground_contacts=state.contacts.copy())
    rows, failure, disturbance = [], None, None
    maximum_command_difference = 0.
    first_command_difference = None
    for tick in range(record['tick'], 800):
        if time.monotonic() > deadline:
            failure = dict(tick=tick, reason='compute_budget'); break
        if tick == 300:
            before = state.v.copy()
            world_delta = np.array([0., pitch_delta, 0.])
            state.v[3:6] += state.root_rotation_world.T @ world_delta
            achieved = state.root_rotation_world @ (state.v[3:6]-before[3:6])
            if not np.allclose(achieved, world_delta, atol=1e-12, rtol=0.):
                raise ValueError('Disturbance coordinate conversion mismatch')
            disturbance = dict(tick=tick, world_angular_velocity_delta_rad_s=achieved,
                before_v=before, after_v=state.v.copy(), contacts=state.contacts.copy(),
                native_execution=False)
        planned = np.asarray(reference[tick-record['tick']]['target'], float)
        q_before, v_before = state.q.copy(), state.v.copy()
        if mode == 'feedback':
            target, diag = c.command(state, phase['phase'], float(np.float32(tick)*np.float32(.01)))
            source = diag['source']
        else:
            target, source = planned.copy(), 'fixed_full_sequence'
        if not np.array_equal(q_before, state.q) or not np.array_equal(v_before, state.v):
            raise ValueError('Command selection changed observed state')
        delta = float(np.max(abs(target-planned)))
        maximum_command_difference = max(maximum_command_difference, delta)
        if delta > 1e-8 and first_command_difference is None:
            first_command_difference = tick
        recorder = ForceRecorder(c.model)
        try:
            prediction = predict_self_contact_spheres(c.model, c.interface, state, target,
                c.geometry.spheres, substeps=160, geometry=c.path_geometry,
                collision_response=recorder)
        except (ValueError, np.linalg.LinAlgError) as error:
            failure = dict(tick=tick, reason='model_rejected', error=repr(error)); break
        bad = {n: v for n, v in prediction['diagnostics'].items() if
            (n == 'nonfoot_ground_violation_m' and v > 1e-6) or
            (n in ['joint_position_violation_rad', 'joint_velocity_violation_rad_s',
                   'motor_torque_violation_nm'] and v > 3e-4)}
        if bad:
            failure = dict(tick=tick, reason='physical_path_violation', violations=bad,
                collision=prediction['first_ground_violation'], source=source); break
        state = prediction['state']; state.previous_command_target = target.copy()
        force = recorder.last_force; rotation = state.root_rotation_world
        phase = manager.update(episode=0, time_s=float(np.float32(tick+1)*np.float32(.01)),
            had_racket_contact=True, released=True, racket_touching=False, separated_now=True,
            ground_normal_force_n=force[:, 2],
            roll_pitch_rad=[np.arctan2(rotation[2, 1], rotation[2, 2]),
                np.arcsin(np.clip(-rotation[2, 0], -1, 1))],
            root_velocity_world=rotation@state.v[:3], root_angular_velocity_world=rotation@state.v[3:6],
            joint_velocity=state.joint_velocity, nonfoot_contact_now=False)
        state.contacts = phase['ground_contacts'].copy(); state.measured_force_world = force.copy()
        rows.append(dict(tick=tick, q=state.q.copy(), v=state.v.copy(), target=target.copy(),
            ground_force=force.copy(), phase=phase, source=source,
            difference_from_nominal_rad=delta))
    full = failure is None and len(rows) == 800-record['tick']
    return dict(plan=label, mode=mode, pitch_delta_rad_s=pitch_delta, disturbance=disturbance,
        initial_actual_state=record, trajectory=rows, steps=len(rows),
        full_model_path=full, failure=failure, end_time_s=8. if full else None,
        final_model_stable_s=phase['control_sample_stable_duration_s'],
        model_terminal_window_passed=full and phase['control_sample_stable_duration_s'] >= .8,
        maximum_command_difference_rad=maximum_command_difference,
        first_command_difference_tick=first_command_difference,
        native_execution=False, native_success=False, valid_expert_labels=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['baseline-feedback', 'optimized-feedback', 'configuration', 'run', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise ValueError('Fresh output required')
    read = lambda path: json.loads(Path(path).read_text())
    cfg = read(args.configuration)
    if read(args.run/'job.json')['status'] != 'SUCCEEDED':
        raise ValueError('Complete recorded native release required')
    plans = {name: read(path) for name, path in
        [('baseline', args.baseline_feedback), ('optimized', args.optimized_feedback)]}
    pins = {}
    for path in [args.baseline_feedback, args.optimized_feedback]:
        inputs = read(path.parent/'inputs.json')
        for filename, digest in inputs['source_sha256'].items():
            if sha(filename) != digest or (filename in pins and pins[filename] != digest):
                raise ValueError('Changed or conflicting source '+filename)
            pins[filename] = digest
    original = plans['baseline']['initial_actual_state']
    with np.load(args.run/'pure_control_commands.npz') as z:
        ids = np.flatnonzero((z['lane']==original['case']) & (z['tick']==original['tick']))
        if len(ids) != 1 or any(not np.array_equal(z[k][ids[0]], original[k])
                for k in ['q', 'v', 'previous', 'ground_force']):
            raise ValueError('Wrong actual release')
    if not original['tick'] < 300 < 800: raise ValueError('Perturbation outside episode')
    for data in plans.values():
        if (data['initial_actual_state'] != original or data['failure'] is not None or
                not data['full_model_path'] or data['final_model_stable_s'] < .8 or
                len(data['trajectory']) != 800-original['tick']):
            raise ValueError('Require two full stable same-release nominal sequences')
        if any(row['tick'] != original['tick']+i for i, row in enumerate(data['trajectory'])):
            raise ValueError('Missing control interval')
    if pins.get(str(args.configuration.resolve())) != sha(args.configuration):
        raise ValueError('Changed actual interface')
    nominal = read(args.run/'native_command_inputs.json')['default_joint_position_rad']
    # Load optional runtime dependencies before freezing the source manifest.
    controller = PureControlStanceFootReferenceV15(PositionPDInterface(**cfg['interface']), nominal,
        cfg['model_reference_library'])
    schema_checks = []
    for label, data in plans.items():
        adapted = []
        for i, row in enumerate(data['trajectory']):
            contacts = row['phase']['ground_contacts']
            if len(contacts) != 4 or any(type(x) is not bool for x in contacts):
                raise ValueError('Invalid post-interval contact observation')
            if 'contacts' in row and row['contacts'] != contacts:
                raise ValueError('Conflicting contact observations')
            adapted.append(dict(row, contacts=contacts, step=i))
        # Check the actual reference reader before launching any pool work.
        reference = dict(trajectory=adapted)
        for i in [0, len(adapted)-1]:
            rebuilt = controller.library.state(reference, i)
            if (not np.array_equal(rebuilt.q, adapted[i]['q']) or
                    not np.array_equal(rebuilt.v, adapted[i]['v']) or
                    not np.array_equal(rebuilt.contacts, adapted[i]['contacts'])):
                raise ValueError('Reference schema changed physical observation')
        data['trajectory'] = adapted
        schema_checks.append(dict(plan=label, steps=len(adapted),
            only_added_contacts_and_step=True, actual_reference_reader_passed=True))
    files = [Path(__file__), args.baseline_feedback, args.optimized_feedback, args.configuration,
        args.run/'job.json', args.run/'native_command_inputs.json', args.run/'pure_control_commands.npz']
    files += [Path(m.__file__) for n, m in list(sys.modules.items())
        if n.startswith('scripts.') and getattr(m, '__file__', None)]
    pins.update({str(p.resolve()): sha(p) for p in files})
    args.output.mkdir(parents=True)
    snapshots = {}
    for name, data in plans.items():
        path = args.output/(name+'_sequence.json')
        write(path, dict(initial_actual_state=original, trajectory=data['trajectory']))
        pins[str(path.resolve())] = sha(path); snapshots[name] = path
    write(args.output/'inputs.json', dict(source_sha256=pins, scope=__doc__, workers=4,
        max_seconds=2000, disturbance_tick=300, world_pitch_deltas_rad_s=[0., -.05, .05],
        physical_substeps=160, scoring_substeps=16, reference_selection_forced=True,
        forced_selection_scope='Same nominal reference isolated in paired model diagnostic',
        reference_schema_checks=schema_checks,
        native_execution=False, training=False, hardware_changed=False, acceptance_changed=False))
    start = time.monotonic(); results = []
    def progress(state):
        write(args.output/'status.json', dict(state=state, pid=os.getpid(), completed=len(results),
            total=12, results=results))
    progress('RUNNING_PAIRED_MARGIN_DISTURBANCE')
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context('spawn')) as pool:
        jobs = {pool.submit(evaluate, cfg, nominal, str(snapshots[name]), name, mode, delta, start+2000):
                (name, mode, delta)
            for delta in [0., -.05, .05] for name in plans for mode in ['fixed', 'feedback']}
        for future in as_completed(jobs):
            try:
                row = future.result()
            except Exception as error:
                label, mode, delta = jobs[future]
                row = dict(plan=label, mode=mode, pitch_delta_rad_s=delta,
                    failure=dict(reason='worker_exception', error=repr(error)),
                    full_model_path=False, model_terminal_window_passed=False,
                    native_execution=False, native_success=False, valid_expert_labels=False)
            name = row['plan']+'_'+row['mode']+'_'+str(row['pitch_delta_rad_s'])+'.json'
            write(args.output/name, row)
            results.append({k: v for k, v in row.items() if k not in ['trajectory', 'initial_actual_state']})
            progress('RUNNING_PAIRED_MARGIN_DISTURBANCE')
    for filename, digest in pins.items():
        if sha(filename) != digest: raise ValueError('Changed diagnostic input '+filename)
    errors = sum((r.get('failure') or {}).get('reason')=='worker_exception' for r in results)
    final = dict(state=('FAILED_PAIRED_MARGIN_WORKER' if errors else 'COMPLETED_PAIRED_MARGIN_DISTURBANCE'),
        worker_exceptions=errors, results=results,
        elapsed_s=time.monotonic()-start, same_predictor_and_plant=True,
        native_execution=False, native_success=False, valid_expert_labels=False)
    write(args.output/'result.json', final); progress(final['state'])
    print(json.dumps(final))


if __name__ == '__main__':
    main()
