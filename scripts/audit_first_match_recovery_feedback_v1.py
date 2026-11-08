"""Check the full controller at its first causal match, not its plan query.

Reconstruct the controller memory using the recorded native prefix, checking
every issued command against the next actual readback. Then continue the CPU
model from the first matching native state through the original 8 s endpoint.
This is model evidence, not native execution, teleportation or expert data.
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
from scripts.pure_control_preserved_stance_replanning_v17 import PureControlPreservedStanceReplanningV17
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.audit_actual_handoff_model_v1 import state_from_record
from scripts.audit_settled_stance_model_v2 import ForceRecorder
from scripts.event_recovery_manager_v1 import EventRecoveryManagerV1, RecoveryPhaseMemory
from scripts.predict_self_contact_spheres_v2 import predict_self_contact_spheres
from scripts.audit_joint_sequence_proposal_v1 import plain


def evaluate(cfg, nominal, library, match, records, deadline):
    controller = PureControlPreservedStanceReplanningV17(
        PositionPDInterface(**cfg['interface']), nominal,
        cfg['model_reference_library'], library)
    prefix_errors = []
    for row in records[:-1]:
        if time.monotonic() > deadline:
            raise TimeoutError('Prefix reconstruction exceeded its compute bound')
        state = state_from_record(controller.model, row)
        q, v = state.q.copy(), state.v.copy()
        target, diagnostic = controller.command(state, row['phase']['phase'], row['time_s'])
        if not np.array_equal(q, state.q) or not np.array_equal(v, state.v):
            raise ValueError('Controller changed the observed state')
        error = float(np.max(abs(target-np.asarray(row['actually_applied']))))
        prefix_errors.append(error)
        if error > 2e-6 or diagnostic['replanned']:
            raise ValueError('Prefix diverged before the audited first match: '+str(row['tick']))
    initial = records[-1]
    state = state_from_record(controller.model, initial)
    phase = initial['phase']
    manager = EventRecoveryManagerV1()
    manager.memory = RecoveryPhaseMemory(
        episode=0, last_time_s=initial['time_s'], phase=phase['phase'],
        release_time_s=phase['release_time_s'], first_ground_time_s=phase['first_ground_time_s'],
        settled_duration_s=phase['phase_settled_duration_s'],
        final_stable_duration_s=0., ground_contacts=state.contacts.copy())
    rows, failure = [], None
    begin = initial['tick']
    stable = 0.
    for tick in range(begin, 800):
        if time.monotonic() > deadline:
            failure = dict(tick=tick, reason='compute_budget')
            break
        now = float(np.float32(tick)*np.float32(.01))
        q, v = state.q.copy(), state.v.copy()
        target, diagnostic = controller.command(state, phase['phase'], now)
        if not np.array_equal(q, state.q) or not np.array_equal(v, state.v):
            raise ValueError('Controller changed the observed state')
        if (not diagnostic['replanned'] or
                diagnostic['replanning_source_key'] != match['source_key']):
            raise ValueError('Runtime did not select the audited causal plan')
        recorder = ForceRecorder(controller.model)
        try:
            prediction = predict_self_contact_spheres(
                controller.model, controller.interface, state, target, controller.geometry.spheres,
                substeps=160, geometry=controller.path_geometry, collision_response=recorder)
        except (ValueError, np.linalg.LinAlgError) as error:
            failure = dict(tick=tick, reason='model_rejected', error=repr(error))
            break
        bad = {name:value for name,value in prediction['diagnostics'].items() if
            (name == 'nonfoot_ground_violation_m' and value > 1e-6) or
            (name in ['joint_position_violation_rad', 'joint_velocity_violation_rad_s',
                      'motor_torque_violation_nm'] and value > 3e-4)}
        if bad:
            failure = dict(tick=tick, reason='physical_path_violation', violations=bad,
                collision=prediction['first_ground_violation'], source=diagnostic['source'])
            break
        state = prediction['state']
        state.previous_command_target = target.copy()
        force = recorder.last_force
        rotation = state.root_rotation_world
        # Observe AFTER every interval, including the final state at 8 s.
        phase = manager.update(episode=0,
            time_s=float(np.float32(tick+1)*np.float32(.01)),
            had_racket_contact=True, released=True, racket_touching=False, separated_now=True,
            ground_normal_force_n=force[:,2],
            roll_pitch_rad=[np.arctan2(rotation[2,1], rotation[2,2]),
                           np.arcsin(np.clip(-rotation[2,0], -1, 1))],
            root_velocity_world=rotation@state.v[:3],
            root_angular_velocity_world=rotation@state.v[3:6],
            joint_velocity=state.joint_velocity, nonfoot_contact_now=False)
        state.contacts = phase['ground_contacts'].copy()
        state.measured_force_world = force.copy()
        stable = phase['control_sample_stable_duration_s']
        rows.append(dict(tick=tick, q=state.q.copy(), v=state.v.copy(),
            target=target.copy(), ground_force=force.copy(), phase=phase['phase'],
            stable_s=stable, source=diagnostic['source'],
            reference_active=diagnostic['reference_active'],
            terminal_stance_active=diagnostic['terminal_stance_active']))
    complete = failure is None and len(rows) == 800-begin
    return dict(case=match['case'], actual_first_match_tick=begin,
        query_source_case=match['query_source_case'], query_tick=match['original_query_tick'],
        source_key=match['source_key'], match_score=match['score'],
        initial_actual_state=initial, reconstructed_prefix_steps=len(prefix_errors),
        maximum_native_prefix_readback_error_rad=max(prefix_errors, default=0.),
        steps=len(rows), required_steps=800-begin, failure=failure,
        end_time_s=8. if complete else None, final_model_stable_s=stable,
        model_terminal_window_passed=complete and stable >= .8,
        trajectory=rows, native_success=False, valid_expert_labels=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['run', 'configuration', 'library', 'matches', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    read = lambda path: json.loads(Path(path).read_text())
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if args.output.exists() or read(args.run/'job.json')['status'] != 'SUCCEEDED':
        raise ValueError('Fresh output and completed native history required')
    matches = read(args.matches)
    for data in [matches, read(args.library)]:
        for filename, digest in data['source_sha256'].items():
            if sha(filename) != digest:
                raise ValueError('Changed source '+filename)
    cfg = read(args.configuration)
    nominal = read(args.run/'native_command_inputs.json')['default_joint_position_rad']
    with np.load(args.run/'pure_control_commands.npz') as data:
        arrays = {key:data[key].copy() for key in data.files}
    diagnostics = [json.loads(line) for line in (args.run/'pure_control_diagnostics.jsonl').read_text().splitlines()]
    if len(diagnostics) != len(arrays['lane']):
        raise ValueError('Misaligned command diagnostics')
    jobs = []
    for match in matches['matches']:
        ids = np.flatnonzero((arrays['lane'] == match['case']) &
                            (arrays['tick'] <= match['actual_tick']))
        if len(ids) < 2 or np.any(np.diff(arrays['tick'][ids]) != 1):
            raise ValueError('Incomplete native prefix')
        records = []
        for j, index in enumerate(ids):
            index = int(index)
            diagnostic = diagnostics[index]
            if (diagnostic['tick'], diagnostic['lane']) != (int(arrays['tick'][index]), match['case']):
                raise ValueError('Diagnostic alignment changed')
            row = {k:arrays[k][index].copy() for k in ['q', 'v', 'previous', 'ground_force']}
            row.update(case=match['case'], tick=int(arrays['tick'][index]),
                       time_s=float(arrays['time_s'][index]), phase=diagnostic['phase'])
            if j+1 < len(ids):
                row['actually_applied'] = arrays['previous'][int(ids[j+1])].copy()
            records.append(row)
        recorded = match['first_matching_actual_state']
        for name in ['q', 'v', 'previous', 'ground_force']:
            if not np.array_equal(records[-1][name], np.asarray(recorded[name])):
                raise ValueError('First-match evidence refers to another native state')
        jobs.append((match, records))
    if not jobs:
        raise ValueError('No covered actual first match')
    PureControlPreservedStanceReplanningV17(PositionPDInterface(**cfg['interface']),
        nominal, cfg['model_reference_library'], args.library)
    files = [Path(__file__), args.configuration, args.library, args.matches,
        Path(cfg['model_reference_library'])]
    files += [args.run/name for name in ['job.json', 'native_command_inputs.json',
        'pure_control_commands.npz', 'pure_control_diagnostics.jsonl']]
    files += [Path(module.__file__) for name,module in list(sys.modules.items())
        if name.startswith('scripts.') and getattr(module, '__file__', None)]
    pins = {str(path.resolve()):sha(path) for path in files}
    args.output.mkdir(parents=True)
    def write(name, value):
        path = args.output/name
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(plain(value), indent=2, allow_nan=False)+'\n')
        temporary.replace(path)
    write('inputs.json', dict(source_sha256=pins, max_seconds=1800, physical_substeps=160,
        native_teleport=False, training=False, unchanged_acceptance=True))
    started = time.monotonic()
    results = []
    write('status.json', dict(state='RUNNING_FIRST_MATCH_FEEDBACK_MODEL',
        pid=os.getpid(), completed=0, total=len(jobs)))
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context('spawn')) as pool:
        futures = [pool.submit(evaluate, cfg, nominal, str(args.library), match,
            records, started+1800.) for match,records in jobs]
        for future in as_completed(futures):
            result = future.result()
            write('case_%03d.json'%result['case'], result)
            results.append({k:v for k,v in result.items() if k not in ['trajectory', 'initial_actual_state']})
            write('status.json', dict(state='RUNNING_FIRST_MATCH_FEEDBACK_MODEL',
                pid=os.getpid(), completed=len(results), total=len(jobs)))
    for filename,digest in pins.items():
        if sha(filename) != digest:
            raise ValueError('Changed audit input '+filename)
    result = dict(state='COMPLETED_FIRST_MATCH_FEEDBACK_MODEL', results=results,
        wall_seconds=time.monotonic()-started, native_success=False, valid_expert_labels=False)
    write('result.json', result)
    write('status.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
