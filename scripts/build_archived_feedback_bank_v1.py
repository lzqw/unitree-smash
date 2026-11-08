"""Package completed current-state V15 feedback as model-only initial references.

Recompute the original final stability window, retain the parent bank verbatim,
and preserve provenance. This does not launch native execution or certify it.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import numpy as np
import pinocchio as pin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.retry_budgeted_settling_feedback_v1 import sha, write
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1
from scripts.event_recovery_manager_v1 import EventRecoveryManagerV1, RecoveryPhaseMemory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['parent-library', 'feedback', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    a = parser.parse_args()
    read = lambda path: json.loads(path.read_text())
    if a.output.exists():
        raise ValueError('Fresh immutable preparation required')
    parent = read(a.parent_library)
    inputs = read(a.feedback / 'inputs.json')
    result = read(a.feedback / 'result.json')
    if result['state'] != 'COMPLETED_ARCHIVED_REFERENCE_RECHECK' or inputs['controller'] != 'V15' or inputs['physical_substeps'] != 160:
        raise ValueError('Completed original V15 feedback required')
    pins = {}
    for source in [parent, inputs]:
        for path, digest in source['source_sha256'].items():
            if sha(path) != digest or (path in pins and pins[path] != digest):
                raise ValueError('Changed or conflicting evidence ' + path)
            pins[path] = digest
    provisional_path = a.feedback / 'provisional_reference_library.json'
    provisional = read(provisional_path)
    new_ids = {r['source_key'] for r in result['coverage']['new_matches']}
    old = [r for r in provisional['references'] if r['source_key'] not in new_ids]
    if parent['references'][:len(old)] != old:
        raise ValueError('Parent no longer preserves the validated original bank')
    additions = [copy.deepcopy(r) for r in provisional['references'] if r['source_key'] in new_ids]
    if len(additions) != len(new_ids) or any(r['source_key'] in new_ids for r in parent['references']):
        raise ValueError('Ambiguous or duplicate additions')
    model = LoadedPinocchioModelV1()
    checks = []
    files = [Path(__file__), a.parent_library, a.feedback / 'inputs.json', a.feedback / 'result.json', provisional_path]
    for entry in additions:
        match = next(r for r in result['coverage']['new_matches'] if r['source_key'] == entry['source_key'])
        key = str(match['case']) + ':' + str(match['actual_tick'])
        path = a.feedback / ('state_' + key.replace(':', '_') + '.json')
        snapshot = a.feedback / 'candidate_snapshots' / path.name
        data, source = read(path), read(snapshot)
        initial, trace = data['initial_actual_state'], data['trajectory']
        summaries = [r for r in result['results'] if r['state_key'] == key]
        if len(summaries) != 1 or any(summaries[0][f] != data[f] for f in ['steps', 'failure', 'full_model_path', 'final_model_stable_s']):
            raise ValueError('Worker result differs from owner summary')
        if (initial != source['initial_actual_state'] or source['current_reference_key'] != entry['source_key']
                or entry['trajectory'] != source['fine']['trajectory'] or data['failure'] is not None
                or not data['full_model_path'] or data['native_success'] or data['valid_expert_labels']
                or len(trace) != 800 - initial['tick']):
            raise ValueError('Incomplete or mismatched model feedback')
        if entry['query'] != dict(q=initial['q'], v=initial['v'], target=initial['previous'], contacts=initial['phase']['ground_contacts']):
            raise ValueError('Model and entry query differ')
        phase = initial['phase']
        manager = EventRecoveryManagerV1()
        manager.memory = RecoveryPhaseMemory(episode=0, last_time_s=initial['time_s'], phase=phase['phase'],
            release_time_s=phase['release_time_s'], first_ground_time_s=phase['first_ground_time_s'],
            settled_duration_s=phase['phase_settled_duration_s'], final_stable_duration_s=0.,
            ground_contacts=np.asarray(phase['ground_contacts'], bool))
        for i, row in enumerate(trace):
            tick = initial['tick'] + i
            if row['tick'] != tick:
                raise ValueError('Discontinuous feedback')
            q, v = np.asarray(row['q']), np.asarray(row['v'])
            rotation = pin.Quaternion(q[3:7]).matrix()
            now = float(np.float32(tick + 1) * np.float32(.01))
            phase = manager.update(episode=0, time_s=now, had_racket_contact=True, released=True,
                racket_touching=False, separated_now=True, ground_normal_force_n=np.asarray(row['ground_force'])[:, 2],
                roll_pitch_rad=[np.arctan2(rotation[2, 1], rotation[2, 2]), np.arcsin(np.clip(-rotation[2, 0], -1, 1))],
                root_velocity_world=rotation @ v[:3], root_angular_velocity_world=rotation @ v[3:6],
                joint_velocity=v[model.v_indices], nonfoot_contact_now=False)
        stable = phase['control_sample_stable_duration_s']
        if now != 8. or stable < .8 or abs(stable - data['final_model_stable_s']) > 1e-9:
            raise ValueError('Original end-of-episode stability not reproduced')
        checks.append(dict(state_key=key, complete_model_feedback=True, end_time_s=now,
            end_observation_stable_s=stable, model_terminal_window_passed=True,
            native_success=False, valid_expert_labels=False))
        entry.update(source_actual_tick=initial['tick'], source_search=str(snapshot.resolve()),
            source_feedback=str(path.resolve()), requires_controller='PureControlStanceFootReferenceV15',
            reference_open_loop_complete=False, model_feedback_stable_s=stable,
            native_success=False, valid_expert_labels=False)
        files += [path, snapshot]
    files += [Path(m.__file__) for name, m in list(sys.modules.items())
              if name.startswith('scripts.') and getattr(m, '__file__', None)]
    pins.update({str(path.resolve()): sha(path) for path in files})
    a.output.mkdir(parents=True)
    end = a.output / 'end_check.json'
    write(end, dict(results=checks, source_sha256=pins, acceptance_changed=False,
                    native_success=False, valid_expert_labels=False))
    bank = dict(parent, references=parent['references'] + additions,
        source_sha256={**pins, str(end.resolve()): sha(end)},
        requires_controller='PureControlStanceFootReferenceV15',
        original_reference_count=len(parent['references']), added_reference_count=len(additions),
        model_feedback_is_not_native_success=True, native_success=False, valid_expert_labels=False, training=False)
    write(a.output / 'reference_library.json', bank)
    print(json.dumps(dict(references=len(bank['references']), additions=len(additions), end_checks=checks,
                          native_execution=False)))


if __name__ == '__main__':
    main()
