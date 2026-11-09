"""Full model feedback for new initial references in the complete V20 controller.

Uses current recorded native release states and natural state matching. No
forced reference selection, native reset, learning, or performance promotion.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.pure_control_preserved_foot_replanning_v20 import PureControlPreservedFootReplanningV20
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.audit_actual_handoff_model_v1 import state_from_record
from scripts.audit_settled_stance_model_v2 import ForceRecorder
from scripts.event_recovery_manager_v1 import EventRecoveryManagerV1, RecoveryPhaseMemory
from scripts.predict_self_contact_spheres_v2 import predict_self_contact_spheres
from scripts.retry_budgeted_settling_feedback_v1 import sha, write


def evaluate(cfg, nominal, record, expected_source, deadline):
    c = PureControlPreservedFootReplanningV20(PositionPDInterface(**cfg['interface']),
        nominal, cfg['model_reference_library'], cfg['model_replanning_library'])
    state = state_from_record(c.model, record)
    phase = record['phase']
    manager = EventRecoveryManagerV1()
    manager.memory = RecoveryPhaseMemory(episode=0, last_time_s=record['time_s'],
        phase=phase['phase'], release_time_s=phase['release_time_s'],
        first_ground_time_s=phase['first_ground_time_s'],
        settled_duration_s=phase['phase_settled_duration_s'],
        final_stable_duration_s=0., ground_contacts=state.contacts.copy())
    rows, failure = [], None
    for tick in range(record['tick'], 800):
        if time.monotonic() > deadline:
            failure = dict(tick=tick, reason='compute_budget'); break
        q, v = state.q.copy(), state.v.copy()
        target, diag = c.command(state, phase['phase'], float(np.float32(tick)*np.float32(.01)))
        if not np.array_equal(q, state.q) or not np.array_equal(v, state.v):
            raise ValueError('Controller modified the observed state')
        if tick == record['tick'] and diag.get('offline_reference_source_key') != expected_source:
            raise ValueError('Natural initial selection differs from the proposed reference')
        recorder = ForceRecorder(c.model)
        try:
            pred = predict_self_contact_spheres(c.model, c.interface, state, target,
                c.geometry.spheres, substeps=160, geometry=c.path_geometry,
                collision_response=recorder)
        except (ValueError, np.linalg.LinAlgError) as error:
            failure = dict(tick=tick, reason='model_rejected', error=repr(error)); break
        bad = {n: value for n, value in pred['diagnostics'].items() if
            (n == 'nonfoot_ground_violation_m' and value > 1e-6) or
            (n in ['joint_position_violation_rad', 'joint_velocity_violation_rad_s',
                   'motor_torque_violation_nm'] and value > 3e-4)}
        if bad:
            failure = dict(tick=tick, reason='physical_path_violation',
                violations=bad, collision=pred['first_ground_violation']); break
        state = pred['state']; state.previous_command_target = target.copy()
        force = recorder.last_force; rotation = state.root_rotation_world
        phase = manager.update(episode=0,
            time_s=float(np.float32(tick+1)*np.float32(.01)), had_racket_contact=True,
            released=True, racket_touching=False, separated_now=True,
            ground_normal_force_n=force[:, 2],
            roll_pitch_rad=[np.arctan2(rotation[2, 1], rotation[2, 2]),
                           np.arcsin(np.clip(-rotation[2, 0], -1, 1))],
            root_velocity_world=rotation@state.v[:3],
            root_angular_velocity_world=rotation@state.v[3:6],
            joint_velocity=state.joint_velocity, nonfoot_contact_now=False)
        state.contacts = phase['ground_contacts'].copy(); state.measured_force_world = force.copy()
        rows.append(dict(tick=tick, q=state.q.copy(), v=state.v.copy(), target=target.copy(),
            ground_force=force.copy(), phase=phase, source=diag['source'],
            replanned=diag['replanned'], replanning_source_key=diag.get('replanning_source_key')))
    full = failure is None and len(rows) == 800-record['tick']
    return dict(state_key=str(record['case'])+':'+str(record['tick']), initial_actual_state=record,
        expected_initial_reference=expected_source, steps=len(rows), required_steps=800-record['tick'],
        full_model_path=full, failure=failure, end_time_s=8. if full else None,
        final_model_stable_s=phase['control_sample_stable_duration_s'],
        model_terminal_window_passed=full and phase['control_sample_stable_duration_s'] >= .8,
        replanned_steps=sum(bool(r['replanned']) for r in rows), trajectory=rows,
        native_success=False, valid_expert_labels=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['baseline', 'configuration', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(); read = lambda p: json.loads(Path(p).read_text())
    if args.output.exists(): raise ValueError('Fresh output required')
    baseline = read(args.baseline); cfg = read(args.configuration)
    if baseline['version'] != 'pure_control_preserved_foot_replanning_v20':
        raise ValueError('Complete verified V20 required')
    old_cfg = read(baseline['configuration']); run = Path(baseline['native_run'])
    if read(run/'job.json')['status'] != 'SUCCEEDED': raise ValueError('Complete native history required')
    for key in set(cfg) | set(old_cfg):
        if key not in ['method', 'model_reference_library'] and cfg.get(key) != old_cfg.get(key):
            raise ValueError('Changed protected execution setting '+key)
    bank = read(cfg['model_reference_library']); old = read(old_cfg['model_reference_library'])
    if bank['references'][:len(old['references'])] != old['references']:
        raise ValueError('Old references changed')
    added = bank['references'][len(old['references']):]
    if not added or bank['native_success'] or bank['valid_expert_labels']:
        raise ValueError('Model-only additions required')
    pins = {}
    for data in [baseline, bank, read(cfg['model_replanning_library'])]:
        for filename, digest in data['source_sha256'].items():
            if sha(filename) != digest or (filename in pins and pins[filename] != digest):
                raise ValueError('Changed or conflicting input '+filename)
            pins[filename] = digest
    diag = [json.loads(s) for s in (run/'pure_control_diagnostics.jsonl').read_text().splitlines()]
    records = []
    with np.load(run/'pure_control_commands.npz') as z:
        if len(diag) != len(z['lane']): raise ValueError('Misaligned native history')
        for reference in added:
            ids = np.flatnonzero(z['lane'] == reference['source_case'])
            if not len(ids): raise ValueError('Missing native release history')
            i = int(ids[0]); row = diag[i]
            if (row['tick'], row['lane']) != (int(z['tick'][i]), int(z['lane'][i])):
                raise ValueError('Native record alignment mismatch')
            if int(z['tick'][i]) != reference['source_actual_tick']:
                raise ValueError('Reference is not at the actual initial takeover')
            record = {k: z[k][i].copy() for k in ['q', 'v', 'previous', 'ground_force']}
            record.update(case=int(z['lane'][i]), tick=int(z['tick'][i]),
                time_s=float(z['time_s'][i]), phase=row['phase'])
            records.append((record, reference['source_key']))
    nominal = read(run/'native_command_inputs.json')['default_joint_position_rad']
    PureControlPreservedFootReplanningV20(PositionPDInterface(**cfg['interface']), nominal,
        cfg['model_reference_library'], cfg['model_replanning_library'])
    files = [Path(__file__), args.baseline, args.configuration,
        Path(cfg['model_reference_library']), Path(cfg['model_replanning_library'])]
    files += [run/n for n in ['job.json', 'native_command_inputs.json',
                             'pure_control_commands.npz', 'pure_control_diagnostics.jsonl']]
    files += [Path(m.__file__) for n, m in list(sys.modules.items())
              if n.startswith('scripts.') and getattr(m, '__file__', None)]
    pins.update({str(p.resolve()): sha(p) for p in files})
    args.output.mkdir(parents=True)
    write(args.output/'inputs.json', dict(source_sha256=pins, records=records,
        workers=4, max_seconds=2000, physical_substeps=160, scoring_substeps=16,
        forced_selection=False, controller='V20', training=False, native_execution=False))
    start = time.monotonic(); results = []
    def status(state):
        write(args.output/'status.json', dict(state=state, pid=os.getpid(),
            completed=len(results), total=len(records), results=results))
    status('RUNNING_COMBINED_INITIAL_FEEDBACK')
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context('spawn')) as pool:
        jobs = [pool.submit(evaluate, cfg, nominal, record, key, start+2000) for record, key in records]
        for job in as_completed(jobs):
            row = job.result(); write(args.output/('state_'+row['state_key'].replace(':', '_')+'.json'), row)
            results.append({k: v for k, v in row.items() if k not in ['trajectory', 'initial_actual_state']})
            status('RUNNING_COMBINED_INITIAL_FEEDBACK')
    for filename, digest in pins.items():
        if sha(filename) != digest: raise ValueError('Changed feedback input '+filename)
    final = dict(state='COMPLETED_COMBINED_INITIAL_FEEDBACK', results=results,
        elapsed_s=time.monotonic()-start, native_success=False, valid_expert_labels=False)
    write(args.output/'result.json', final); status(final['state']); print(json.dumps(final))


if __name__ == '__main__':
    main()
