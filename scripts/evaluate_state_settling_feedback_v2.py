"""Compare both settling rules with fine-model candidate selection.

Reuse the preceding immutable candidate snapshots, including failed candidates.
Their checked prefixes are only experimental references, not expert labels.
Both controllers execute from the original native observation to time 8 s;
no predicted intermediate state is passed off as a native reset.
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
from scripts.pure_control_fine_selector_v24 import PureControlFinePlanEndV24, PureControlFineStateSettlingV24
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.audit_actual_handoff_model_v1 import state_from_record
from scripts.audit_settled_stance_model_v2 import ForceRecorder
from scripts.event_recovery_manager_v1 import EventRecoveryManagerV1, RecoveryPhaseMemory
from scripts.predict_self_contact_spheres_v2 import predict_self_contact_spheres
from scripts.audit_joint_sequence_proposal_v1 import plain


def evaluate(cfg, nominal, path, mode, deadline):
    data = json.loads(Path(path).read_text())
    record = data['initial_actual_state']
    cls = PureControlFinePlanEndV24 if mode == 'plan_end' else PureControlFineStateSettlingV24
    c = cls(PositionPDInterface(**cfg['interface']), nominal, cfg['model_reference_library'])
    state = state_from_record(c.model, record)
    phase = record['phase']
    c.chosen = dict(query=dict(q=record['q'], v=record['v'],
        contacts=phase['ground_contacts'], target=record['previous']),
        trajectory=data['fine']['trajectory'],
        source_key=hashlib.sha256(Path(path).read_bytes()).hexdigest(), native_success=False)
    c.selection_done = True
    c.selection_score = None
    manager = EventRecoveryManagerV1()
    manager.memory = RecoveryPhaseMemory(episode=0, last_time_s=record['time_s'],
        phase=phase['phase'], release_time_s=phase['release_time_s'],
        first_ground_time_s=phase['first_ground_time_s'],
        settled_duration_s=phase['phase_settled_duration_s'],
        final_stable_duration_s=0., ground_contacts=state.contacts.copy())
    rows = []
    failure = None
    capture_tick = None
    for tick in range(record['tick'], 800):
        if time.monotonic() > deadline:
            failure = dict(tick=tick, reason='compute_budget')
            break
        now = float(np.float32(tick)*np.float32(.01))
        target, diag = c.command(state, phase['phase'], now)
        if diag.get('state_settling_capture_now'):
            capture_tick = tick
        recorder = ForceRecorder(c.model)
        try:
            pred = predict_self_contact_spheres(c.model, c.interface, state, target,
                c.geometry.spheres, substeps=160, geometry=c.path_geometry,
                collision_response=recorder)
        except (ValueError, np.linalg.LinAlgError) as error:
            failure = dict(tick=tick, reason='model_rejected', error=repr(error))
            break
        bad = {n:v for n,v in pred['diagnostics'].items() if
            (n == 'nonfoot_ground_violation_m' and v > 1e-6) or
            (n in ['joint_position_violation_rad', 'joint_velocity_violation_rad_s',
                'motor_torque_violation_nm'] and v > 3e-4)}
        if bad:
            failure = dict(tick=tick, reason='physical_path_violation',
                violations=bad, collision=pred['first_ground_violation'])
            break
        state = pred['state']
        state.previous_command_target = target.copy()
        force = recorder.last_force
        r = state.root_rotation_world
        phase = manager.update(episode=0,
            time_s=float(np.float32(tick+1)*np.float32(.01)),
            had_racket_contact=True, released=True, racket_touching=False, separated_now=True,
            ground_normal_force_n=force[:, 2],
            roll_pitch_rad=[np.arctan2(r[2, 1], r[2, 2]), np.arcsin(np.clip(-r[2, 0], -1, 1))],
            root_velocity_world=r@state.v[:3], root_angular_velocity_world=r@state.v[3:6],
            joint_velocity=state.joint_velocity, nonfoot_contact_now=False)
        state.contacts = phase['ground_contacts'].copy()
        state.measured_force_world = force.copy()
        rows.append(dict(tick=tick, q=state.q.copy(), v=state.v.copy(), target=target.copy(),
            ground_force=force.copy(), phase=phase, source=diag['source'],
            solver_accepted=diag['solver_accepted'],
            terminal_stance_active=diag['terminal_stance_active'],
            reference_active=diag['reference_active'],
            zero_cost_planned_candidate=diag['zero_cost_planned_candidate'],
            feedback_candidate_substeps=diag['feedback_candidate_substeps']))
    return dict(state_key=data['state_key'], mode=mode, steps=len(rows),
        required_steps=800-record['tick'], failure=failure,
        full_model_path=failure is None and len(rows)==800-record['tick'],
        final_model_stable_s=phase['control_sample_stable_duration_s'],
        state_settling_capture_tick=capture_tick, trajectory=rows,
        initial_actual_state=record, native_success=False, valid_expert_labels=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['search', 'configuration', 'run', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    a = parser.parse_args()
    read = lambda p: json.loads(p.read_text())
    def sha(p):
        h = hashlib.sha256()
        with Path(p).open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024), b''):
                h.update(block)
        return h.hexdigest()
    if a.output.exists() or read(a.run/'job.json')['status'] != 'SUCCEEDED':
        raise ValueError('Fresh paired diagnostic from completed native evidence required')
    inputs = read(a.search/'inputs.json')
    for path, digest in inputs['source_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Changed search input '+path)
    cfg = read(a.configuration)
    if inputs['source_sha256'].get(str(a.configuration.resolve())) != sha(a.configuration):
        raise ValueError('Keep original execution interface')
    nominal = read(a.run/'native_command_inputs.json')['default_joint_position_rad']
    # Reuse the exact frozen candidates from the preceding paired experiment.
    snapshots = []
    with np.load(a.run/'pure_control_commands.npz') as z:
        for path in sorted((a.search/'candidate_snapshots').glob('state_*.json')):
            raw = path.read_bytes()
            data = json.loads(raw)
            record = data['initial_actual_state']
            fine = data['fine']
            if fine['substeps'] != 160 or not fine['trajectory'] or data['native_success'] or data['valid_expert_labels']:
                raise ValueError('Require non-expert fine planning candidate')
            ids = np.flatnonzero((z['lane']==record['case']) & (z['tick']==record['tick']))
            if len(ids) != 1 or any(not np.array_equal(z[k][ids[0]], record[k]) for k in ['q','v','previous','ground_force']):
                raise ValueError('Candidate is not rooted at the recorded actual state')
            snapshots.append((path.name, raw))
    if not snapshots:
        raise ValueError('No candidates available')
    a.output.mkdir(parents=True)
    snapdir = a.output/'candidate_snapshots'
    snapdir.mkdir()
    for name, raw in snapshots:
        (snapdir/name).write_bytes(raw)
    def write(name, value):
        path = a.output/name
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(plain(value), indent=2, allow_nan=False)+'\n')
        tmp.replace(path)
    c = PureControlFineStateSettlingV24(PositionPDInterface(**cfg['interface']), nominal, cfg['model_reference_library'])
    files = [Path(__file__), a.configuration, a.run/'native_command_inputs.json', a.search/'inputs.json']
    files += list(snapdir.glob('*.json'))
    files += [Path(m.__file__) for n,m in list(sys.modules.items()) if n.startswith('scripts.') and getattr(m,'__file__',None)]
    pins = dict(inputs['source_sha256'])
    pins.update({str(f.resolve()):sha(f) for f in files})
    write('inputs.json', dict(source_sha256=pins, workers=4, max_seconds=1800,
        physical_substeps=160, source_search_may_still_be_running=False,
        feedback_candidate_substeps=160, reused_frozen_snapshots=True,
        reference_snapshots=[str((snapdir/name).resolve()) for name,_ in snapshots],
        candidates_are_unverified_references=True, native_teleport=False,
        hardware_changed=False, acceptance_changed=False, training=False))
    started = time.monotonic()
    results = []
    write('status.json', dict(state='RUNNING_FINE_SELECTOR_SETTLING_FEEDBACK',pid=os.getpid(),completed=0,total=2*len(snapshots)))
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context('spawn')) as pool:
        jobs = [pool.submit(evaluate,cfg,nominal,str(snapdir/name),mode,started+1800)
            for name,_ in snapshots for mode in ['plan_end','state_settling']]
        for future in as_completed(jobs):
            result = future.result()
            write(result['state_key'].replace(':','_')+'_'+result['mode']+'.json',result)
            results.append({k:v for k,v in result.items() if k not in ['trajectory','initial_actual_state']})
            write('status.json',dict(state='RUNNING_FINE_SELECTOR_SETTLING_FEEDBACK',pid=os.getpid(),
                completed=len(results),total=len(jobs),results=results))
    for path,digest in pins.items():
        if sha(path) != digest:
            raise ValueError('Changed diagnostic input '+path)
    summary = dict(state='COMPLETED_FINE_SELECTOR_SETTLING_FEEDBACK', results=results,
        wall_seconds=time.monotonic()-started, native_success=False,valid_expert_labels=False)
    write('result.json',summary)
    write('status.json',summary)
    print(json.dumps(plain(summary)),flush=True)


if __name__ == '__main__':
    main()
