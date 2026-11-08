"""Separate an exhausted contact solver from an infeasible recovery action.

Replay one already searched command sequence from its actual initial state.
Only after the original numerical iteration bound fails, retry the identical
microstep with a larger bound and unchanged equations/tolerances. This is a
CPU diagnostic, not a deployed controller or a native recovery result.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.search_current_native_recovery_v4 import knots_for
from scripts.pure_control_reference_clock_v12 import PureControlReferenceClockV12
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.audit_actual_handoff_model_v1 import state_from_record
from scripts.audit_settled_stance_model_v2 import ForceRecorder
from scripts.predict_self_contact_spheres_v2 import predict_self_contact_spheres
from scripts.event_recovery_manager_v1 import EventRecoveryManagerV1, RecoveryPhaseMemory
from scripts.audit_joint_sequence_proposal_v1 import plain


class RetryRecorder(ForceRecorder):
    def __init__(self, model, events, control_step):
        super().__init__(model)
        self.events, self.control_step, self.micro = events, control_step, 0

    def step(self, *args, **kwargs):
        micro = self.micro
        self.micro += 1
        try:
            return super().step(*args, **kwargs)
        except ValueError as error:
            if not str(error).startswith('Coupled contact/joint interval law did not converge:'):
                raise
            event = dict(control_step=self.control_step, micro=micro,
                original_error=str(error), original_iterations=400, retry_iterations=4000,
                changed_tolerance=False, solved=False)
            self.events.append(event)
            result = super().step(*args, **dict(kwargs, max_iterations=4000))
            event.update(solved=True, contact=result['contact'],
                momentum_residual=result['momentum_residual'],
                box_violation=result['joint_box_violation'], box_fixed_point=result['joint_box_fixed_point'])
            return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ['search', 'configuration', 'fixture-run', 'output']:
        p.add_argument('--' + n, type=Path, required=True)
    p.add_argument('--state', required=True)
    a = p.parse_args()
    read = lambda path: json.loads(path.read_text())
    if a.output.exists() or read(a.search/'result.json')['state'] != 'COMPLETED_ACTUAL_RECOVERY_SEQUENCE_SEARCH':
        raise ValueError('Fresh audit of a completed search required')
    for f, h in read(a.search/'inputs.json')['source_sha256'].items():
        if hashlib.sha256(Path(f).read_bytes()).hexdigest() != h:
            raise ValueError('Changed search input ' + f)
    source = a.search / ('state_' + a.state.replace(':', '_') + '.json')
    data = read(source)
    if data['fine']['failure']['reason'] != 'model_rejected':
        raise ValueError('Only diagnose a numerically rejected fine path')
    record = data['initial_actual_state']
    cfg = read(a.configuration)
    nominal = read(a.fixture_run/'native_command_inputs.json')['default_joint_position_rad']
    c = PureControlReferenceClockV12(PositionPDInterface(**cfg['interface']), nominal, cfg['model_reference_library'])
    state = state_from_record(c.model, record)
    limits = c.interface.limits
    targets = limits[:, 0] + .5 * (np.asarray(data['fine']['nodes']) + 1.) * np.diff(limits).ravel()
    curve = PchipInterpolator(knots_for(record), targets, axis=0)
    phase = record['phase']
    manager = EventRecoveryManagerV1()
    manager.memory = RecoveryPhaseMemory(episode=0, last_time_s=record['time_s'], phase=phase['phase'],
        release_time_s=phase['release_time_s'], first_ground_time_s=phase['first_ground_time_s'],
        settled_duration_s=phase['phase_settled_duration_s'], final_stable_duration_s=0.,
        ground_contacts=state.contacts.copy())
    events, rows = [], []
    prefix_error = 0.
    failure = None
    started = time.monotonic()
    for k in range(800-record['tick']):
        if time.monotonic()-started > 900:
            failure = dict(step=k, reason='compute_budget'); break
        command = np.clip(curve(k*.01), limits[:, 0], limits[:, 1])
        recorder = RetryRecorder(c.model, events, k)
        try:
            result = predict_self_contact_spheres(c.model, c.interface, state, command, c.geometry.spheres,
                substeps=160, geometry=c.path_geometry, collision_response=recorder)
        except (ValueError, np.linalg.LinAlgError) as error:
            failure = dict(step=k, reason='model_rejected', error=repr(error)); break
        bad = {n:v for n,v in result['diagnostics'].items() if
            (n == 'nonfoot_ground_violation_m' and v > 1e-6) or
            (n in ['joint_position_violation_rad', 'joint_velocity_violation_rad_s', 'motor_torque_violation_nm'] and v > 3e-4)}
        if bad:
            failure = dict(step=k, reason='physical_path_violation', violations=bad,
                collision=result['first_ground_violation']); break
        state = result['state']
        state.previous_command_target = command.copy()
        if k < len(data['fine']['trajectory']):
            old = data['fine']['trajectory'][k]
            prefix_error = max(prefix_error, float(np.max(abs(state.q-np.asarray(old['q'])))),
                float(np.max(abs(state.v-np.asarray(old['v'])))))
            if prefix_error > 1e-12:
                raise ValueError('The unchanged pre-retry trajectory differs')
        r = state.root_rotation_world
        phase = manager.update(episode=0, time_s=float(np.float32(record['tick']+k+1)*np.float32(.01)),
            had_racket_contact=True, released=True, racket_touching=False, separated_now=True,
            ground_normal_force_n=recorder.last_force[:, 2],
            roll_pitch_rad=[np.arctan2(r[2,1], r[2,2]), np.arcsin(np.clip(-r[2,0], -1, 1))],
            root_velocity_world=r@state.v[:3], root_angular_velocity_world=r@state.v[3:6],
            joint_velocity=state.joint_velocity, nonfoot_contact_now=False)
        state.contacts = phase['ground_contacts'].copy()
        state.measured_force_world = recorder.last_force.copy()
        rows.append(dict(step=k, q=state.q.copy(), v=state.v.copy(), command=command,
            stable_s=phase['control_sample_stable_duration_s']))
    files = [Path(__file__), source, a.search/'inputs.json', a.configuration,
        a.fixture_run/'native_command_inputs.json', Path(cfg['model_reference_library'])]
    files += [Path(m.__file__) for n,m in list(sys.modules.items())
        if n.startswith('scripts.') and getattr(m, '__file__', None)]
    output = dict(state_key=a.state, steps=len(rows), required_steps=800-record['tick'],
        failure=failure, retries=events, prefix_max_error=prefix_error,
        final_model_stable_s=rows[-1]['stable_s'] if rows else 0., trajectory=rows,
        wall_seconds=time.monotonic()-started, native_success=False, valid_expert_labels=False,
        physics_and_tolerances_changed=False,
        source_sha256={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = a.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(plain(output), indent=2, allow_nan=False)+'\n')
    temporary.replace(a.output)
    print(json.dumps(plain({k:v for k,v in output.items() if k not in ['trajectory','source_sha256']})))


if __name__ == '__main__':
    main()
