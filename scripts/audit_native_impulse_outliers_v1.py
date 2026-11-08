"""Compare integrated foot impulses on recorded, clean response outliers.

No controller/physics edits, parameter fitting, native teleportation or expert
labels. Linear momentum is reconstructed using authored body inertias; this
does not identify a unique cause or certify long-horizon model accuracy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import pinocchio as pin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_native_contact_retry_v3 import initialize
import scripts.audit_native_contact_retry_v3 as setup
from scripts.audit_actual_handoff_model_v1 import state_from_record
from scripts.hip_mount_collision_response_v1 import HipMountCollisionResponseV1
from scripts.predict_self_contact_spheres_v2 import predict_self_contact_spheres
from scripts.audit_joint_sequence_proposal_v1 import plain
from scripts.audit_native_landing_friction_v1 import interval_balance


class Recorder(HipMountCollisionResponseV1):
    def __init__(self, model):
        super().__init__(model)
        self.impulses = []

    def step(self, *args, **kwargs):
        result = super().step(*args, **kwargs)
        self.impulses.append(result['impulse'].reshape(4, 3).copy())
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['run', 'configuration', 'baseline', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    read = lambda p: json.loads(Path(p).read_text())
    def sha(path):
        digest = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        return digest.hexdigest()
    if args.output.exists() or read(args.run / 'job.json')['status'] != 'SUCCEEDED':
        raise ValueError('Fresh output and completed native run required')
    metadata = read(args.run / 'landing_contact_detail.json')
    if not metadata['complete'] or metadata['maximum_points_per_pair'] > 8:
        raise ValueError('Complete untruncated impulse telemetry required')
    cfg = read(args.configuration)
    initialize(cfg, read(args.run / 'native_command_inputs.json')['default_joint_position_rad'])
    c = setup.CONTROL
    mass = sum(i.mass for i in c.model.model.inertias)
    def arrays(name, keys):
        with np.load(args.run / name) as z:
            return {key: z[key].copy() for key in keys}
    data = arrays('pure_control_commands.npz', ['lane', 'tick', 'episode', 'q', 'v', 'previous', 'final', 'ground_force'])
    detail = arrays('landing_contact_detail.npz', ['normal_impulse', 'friction_impulse', 'net_peak'])
    ground = arrays('pure_control_ground_forces.npz', ['interval_peak_normal_n'])['interval_peak_normal_n']
    names = read(args.run / 'pure_control_ground_forces.json')['body_names']
    nonfoot = [i for i, name in enumerate(names) if not name.endswith('_foot')]
    diagnostics = [json.loads(line) for line in (args.run / 'pure_control_diagnostics.jsonl').read_text().splitlines()]
    if len(diagnostics) != len(data['lane']):
        raise ValueError('Misaligned commands and phases')
    baseline = read(args.baseline)['rows']
    outliers = [(33, 298), (96, 306), (87, 304), (71, 301)]
    controls = [(r['case'], r['tick']) for r in baseline if r['case'] in [30, 42, 50, 85, 90]
                and r['label'] == 'first_supported_interval']
    if len(controls) != 5:
        raise ValueError('Expected five verified-case comparison intervals')
    paths = [Path(__file__), args.configuration, args.baseline, Path(cfg['model_reference_library'])]
    paths += [args.run / name for name in ['job.json', 'native_command_inputs.json',
        'pure_control_commands.npz', 'pure_control_diagnostics.jsonl', 'pure_control_ground_forces.npz',
        'pure_control_ground_forces.json', 'landing_contact_detail.npz', 'landing_contact_detail.json']]
    paths += [Path(module.__file__) for name, module in list(sys.modules.items())
              if name.startswith('scripts.') and getattr(module, '__file__', None)]
    pins = {str(path.resolve()): sha(path) for path in paths}
    args.output.mkdir(parents=True)
    def write(name, value):
        path = args.output / name
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + '\n')
        temporary.replace(path)
    write('inputs.json', dict(source_sha256=pins, outliers=outliers, controls=controls,
        maximum_seconds=300, substeps=160, hardware_changed=False, no_native_state_restore=True))
    started = time.monotonic()
    rows = []
    for case, tick in outliers + controls:
        if time.monotonic() - started > 300:
            raise TimeoutError('Bounded diagnostic expired')
        ids = [np.flatnonzero((data['lane'] == case) & (data['tick'] == t)) for t in [tick, tick + 1]]
        if any(len(i) != 1 for i in ids):
            raise ValueError('Unique consecutive observations required')
        i, j = [int(x[0]) for x in ids]
        if (data['episode'][i] != data['episode'][j]
            or ground[tick, case, nonfoot].max() > 0 or detail['net_peak'][tick, case].max() > 0):
            raise ValueError('Selected interval crosses reset or contains external nonfoot contact')
        if any((diagnostics[k]['lane'], diagnostics[k]['tick']) != (case, t)
               for k, t in [(i, tick), (j, tick + 1)]):
            raise ValueError('Misaligned selected phase')
        record = {key: data[key][i] for key in ['q', 'v', 'previous', 'final', 'ground_force']}
        record['phase'] = diagnostics[i]['phase']
        state = state_from_record(c.model, record)
        h0 = c.model.snapshot(state.q, state.v)['momentum_world'][:3].copy()
        h1 = c.model.snapshot(data['q'][j], data['v'][j])['momentum_world'][:3].copy()
        normal = detail['normal_impulse'][tick, case]
        friction = detail['friction_impulse'][tick, case]
        row = dict(case=case, tick=tick, selected_response_outlier=(case, tick) in outliers,
            actual_clean_ground_and_net_interval=True, predicted=False,
            measured_impulse_per_foot_ns=normal + friction,
            native_momentum_balance=interval_balance(h0, h1, mass, normal, friction))
        recorder = Recorder(c.model)
        try:
            prediction = predict_self_contact_spheres(c.model, c.interface, state, record['final'],
                c.geometry.spheres, substeps=160, geometry=c.path_geometry, collision_response=recorder)
            y = prediction['state']
            if len(recorder.impulses) != 160:
                raise ValueError('Incomplete microstep impulse recording')
            impulse = np.sum(recorder.impulses, axis=0)
            hp = c.model.snapshot(y.q, y.v)['momentum_world'][:3].copy()
            actual_r = pin.Quaternion(data['q'][j][3:7]).matrix()
            model_r = pin.Quaternion(y.q[3:7]).matrix()
            errors = abs(y.v[c.model.v_indices] - data['v'][j][c.model.v_indices])
            old = next(r for r in baseline if (r['case'], r['tick']) == (case, tick))
            observed_error = float(np.max(errors))
            if abs(observed_error - old['max_joint_velocity_error_rad_s']) > 1e-10:
                raise ValueError('Impulse recorder changed original prediction')
            row.update(predicted=True, model_impulse_per_foot_ns=impulse,
                total_impulse_error_ns=float(np.linalg.norm((impulse-normal-friction).sum(0))),
                per_foot_impulse_error_ns=np.linalg.norm(impulse-normal-friction, axis=1),
                model_momentum_balance_residual_ns=hp-h0-np.array([0., 0., -mass*9.81*.01])-impulse.sum(0),
                max_joint_velocity_error_rad_s=observed_error,
                linear_velocity_error_m_s=float(np.linalg.norm(model_r@y.v[:3]-actual_r@data['v'][j][:3])),
                angular_velocity_error_rad_s=float(np.linalg.norm(model_r@y.v[3:6]-actual_r@data['v'][j][3:6])),
                original_prediction_metric_change=abs(observed_error-old['max_joint_velocity_error_rad_s']),
                model_nonfoot_violation_m=prediction['diagnostics']['nonfoot_ground_violation_m'])
        except (ValueError, np.linalg.LinAlgError) as error:
            row['rejection'] = repr(error)
        rows.append(row)
        write('status.json', dict(state='RUNNING', completed=len(rows), total=9))
    for path, digest in pins.items():
        if sha(path) != digest:
            raise ValueError('Source changed during audit: ' + path)
    result = dict(state='COMPLETED_IMPULSE_COMPARISON', rows=rows, source_sha256=pins,
        wall_seconds=time.monotonic()-started, native_success=False, valid_expert_labels=False,
        model_calibrated=False, control_changed=False, selected_sample_not_population_estimate=True,
        linear_momentum_uses_authored_inertias=True)
    write('result.json', result)
    write('status.json', dict(state=result['state'], completed=len(rows), returned=sum(r['predicted'] for r in rows)))
    print(json.dumps(plain(rows)))


if __name__ == '__main__':
    main()
