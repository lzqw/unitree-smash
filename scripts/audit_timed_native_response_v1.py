"""Compare native feedback with nominal and measured-state one-step predictions.

Nominal trajectory rows contain END states; native trace rows contain START
states. This audit deliberately compares nominal tick t with native tick t+1.
One-step replay uses the actual dispatched command and previous command,
without replacing the measured state by the nominal reference. Neither kind
of error is evidence of a physical impossibility or a valid expert label.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pinocchio as pin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.check_constrained_recovery_suffix_v1 import source_context, state_from_row
from scripts.search_preserved_early_support_v1 import initialize
from scripts.search_landing_support_tail_v3 import _WORKER
from scripts.hip_mount_path_constraint_v1 import HipMountPathConstraintV1
from scripts.predict_speed_limited_spheres_v1 import predict_speed_limited_spheres
from scripts.audit_joint_sequence_proposal_v1 import plain


def state_error(model, predicted, actual):
    dq = pin.difference(model.model, np.asarray(predicted['q']), np.asarray(actual['q']))
    dv = np.asarray(actual['v']) - np.asarray(predicted['v'])
    return dict(root_position_m=float(np.linalg.norm(np.asarray(actual['q'])[:3] - np.asarray(predicted['q'])[:3])),
        root_attitude_rad=float(np.linalg.norm(dq[3:6])),
        joint_position_max_rad=float(np.max(abs(dq[model.v_indices]))),
        root_linear_velocity_max_m_s=float(np.max(abs(dv[:3]))),
        root_angular_velocity_max_rad_s=float(np.max(abs(dv[3:6]))),
        joint_velocity_max_rad_s=float(np.max(abs(dv[model.v_indices]))),
        contacts_equal=bool(np.array_equal(predicted['contacts'], actual['contacts'])))


def summarize(rows, field):
    values = [r[field] for r in rows if field in r]
    if not values:
        return dict(samples=0)
    return dict(samples=len(values), contact_mismatches=sum(not r['contacts_equal'] for r in values),
        errors={key: dict(median=float(np.median([r[key] for r in values])),
                         maximum=float(max(r[key] for r in values)))
                for key in values[0] if key != 'contacts_equal'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['run', 'bundle', 'search', 'model-plan', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve existing evidence')
    read = lambda p: json.loads(Path(p).read_text())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    manifest = read(args.bundle/'manifest.json')
    search = read(args.search)
    job = read(args.run/'job.json')
    result = read(args.run/'sequence_native_result.json')
    if (job['status'] != 'SUCCEEDED' or job['exit_code'] != 0 or
            not result['query_match']['passed'] or result['case'] != search['case'] or
            result['case'] != manifest['case'] or manifest['proposal'] != search['best']['plan']):
        raise ValueError('Completed matching actual proposal required')
    for path, digest in manifest['source_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Changed proposal evidence ' + path)
    setup = read(args.model_plan); source = read(setup['source'])
    *_, audit, context = source_context(source['configuration']['source'])
    initialize(*context)
    controller = _WORKER['controller']; model = controller.qp.model
    if manifest['configuration'] != context[0]:
        raise ValueError('Predictive control interface differs')
    trace_path = args.run/'sequence_native_trace.npz'
    with np.load(trace_path, allow_pickle=False) as z:
        trace = {k: z[k].copy() for k in z.files}
    ticks = np.array([r['tick'] for r in result['records']], dtype=int)
    if not np.array_equal(ticks, np.arange(280, 280+len(ticks))) or len(ticks) < 2:
        raise ValueError('Continuous native feedback required')
    nominal = search['best']['result']['trajectory']
    if not np.array_equal([r['control_tick'] for r in nominal], np.arange(280, 280+len(nominal))):
        raise ValueError('Nominal end-state clock differs')
    limits = np.asarray(manifest['configuration']['interface']['limits'])
    commands = limits[:, 0] + .5*(trace['final_action']+1)*np.diff(limits).ravel()
    if (np.max(abs(commands-trace['requested_target_rad'])) > 1e-6 or
            not np.allclose(trace['previous_target_rad'][1:], commands[:-1], atol=1e-12, rtol=0)):
        raise ValueError('Actual command/previous-command chain differs')
    rows = []
    for i, row in enumerate(nominal):
        if i+1 >= len(ticks):
            break
        actual = {k: trace[k][i+1] for k in ['q', 'v', 'contacts']}
        current = {k: trace[k][i] for k in ['q', 'v', 'contacts']}
        current['target'] = trace['previous_target_rad'][i]
        state = state_from_row(current)
        measured_q = trace['q'][i][model.q_indices]
        record = dict(command_tick=int(ticks[i]), state_comparison_tick=int(ticks[i+1]),
            nominal_error=state_error(model, row, actual),
            nominal_vs_actual_command_max_rad=float(np.max(abs(np.asarray(row['target'])-commands[i]))),
            requested_minus_measured_joint_max_rad=float(np.max(abs(commands[i]-measured_q))),
            actual_ground_contacts=trace['contacts'][i].tolist(),
            actual_ground_normal_force_n=trace['foot_force_world_n'][i].tolist())
        geometry = HipMountPathConstraintV1(_WORKER['geometry'], audit, margin_rad=.02)
        try:
            prediction = predict_speed_limited_spheres(model, controller.qp.interface, state,
                commands[i], _WORKER['spheres'], substeps=160, geometry=geometry)
            predicted = prediction['state']
            record['one_step_error'] = state_error(model,
                dict(q=predicted.q, v=predicted.v, contacts=predicted.contacts), actual)
            record['one_step_diagnostics'] = prediction['diagnostics']
            record['one_step_first_ground_violation'] = prediction['first_ground_violation']
        except (ValueError, np.linalg.LinAlgError) as error:
            record['one_step_rejected'] = repr(error)
            record['one_step_collision'] = geometry.last_violation
        rows.append(record)
    pins = [Path(__file__), args.search, args.model_plan, args.bundle/'manifest.json',
        args.run/'job.json', args.run/'sequence_native_result.json', trace_path]
    for name, module in list(sys.modules.items()):
        if name.startswith('scripts.') and getattr(module, '__file__', None):
            pins.append(Path(module.__file__))
    final = dict(scope=__doc__, case=result['case'],
        nominal_state_semantics='post-command at t+1', native_state_semantics='pre-command at t',
        command_mapping_error_rad=float(np.max(abs(commands-trace['requested_target_rad']))),
        native_steps=len(ticks), nominal_steps=len(nominal), compared_steps=len(rows),
        nominal=summarize(rows, 'nominal_error'), one_step=summarize(rows, 'one_step_error'),
        one_step_rejections=sum('one_step_rejected' in r for r in rows), records=rows,
        original_model_failure=search['best']['result']['failure'],
        model_is_physx_equivalent=False, physical_impossibility_claimed=False,
        valid_expert_labels=False, source_sha256={str(p.resolve()): sha(p) for p in pins})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix('.tmp')
    tmp.write_text(json.dumps(plain(final), indent=2, allow_nan=False)+'\n'); tmp.replace(args.output)
    print(json.dumps({k: v for k, v in final.items() if k not in ['records', 'source_sha256']}))


if __name__ == '__main__':
    main()
