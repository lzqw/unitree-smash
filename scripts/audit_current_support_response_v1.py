"""Compare one-step model predictions with the actual post-command state.

Sample stance capture, the middle of the subsequent interval, and the last
step before nonfoot contact (or the clean episode end). Use actual final
commands, identical 160 substeps, and current whole-body contact dynamics.
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
from scripts.pure_control_reference_clock_v12 import PureControlReferenceClockV12
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.audit_actual_handoff_model_v1 import state_from_record
from scripts.audit_settled_stance_model_v2 import ForceRecorder
from scripts.predict_self_contact_spheres_v2 import predict_self_contact_spheres
from scripts.audit_joint_sequence_proposal_v1 import plain


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['run', 'configuration', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--cases', type=int, nargs='+', required=True)
    a = p.parse_args()
    read = lambda path: json.loads(path.read_text())
    if a.output.exists() or read(a.run/'job.json')['status'] != 'SUCCEEDED':
        raise ValueError('Fresh output and completed native evidence required')
    cfg = read(a.configuration)
    c = PureControlReferenceClockV12(PositionPDInterface(**cfg['interface']),
        read(a.run/'native_command_inputs.json')['default_joint_position_rad'], cfg['model_reference_library'])
    with np.load(a.run/'pure_control_commands.npz') as z:
        arrays = {k:z[k].copy() for k in z.files}
    with np.load(a.run/'pure_control_ground_forces.npz') as z:
        peaks = z['interval_peak_normal_n'].copy()
    nonfoot = [i for i,n in enumerate(read(a.run/'pure_control_ground_forces.json')['body_names'])
               if not n.endswith('_foot')]
    diagnostics = [json.loads(s) for s in (a.run/'pure_control_diagnostics.jsonl').read_text().splitlines()]
    rows = []
    started = time.monotonic()
    for case in a.cases:
        ids = np.flatnonzero(arrays['lane'] == case)
        lookup = {int(arrays['tick'][i]):int(i) for i in ids}
        captures = [t for t,i in lookup.items() if diagnostics[i]['diagnostic'].get('terminal_candidate_now')]
        if len(captures) != 1:
            raise ValueError('Exactly one causal stance capture required')
        start = min(lookup)
        touched = np.flatnonzero(np.any(peaks[start:,case][:,nonfoot] > 0., axis=1))
        end = start + int(touched[0]) if len(touched) else 799
        begin = captures[0]
        if end <= begin:
            raise ValueError('Capture precedes nonfoot contact in this audit')
        samples = [(begin, 'capture'), ((begin+end)//2, 'middle'), (end-1, 'pre_contact_or_last_recorded_interval')]
        for tick, label in samples:
            i, j = lookup[tick], lookup[tick+1]
            if (diagnostics[i]['tick'],diagnostics[i]['lane']) != (tick,case):
                raise ValueError('Misaligned diagnostics')
            record = {k:arrays[k][i] for k in ['q','v','previous','ground_force']}
            record['phase'] = diagnostics[i]['phase']
            state = state_from_record(c.model, record)
            row = dict(case=case, tick=tick, label=label, qp_accepted=diagnostics[i]['diagnostic']['solver_accepted'],
                source=diagnostics[i]['diagnostic']['source'], predicted=False,
                actual_nonfoot_force_in_interval=bool(np.any(peaks[tick,case,nonfoot] > 0.)))
            recorder = ForceRecorder(c.model)
            try:
                prediction = predict_self_contact_spheres(c.model,c.interface,state,arrays['final'][i],
                    c.geometry.spheres, substeps=160, geometry=c.path_geometry, collision_response=recorder)
                y = prediction['state']
                q,v = arrays['q'][j], arrays['v'][j]
                rp,ra = pin.Quaternion(y.q[3:7]).matrix(),pin.Quaternion(q[3:7]).matrix()
                difference = pin.difference(c.model.model, q, y.q)
                foot_mask = np.asarray(record['phase']['ground_contacts'], bool)
                model_mask = np.where(foot_mask, recorder.last_force[:,2] > .5, recorder.last_force[:,2] > 1.)
                row.update(predicted=True, position_error_m=float(np.linalg.norm(y.q[:3]-q[:3])),
                    attitude_error_rad=float(np.linalg.norm(difference[3:6])),
                    max_joint_position_error_rad=float(np.max(abs(y.q[c.model.q_indices]-q[c.model.q_indices]))),
                    linear_velocity_error_m_s=float(np.linalg.norm(rp@y.v[:3]-ra@v[:3])),
                    angular_velocity_error_rad_s=float(np.linalg.norm(rp@y.v[3:6]-ra@v[3:6])),
                    max_joint_velocity_error_rad_s=float(np.max(abs(y.v[c.model.v_indices]-v[c.model.v_indices]))),
                    predicted_observed_contacts=model_mask,
                    actual_next_contacts=diagnostics[j]['phase']['ground_contacts'],
                    model_nonfoot_violation_m=prediction['diagnostics']['nonfoot_ground_violation_m'])
            except (ValueError, np.linalg.LinAlgError) as error:
                row['rejection'] = repr(error)
            rows.append(row)
    accepted = [r for r in rows if r['predicted']]
    keys = ['position_error_m','attitude_error_rad','max_joint_position_error_rad',
        'linear_velocity_error_m_s','angular_velocity_error_rad_s','max_joint_velocity_error_rad_s']
    summary = {key:dict(zip(['median','maximum'], np.quantile([r[key] for r in accepted],[.5,1.])))
               for key in keys} if accepted else {}
    files = [Path(__file__),a.configuration,Path(cfg['model_reference_library'])] + [a.run/n for n in
        ['job.json','native_command_inputs.json','pure_control_commands.npz','pure_control_diagnostics.jsonl',
         'pure_control_ground_forces.npz','pure_control_ground_forces.json']]
    files += [Path(m.__file__) for n,m in list(sys.modules.items())
              if n.startswith('scripts.') and getattr(m,'__file__',None)]
    result = dict(rows=rows,summary=summary,wall_seconds=time.monotonic()-started,
        source_sha256={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        native_rerun=False,actual_final_commands=True,substeps=160,acceptance_changed=False,
        forces_are_model_estimates=True,long_horizon_accuracy_not_inferred=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    temp = a.output.with_suffix('.tmp')
    temp.write_text(json.dumps(plain(result),indent=2,allow_nan=False)+'\n')
    temp.replace(a.output)
    print(json.dumps(plain(dict(samples=len(rows),predicted=len(accepted),summary=summary))))


if __name__ == '__main__':
    main()
