"""Compare the coupled predictor with actual pure-control state transitions.

Every prediction starts at a recorded native state and uses the target read
back at the next control boundary. This measures one-step error, not drift
from an offline trajectory. No native state is changed or teleported.
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
import pinocchio as pin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.pure_control_reference_clock_v12 import PureControlReferenceClockV12
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.loaded_pinocchio_model_v1 import PublicState
from scripts.predict_self_contact_spheres_v2 import predict_self_contact_spheres
from scripts.audit_settled_stance_model_v2 import ForceRecorder
from scripts.audit_joint_sequence_proposal_v1 import plain

WORK = {}


def initialize(configuration, nominal):
    WORK['controller'] = PureControlReferenceClockV12(
        PositionPDInterface(**configuration['interface']), nominal,
        configuration['model_reference_library'])


def predict(record, substeps, deadline):
    if time.monotonic() >= deadline:
        return dict(case=record['case'], tick=record['tick'], substeps=substeps,
                    completed=False, reason='compute_budget')
    c = WORK['controller']
    m = c.model
    q, v = np.asarray(record['q']), np.asarray(record['v'])
    r = pin.Quaternion(q[3:7]).matrix()
    state = PublicState(q.copy(), v.copy(), r, q[:3].copy(), q[m.q_indices].copy(),
        v[m.v_indices].copy(), np.asarray(record['contacts'], bool), 0.)
    state.previous_command_target = np.asarray(record['previous']).copy()
    state.measured_force_world = np.asarray(record['ground_force']).copy()
    recorder = ForceRecorder(m)
    base = {key:record[key] for key in ['case', 'tick', 'phase', 'reference_active',
        'reference_end_tick', 'contacts', 'next_contacts', 'command_source', 'command_roundtrip_error_rad']}
    base.update(substeps=substeps)
    try:
        result = predict_self_contact_spheres(m, c.interface, state,
            np.asarray(record['actually_applied']), c.geometry.spheres,
            substeps=substeps, geometry=c.path_geometry, collision_response=recorder)
    except (ValueError, np.linalg.LinAlgError) as error:
        return dict(base, completed=False, reason='model_rejected', error=repr(error))
    predicted = result['state']
    aq, av = np.asarray(record['next_q']), np.asarray(record['next_v'])
    ar = pin.Quaternion(aq[3:7]).matrix()
    tangent = pin.difference(m.model, aq, predicted.q)
    jv = predicted.v[m.v_indices]-av[m.v_indices]
    force = recorder.last_force
    contact = np.where(state.contacts, abs(force[:,2])>.5, abs(force[:,2])>1.)
    return dict(base, completed=True,
        root_position_error_m=float(np.linalg.norm(predicted.q[:3]-aq[:3])),
        root_rotation_error_rad=float(np.linalg.norm(tangent[3:6])),
        root_linear_velocity_error_m_s=float(np.linalg.norm(predicted.root_rotation_world@predicted.v[:3]-ar@av[:3])),
        root_angular_velocity_error_rad_s=float(np.linalg.norm(predicted.root_rotation_world@predicted.v[3:6]-ar@av[3:6])),
        max_joint_position_error_rad=float(np.max(abs(tangent[m.v_indices]))),
        max_joint_velocity_error_rad_s=float(np.max(abs(jv))),
        joint_velocity_error_rad_s=jv, worst_joint=m.joint_order[int(np.argmax(abs(jv)))],
        predicted_last_normal_force_n=force[:,2], native_next_normal_force_n=np.asarray(record['next_ground_force'])[:,2],
        predicted_observed_contacts=contact,
        contact_mismatch_feet=int(np.sum(contact!=np.asarray(record['next_contacts']))),
        diagnostics=result['diagnostics'], first_ground_violation=result['first_ground_violation'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['run', 'configuration', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--cases', nargs='+', type=int, required=True)
    args = p.parse_args()
    read = lambda f: json.loads(Path(f).read_text())
    if args.output.exists() or read(args.run/'job.json')['status'] != 'SUCCEEDED':
        raise ValueError('Fresh audit of a completed native run required')
    cfg = read(args.configuration)
    nominal = read(args.run/'native_command_inputs.json')['default_joint_position_rad']
    initialize(cfg, nominal)
    with np.load(args.run/'pure_control_commands.npz') as z:
        a = {k:z[k].copy() for k in z.files}
    rows = [json.loads(line) for line in (args.run/'pure_control_diagnostics.jsonl').read_text().splitlines()]
    meta = read(args.run/'pure_control_ground_forces.json')
    if not meta['complete'] or not meta['interval_aggregates_complete']:
        raise ValueError('Complete native contact recording required')
    with np.load(args.run/'pure_control_ground_forces.npz') as z:
        peaks = z['interval_peak_normal_n'].copy()
    nonfoot = [i for i,n in enumerate(meta['body_names']) if not n.endswith('_foot')]
    if len(rows)!=len(a['lane']) or any((r['tick'],r['lane'])!=(a['tick'][i],a['lane'][i]) for i,r in enumerate(rows)):
        raise ValueError('Aligned command and diagnostic observations required')
    records, selection = [], []
    for lane in args.cases:
        ids = np.flatnonzero(a['lane']==lane)
        if len(ids)<2:
            raise ValueError('Missing owned case '+str(lane))
        ticks = a['tick'][ids]
        if np.any(np.diff(ticks)!=1) or len(set(a['episode'][ids].tolist()))!=1:
            raise ValueError('Contiguous single native episode required')
        start, end = int(ticks[0]), int(ticks[-1])+1
        touched = np.any(peaks[start:end,lane][:,nonfoot]>0., axis=1)
        first = start+int(np.flatnonzero(touched)[0]) if touched.any() else end
        active = [i for i in ids if rows[int(i)]['diagnostic'].get('reference_active')]
        ref_end = int(a['tick'][active[-1]])+1 if active else None
        selected = set(range(0, len(ids)-1, 10))
        for j in range(1,len(ids)):
            before, after = rows[int(ids[j-1])]['phase'], rows[int(ids[j])]['phase']
            if (before['ground_contacts']!=after['ground_contacts']
                    or int(ticks[j])==ref_end
                    or (before['control_sample_stable_duration_s']>0 and after['control_sample_stable_duration_s']==0)):
                selected.update(range(max(0,j-2),min(len(ids)-1,j+3)))
        selected = sorted(j for j in selected if int(ticks[j])<first)
        selection.append(dict(case=lane, first_nonfoot_positive_force_tick=first if touched.any() else None,
            reference_end_tick=ref_end, evaluated_intervals=len(selected)))
        for j in selected:
            i, ni = int(ids[j]), int(ids[j+1])
            error = float(np.max(abs(a['previous'][ni]-a['final'][i])))
            if error>2e-6:
                raise ValueError('Actual next-boundary target differs from dispatched target')
            row = rows[i]
            records.append(dict(case=lane,tick=int(a['tick'][i]), q=a['q'][i],v=a['v'][i],
                previous=a['previous'][i],actually_applied=a['previous'][ni],
                next_q=a['q'][ni],next_v=a['v'][ni],ground_force=a['ground_force'][i],
                next_ground_force=a['ground_force'][ni],contacts=row['phase']['ground_contacts'],
                next_contacts=rows[ni]['phase']['ground_contacts'],phase=row['phase']['phase'],
                command_source=row['diagnostic']['source'],reference_active=row['diagnostic'].get('reference_active',False),
                reference_end_tick=ref_end, command_roundtrip_error_rad=error))
    files = [Path(__file__),args.configuration,Path(cfg['model_reference_library'])]
    files += [args.run/n for n in ['job.json','native_command_inputs.json','pure_control_commands.npz',
        'pure_control_diagnostics.jsonl','pure_control_ground_forces.json','pure_control_ground_forces.npz']]
    files += [Path(m.__file__) for n,m in list(sys.modules.items()) if n.startswith('scripts.') and getattr(m,'__file__',None)]
    pins = {str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    started = time.monotonic()
    result = []
    with ProcessPoolExecutor(max_workers=6,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(cfg,nominal)) as pool:
        jobs = [pool.submit(predict,r,s,started+900.) for s in [16,160] for r in records]
        for future in as_completed(jobs):
            result.append(future.result())
    result.sort(key=lambda r:(r['case'],r['tick'],r['substeps']))
    keys = ['root_position_error_m','root_rotation_error_rad','root_linear_velocity_error_m_s',
        'root_angular_velocity_error_rad_s','max_joint_position_error_rad','max_joint_velocity_error_rad_s']
    groups = []
    for case in args.cases:
        for steps in [16,160]:
            for part in ['reference','after_reference']:
                group = [r for r in result if r['case']==case and r['substeps']==steps
                    and r['reference_active']==(part=='reference')]
                ok = [r for r in group if r['completed']]
                groups.append(dict(case=case,substeps=steps,part=part,intervals=len(group),completed=len(ok),
                    errors={k:dict(median=float(np.median([r[k] for r in ok])),p95=float(np.percentile([r[k] for r in ok],95)),
                        maximum=float(max(r[k] for r in ok))) for k in keys} if ok else {},
                    force_label_mismatch_feet=sum(r['contact_mismatch_feet'] for r in ok)))
    for f,h in pins.items():
        if hashlib.sha256(Path(f).read_bytes()).hexdigest()!=h:
            raise ValueError('Changed input '+f)
    output = dict(selection=selection,groups=groups,records=result,source_sha256=pins,
        wall_seconds=time.monotonic()-started,actual_readback_targets_used=True,
        accumulated_reference_error=False, native_rerun=False, valid_expert_labels=False,
        limitations=['Sampled intervals before the first positive nonfoot ground-force interval, not independent trials.',
            'Net and other contacts may still affect native motion; errors do not uniquely identify PD parameters.',
            'Native sensor reports normal force; model final-microstep estimates are diagnostic.',
            'No threshold or physical parameter is changed by this read-only audit.'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(plain(output),indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(intervals=len(records),predictions=len(result),completed=sum(r['completed'] for r in result),
        wall_seconds=output['wall_seconds'],native_success=False)),flush=True)


if __name__=='__main__':
    main()
