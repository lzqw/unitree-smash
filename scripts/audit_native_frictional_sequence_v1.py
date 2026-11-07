"""Matched one-step predictions on a completed native recovery command replay.

Each predictor starts from the same measured native q/v and receives the same
actual command. Nominal open-loop errors are not confused with these errors.
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
import scripts.search_frictional_recovery_cem_v1 as core
from scripts.extend_frictional_recovery_v1 import set_state
from scripts.predict_frictional_spheres_v1 import predict_frictional_spheres
from scripts.predict_sphere_normal_impact_v1 import predict_sphere_normal_impact
from scripts.summarize_precontact_prediction_v1 import stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['run', 'dispatch-audit', 'bundle', 'sphere-audit', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Preserve evidence')
    sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
    audit = json.loads(a.dispatch_audit.read_text())
    if not audit['passed_dispatch_integrity'] or not audit['native_execution']:
        raise ValueError('Require native dispatch verification')
    hashes = dict(audit['source_sha256'])
    for f, digest in hashes.items():
        if sha(f) != digest:
            raise ValueError('Changed native evidence '+f)
    trace_path = a.run/'sequence_native_trace.npz'
    if str(trace_path.resolve()) not in hashes:
        raise ValueError('Different native run')
    sphere = json.loads(a.sphere_audit.read_text())
    for f, digest in sphere['source_sha256'].items():
        if sha(f) != digest:
            raise ValueError('Changed geometry evidence '+f)
    core.init_worker(str(a.bundle), str(a.sphere_audit), audit['case'])
    c, template, previous, _, spheres, geometry = core.CONTEXT
    with np.load(trace_path) as z:
        trace = {k:z[k].copy() for k in z.files}
    files = [Path(__file__), a.dispatch_audit, a.sphere_audit, a.bundle/'manifest.json', c.qp.model.path]
    files += [Path(m.__file__) for n,m in list(sys.modules.items()) if n.startswith('scripts.') and getattr(m,'__file__',None)]
    for f in files:
        path = str(f.resolve()); digest = sha(f)
        if path in hashes and hashes[path] != digest:
            raise ValueError('Model source differs from declared native snapshot '+path)
        hashes[path] = digest
    rows = []
    first_event = audit['selected_first_nonfoot_contact_s']
    for i in range(len(trace['q'])-1):
        state = set_state(template, c.qp.model, trace['q'][i], trace['v'][i], trace['contacts'][i], trace['previous_target_rad'][i])
        target = trace['final_target_rad'][i]
        next_q = trace['q'][i+1]; next_v = trace['v'][i+1]
        gaps, _ = geometry.evaluate(state.q); nearest = int(np.argmin(gaps))
        tick = audit['start_tick']+i
        row = dict(tick=tick, before_recorded_nonfoot=first_event is None or (tick+1)*.01 < first_event,
            measured_nearest_nonfoot_body=geometry.shapes[nearest]['body'],
            measured_nearest_authored_gap_m=float(gaps[nearest]), models={})
        for name in ['refreshed_fixed', 'frictional_spheres']:
            try:
                result = (predict_frictional_spheres(c.qp.model,c.qp.interface,state,target,spheres) if name == 'frictional_spheres'
                    else predict_sphere_normal_impact(c.qp.model,c.qp.interface,state,target,spheres,detect_impacts=False))
                predicted = result['state']
                row['models'][name] = dict(ok=True,
                    leg_error_rad=float(np.max(abs(predicted.q[c.qp.model.q_indices[:12]]-next_q[c.qp.model.q_indices[:12]]))),
                    arm_error_rad=float(np.max(abs(predicted.q[c.qp.model.q_indices[12:]]-next_q[c.qp.model.q_indices[12:]]))),
                    root_error_m=float(np.linalg.norm(predicted.q[:3]-next_q[:3])),
                    velocity_error_max=float(np.max(abs(predicted.v-next_v))),
                    attitude_error_rad=float(np.linalg.norm(pin.log3(predicted.root_rotation_world.T@pin.Quaternion(next_q[3:7]).matrix()))))
            except (ValueError,np.linalg.LinAlgError) as error:
                row['models'][name] = dict(ok=False,error=repr(error))
        rows.append(row)
    groups = {}
    for label, selected in [('all', rows), ('before_recorded_nonfoot', [r for r in rows if r['before_recorded_nonfoot']])]:
        models = {}
        for name in ['refreshed_fixed','frictional_spheres']:
            good = [r['models'][name] for r in selected if r['models'][name]['ok']]
            models[name] = dict(attempted=len(selected), failed=len(selected)-len(good),
                **{k:stats([r[k] for r in good]) for k in ['leg_error_rad','arm_error_rad','root_error_m','velocity_error_max','attitude_error_rad']})
        paired = [r for r in selected if all(v['ok'] for v in r['models'].values())]
        groups[label] = dict(models=models, paired_count=len(paired),
            improved=sum(r['models']['frictional_spheres']['leg_error_rad'] < r['models']['refreshed_fixed']['leg_error_rad']-1e-6 for r in paired),
            worsened=sum(r['models']['frictional_spheres']['leg_error_rad'] > r['models']['refreshed_fixed']['leg_error_rad']+1e-6 for r in paired))
    for f,digest in hashes.items():
        if sha(f) != digest:
            raise ValueError('Input changed during audit '+f)
    result = dict(scope=__doc__, native_new_execution=False, valid_teacher=False, case=audit['case'],
        groups=groups, rows=rows, source_sha256=hashes,
        limitations=['Authored gap is not a measured contact force',
            'Recorded first nonfoot event excludes bodies absent from original sensors',
            'Both models omit nonfoot impulses; later errors do not establish safe recovery'])
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(groups))


if __name__ == '__main__':
    main()
