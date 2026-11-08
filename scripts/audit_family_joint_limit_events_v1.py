"""Inspect predeclared largest one-step errors; diagnosis, not fitted evaluation."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.check_constrained_recovery_suffix_v1 import source_context
from scripts.search_preserved_early_support_v1 import initialize
from scripts.search_landing_support_tail_v3 import _WORKER
from scripts.native_recovery_query_bank_v1 import state_from_sample
from scripts.predict_speed_limited_spheres_v1 import predict_speed_limited_spheres
from scripts.audit_sphere_touchdown_predictions_v1 import sphere_gaps
from scripts.audit_joint_sequence_proposal_v1 import plain


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Fresh output required')
    read = lambda p: json.loads(Path(p).read_text())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    audit = read(args.audit)
    for path, digest in audit['source_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Changed audited source: '+path)
    plan = read(args.plan)
    source = read(plan['source'])
    _, _, _, _, context = source_context(source['configuration']['source'])
    initialize(*context)
    w = _WORKER
    model, interface = w['controller'].qp.model, w['controller'].qp.interface
    bank = Path(plan['bank_run'])
    meta = read(bank/'recovery_query_bank.json')
    with np.load(bank/'recovery_query_bank.npz', allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    def state(case, tick):
        ids = np.flatnonzero(arrays['control_ticks'] == tick)
        if len(ids) != 1:
            raise ValueError('Unique tick required')
        sample = {k: v[int(ids[0]), case] for k, v in arrays.items() if k != 'control_ticks'}
        if sample['episode_index'] != 0 or not sample['unfinished_first_episode']:
            raise ValueError('Cannot cross a reset')
        return state_from_sample(model, sample, meta['origins_world'][case]), sample
    rows = []
    for case in (85, 87):
        eligible = [r for r in audit['rows'] if r['case'] == case and r['completed']
                    and min(r['actual_initial_nonfoot_gap_m'], r['actual_final_nonfoot_gap_m']) > .001]
        for selected in sorted(eligible, key=lambda r: r['maximum_joint_velocity_error_rad_s'], reverse=True)[:3]:
            tick = selected['tick']
            (before, before_sample), (after, after_sample) = state(case, tick), state(case, tick+1)
            prediction = predict_speed_limited_spheres(model, interface, before,
                after.previous_command_target, w['spheres'], substeps=160, geometry=w['geometry'])
            predicted = prediction['state']
            joints = []
            for i, name in enumerate(model.joint_order):
                qi, vi = model.q_indices[i], model.v_indices[i]
                joints.append(dict(name=name, limits=interface.limits[i],
                    actual_q_before=before.q[qi], actual_q_after=after.q[qi], predicted_q=predicted.q[qi],
                    actual_v_before=before.v[vi], actual_v_after=after.v[vi], predicted_v=predicted.v[vi],
                    target=after.previous_command_target[i],
                    velocity_error_rad_s=abs(predicted.v[vi]-after.v[vi])))
            rows.append(dict(case=case, tick=tick, joints=sorted(joints,
                key=lambda r: r['velocity_error_rad_s'], reverse=True), diagnostics=prediction['diagnostics'],
                feet=model.feet, initial_foot_gap_m=sphere_gaps(model,before.q,w['spheres']),
                actual_final_foot_gap_m=sphere_gaps(model,after.q,w['spheres']),
                predicted_final_foot_gap_m=sphere_gaps(model,predicted.q,w['spheres']),
                actual_initial_foot_force_n=before_sample['foot_force_world'],
                actual_final_foot_force_n=after_sample['foot_force_world'],
                predicted_contact_events=prediction['events']))
    pins = dict(audit['source_sha256'])
    for p in [args.audit, Path(__file__).resolve()]:
        pins[str(p.resolve())] = sha(p)
    result = dict(rows=rows, source_sha256=pins, native_execution=False,
        note='Selected error outliers for diagnosis, not an independent performance panel. '
             'Proximity to a joint limit does not prove an actual joint-limit impulse.')
    args.output.write_text(json.dumps(plain(result), indent=2, allow_nan=False)+'\n')
    for row in rows:
        print(json.dumps(plain(dict(case=row['case'], tick=row['tick'], joints=row['joints'][:2]))))


if __name__ == '__main__':
    main()
