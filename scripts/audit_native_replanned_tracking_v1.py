"""Measure native execution of selected recovery references without replay.

Tracking error is NOT one-step model error: feedback may deliberately choose
a different command. Align pre-command native state at tick k with reference
state k-1; reference row k contains the command and its resulting state.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pinocchio as pin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def stats(values):
    x = np.asarray(values, float)
    return dict(count=len(x), median=float(np.median(x)), p95=float(np.quantile(x, .95)),
        maximum=float(np.max(x))) if len(x) else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['preparation', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    read = lambda path: json.loads(path.read_text())
    cfg = read(a.preparation / 'controller.json')
    run = Path(read(a.preparation / 'plan.json')['output'])
    audit = read(a.preparation / 'native_audit.json')
    if a.output.exists() or read(run / 'job.json')['status'] != 'SUCCEEDED' or not audit['passed_dispatch_integrity']:
        raise ValueError('Require fresh output and completed, command-verified native execution')
    libpath = Path(cfg['model_replanning_library'])
    refs = {r['source_key']: r for r in read(libpath)['references']}
    model = LoadedPinocchioModelV1()
    with np.load(run / 'pure_control_commands.npz') as z:
        data = {k: z[k].copy() for k in ['tick', 'lane', 'time_s', 'q', 'v', 'final']}
    rows = [json.loads(line) for line in (run / 'pure_control_diagnostics.jsonl').read_text().splitlines()]
    if len(rows) != len(data['tick']):
        raise ValueError('Command and diagnostic record lengths differ')
    grouped = {}
    for i, row in enumerate(rows):
        if (row['tick'], row['lane']) != (int(data['tick'][i]), int(data['lane'][i])):
            raise ValueError('Misaligned native records')
        d = row['diagnostic']
        if d.get('replanned'):
            key = d['replanning_source_key']
            if key not in refs:
                raise ValueError('Unknown executed recovery reference')
            grouped.setdefault((row['lane'], key), []).append(i)
    outcomes = {r['case']: r for r in audit['cases']}
    cases, traces = [], []
    for (lane, key), ids in sorted(grouped.items()):
        ref = refs[key]
        start = int(data['tick'][ids[0]])
        active, sources, candidates = [], Counter(), Counter()
        planned_rejections, no_candidates, contact_mismatch = [], [], []
        for i in ids:
            d = rows[i]['diagnostic']
            sources[d['source']] += 1
            candidates[d.get('selected_predictive_candidate', 'none')] += 1
            k = int(data['tick'][i]) - start
            if k >= len(ref['trajectory']):
                continue
            if d.get('reference_control_tick') != k:
                raise ValueError('Reference control clock does not match recorded selection')
            before = ref['query'] if k == 0 else ref['trajectory'][k - 1]
            q, v = data['q'][i], data['v'][i]
            rq, rv = np.asarray(before['q']), np.asarray(before['v'])
            rotation = pin.Quaternion(q[3:7]).matrix()
            rr = pin.Quaternion(rq[3:7]).matrix()
            error = dict(root_position_m=float(np.linalg.norm(q[:3] - rq[:3])),
                root_orientation_rad=float(np.linalg.norm(pin.log3(rr.T @ rotation))),
                joint_position_max_rad=float(np.max(abs(q[model.q_indices] - rq[model.q_indices]))),
                root_velocity_world_m_s=float(np.linalg.norm(rotation @ v[:3] - rr @ rv[:3])),
                joint_velocity_max_rad_s=float(np.max(abs(v[model.v_indices] - rv[model.v_indices]))),
                command_from_plan_max_rad=float(np.max(abs(data['final'][i] - ref['trajectory'][k]['target']))))
            row = dict(case=lane, tick=int(data['tick'][i]), reference_step=k,
                source=d['source'], selected_candidate=d.get('selected_predictive_candidate'), **error)
            active.append(row)
            traces.append(row)
            if list(rows[i]['phase']['ground_contacts']) != list(before['contacts']):
                contact_mismatch.append(row['tick'])
            for c in d.get('predictive_candidates', []):
                if c['name'] == 'planned' and not c.get('accepted'):
                    planned_rejections.append(dict(tick=row['tick'], reason=c))
            if d.get('no_feasible_predictive_candidate'):
                no_candidates.append(row['tick'])
        physical = outcomes[lane]
        cases.append(dict(case=lane, selected_source_key=key, selection_tick=start,
            reference_steps=len(ref['trajectory']), observed_reference_steps=len(active),
            executed_replanned_steps=len(ids), sources=dict(sources), selected_candidates=dict(candidates),
            tracking={k: stats([r[k] for r in active]) for k in active[0] if k.endswith(('_m', '_rad', '_m_s', '_rad_s'))},
            command_deviation_over_2e_minus6_rad_ticks=[r['tick'] for r in active if r['command_from_plan_max_rad'] > 2e-6],
            model_reference_contact_disagreement_ticks=contact_mismatch,
            planned_rejections=planned_rejections, no_feasible_candidate_ticks=no_candidates,
            no_recorded_nonfoot_ground_force=physical['no_recorded_nonfoot_ground_force'],
            final_sampled_stable_s=physical['final_control_sample_stable_s'],
            head_before_takeover=physical['head_before_takeover']['any_nonzero']))
    files = [Path(__file__), a.preparation / 'plan.json', a.preparation / 'controller.json',
        a.preparation / 'native_audit.json', run / 'job.json', libpath,
        run / 'pure_control_commands.npz', run / 'pure_control_diagnostics.jsonl']
    result = dict(cases=cases, source_sha256={str(f.resolve()): digest(f) for f in files},
        tracking_error_is_not_one_step_prediction_error=True,
        pre_command_state_aligned_to_previous_reference_row=True,
        new_native_execution=False, formal_acceptance=False, training=False)
    a.output.mkdir(parents=True)
    (a.output / 'tracking.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    (a.output / 'trace.json').write_text(json.dumps(traces, separators=(',', ':'), allow_nan=False) + '\n')
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
