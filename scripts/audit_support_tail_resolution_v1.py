"""Compare identical support references at two model integration resolutions.

This diagnoses discretization sensitivity, not agreement with native physics.
The fine saved trajectory and both source provenances are checked before use.
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
from scripts.search_landing_support_tail_v3 import initialize_worker, evaluate_job, _WORKER
from scripts.audit_joint_sequence_proposal_v1 import plain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve prior audit')
    read = lambda p: json.loads(Path(p).read_text())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    source = read(args.source)
    hashes = dict(source['source_sha256'])
    hashes[str(args.source.resolve())] = sha(args.source)
    hashes[str(Path(__file__).resolve())] = sha(__file__)
    for filename, expected in hashes.items():
        if sha(filename) != expected:
            raise ValueError('Changed evidence ' + filename)
    parent = read(source['configuration']['source_search'])
    cfg = parent['configuration']
    query = read(cfg['query'])
    manifest = read(Path(cfg['bundle'])/'manifest.json')
    initialize_worker(manifest['configuration'], query,
                      source['fixed_early_knots_rad'], cfg['sphere_audit'])
    selected = source['fine_candidates'][0]
    knots = np.asarray(selected['knots_rad'])
    prior = next(x['result'] for x in source['ledger']
                 if np.array_equal(x['knots_rad'], knots))
    _, _, coarse = evaluate_job((0, knots, 16, time.monotonic()+180., True))
    for key in ['accepted', 'completed_model_steps', 'failure', 'terminal', 'rank']:
        if coarse[key] != prior[key]:
            raise ValueError('Coarse audit did not reproduce ' + key)
    fine = selected['fine']
    model = _WORKER['controller'].qp.model
    rows = []
    for index, (a, b) in enumerate(zip(coarse['trajectory'], fine['trajectory'])):
        aq, bq = np.asarray(a['q']), np.asarray(b['q'])
        av, bv = np.asarray(a['v']), np.asarray(b['v'])
        tangent = pin.difference(model.model, aq, bq)
        rows.append(dict(completed_steps=index+1,
            root_position_m=float(np.linalg.norm(aq[:3]-bq[:3])),
            root_rotation_rad=float(np.linalg.norm(tangent[3:6])),
            maximum_joint_position_rad=float(np.max(abs(tangent[model.v_indices]))),
            maximum_joint_velocity_rad_s=float(np.max(abs(av[model.v_indices]-bv[model.v_indices]))),
            maximum_requested_target_rad=float(np.max(abs(np.asarray(a['target'])-b['target']))),
            contact_flags_equal=bool(np.array_equal(a['contacts'], b['contacts']))))
    compact = lambda result: {k: v for k, v in result.items() if k != 'trajectory'}
    summary = dict(scope=__doc__, case=source['case'], tick=source['tick'],
        coarse=compact(coarse), fine=compact(fine), coarse_reproduced=True,
        matched_control_intervals=len(rows),
        first_contact_mode_difference=next((r['completed_steps'] for r in rows if not r['contact_flags_equal']), None),
        first_root_position_difference_over_1mm=next((r['completed_steps'] for r in rows if r['root_position_m']>.001), None),
        samples=[r for r in rows if r['completed_steps'] in [1, 20, 40, 80, 100, 150, 200, 240, len(rows)]],
        source_sha256=hashes, native_execution=False, valid_expert_labels=False,
        limitation='The same feedback law may issue different targets after state trajectories diverge; these are not identical open-loop command sequences.')
    for filename, expected in hashes.items():
        if sha(filename) != expected:
            raise ValueError('Changed source during audit ' + filename)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plain(summary), indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in summary.items() if k in ['matched_control_intervals', 'first_contact_mode_difference', 'first_root_position_difference_over_1mm']}))


if __name__ == '__main__':
    main()
