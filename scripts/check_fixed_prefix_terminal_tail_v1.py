"""Independently replay exact coarse-prefix commands with a fine feedback tail."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.parallel_landing_solver_v1 import build_controller
from scripts.native_terminal_attitude_v1 import load_verified_policy
from scripts.audit_native_frictional_recovery_v1 import state_at
from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance
from scripts.feedback_tail_rollout_v1 import evaluate_feedback_tail
from scripts.audit_joint_sequence_proposal_v1 import plain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['candidate', 'run', 'bundle', 'validation', 'warm-plan', 'sphere-audit', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    a = parser.parse_args()
    if a.output.exists():
        raise ValueError('Preserve existing check')
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    read = lambda p: json.loads(p.read_text())
    candidate = read(a.candidate)
    for path, digest in candidate['source_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Changed generation source '+path)
    manifest = read(a.bundle/'manifest.json')
    for path in [a.run/'sequence_native_trace.npz', a.bundle/'manifest.json', a.validation, a.warm_plan, a.sphere_audit]:
        if candidate['source_sha256'].get(str(path.resolve())) != sha(path):
            raise ValueError('Generation used different input '+str(path))
    with np.load(a.bundle/manifest['bundle_name'], allow_pickle=False) as z:
        query = z['case_003_query_public'].copy()
    with np.load(a.run/'sequence_native_trace.npz', allow_pickle=False) as z:
        trace = {k: z[k].copy() for k in z.files}
    c = build_controller(manifest['configuration'])
    policy, _ = load_verified_policy(c.qp.model, c, query, a.warm_plan, a.validation)
    geometry = AuthoredGroundClearance(ROOT/'configs/go2_piper_authored_ground_geometry_v1.json', c.qp.model, margin_m=0.)
    spheres = read(a.sphere_audit)['spheres']; protocol = read(ROOT/'configs/evaluation_protocol_v2.json')['recovery']
    rows = []
    for row in candidate['rows']:
        if not row['result']['accepted'] or row['result']['integration_substeps'] != 16:
            raise ValueError('Accepted coarse candidate required')
        state = state_at(c.qp.model, trace, row['tick']-460)
        for key, actual in [('initial_q', state.q), ('initial_v', state.v),
                            ('initial_contacts', state.contacts), ('previous_command', state.previous_command_target)]:
            if not np.array_equal(row[key], actual):
                raise ValueError('Different initial state or history')
        prefix = np.asarray(row['result']['commands'])[:80]
        result = evaluate_feedback_tail(c, state, policy, spheres, geometry, protocol,
            control_tick=row['tick'], prefix_commands=prefix, substeps=160, time_limit_s=300.)
        command_error = float(np.max(abs(result['commands'][:80]-prefix))) if result['accepted'] else None
        if result['accepted'] and command_error != 0:
            raise ValueError('Fine verification changed the proposed prefix')
        rows.append(dict(tick=row['tick'], exact_prefix_error_rad=command_error, result=result))
        print(json.dumps(dict(tick=row['tick'], exact_prefix_error_rad=command_error,
            **{k: result[k] for k in ['accepted', 'failure', 'terminal', 'terminal_violations',
                 'trailing_model_kinematic_window_s', 'wall_seconds']})), flush=True)
    hashes = dict(candidate['source_sha256'])
    hashes.update({str(path.resolve()): sha(path) for path in [Path(__file__), a.candidate]})
    for path, digest in hashes.items():
        if sha(path) != digest:
            raise ValueError('Changed verification source '+path)
    a.output.write_text(json.dumps(plain(dict(scope=__doc__, rows=rows, source_sha256=hashes,
        native_execution=False, valid_expert_labels=False)), indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
