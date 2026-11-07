"""Run the complete extended-tail candidate selector on a real failed query."""
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
from scripts.native_frictional_recovery_v1 import shifted_frictional_seed
from scripts.frictional_feedback_tail_planner_v1 import FrictionalFeedbackTailPlannerV1
from scripts.audit_joint_sequence_proposal_v1 import plain


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['run', 'bundle', 'validation', 'warm-plan', 'sphere-audit', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Preserve earlier query test')
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    read = lambda p: json.loads(p.read_text())
    m = read(a.bundle/'manifest.json'); native = read(a.run/'sequence_native_result.json')
    if read(a.run/'job.json')['status'] != 'SUCCEEDED' or native['start_tick'] != 460 or native['case'] != 3:
        raise ValueError('Require completed native source')
    with np.load(a.run/'sequence_native_trace.npz', allow_pickle=False) as z:
        trace = {k: z[k].copy() for k in z.files}
    with np.load(a.bundle/m['bundle_name'], allow_pickle=False) as z:
        query = z['case_003_query_public'].copy()
    c = build_controller(m['configuration'])
    policy, _ = load_verified_policy(c.qp.model, c, query, a.warm_plan, a.validation)
    geometry_path = ROOT/'configs/go2_piper_authored_ground_geometry_v1.json'
    geometry = AuthoredGroundClearance(geometry_path, c.qp.model, margin_m=0.)
    protocol_path = ROOT/'configs/evaluation_protocol_v2.json'
    solver = FrictionalFeedbackTailPlannerV1(c, policy, read(a.sphere_audit)['spheres'],
        geometry, read(protocol_path)['recovery'])
    state = state_at(c.qp.model, trace, 1)
    prior = read(a.run/'recovery_partial/plan_000460.json')
    failed = read(a.run/'recovery_partial/plan_000461.json')
    if failed['decision']['accepted']:
        raise ValueError('Expected genuinely rejected old query')
    for key, value in [('observed_q', state.q), ('observed_v', state.v),
                       ('actual_previous_target_rad', state.previous_command_target)]:
        if not np.array_equal(failed[key], value):
            raise ValueError('Different observed query')
    warm = shifted_frictional_seed(prior['decision']['commands'], 1, 80, state.previous_command_target)
    bounds = c.qp.interface.limits
    student = bounds[:, 0]+.5*(trace['student_action'][1]+1.)*np.diff(bounds).ravel()
    files = [Path(__file__), a.bundle/'manifest.json', a.bundle/m['bundle_name'], a.validation,
        a.warm_plan, a.sphere_audit, a.run/'sequence_native_result.json', a.run/'sequence_native_trace.npz',
        a.run/'job.json', a.run/'recovery_partial/plan_000460.json', a.run/'recovery_partial/plan_000461.json',
        geometry_path, protocol_path, c.qp.model.path]
    files += [Path(v.__file__) for k, v in list(sys.modules.items())
              if k.startswith('scripts.') and getattr(v, '__file__', None)]
    hashes = {str(p.resolve()): sha(p) for p in files}
    before = (state.q.copy(), state.v.copy(), state.previous_command_target.copy())
    result = solver.solve(state, student, warm, control_tick=461)
    if not all(np.array_equal(x, y) for x, y in zip(before, [state.q, state.v, state.previous_command_target])):
        raise ValueError('Solver mutated the actual observation')
    for path, digest in hashes.items():
        if sha(path) != digest:
            raise ValueError('Changed solve input '+path)
    a.output.write_text(json.dumps(plain(dict(scope=__doc__, tick=461, result=result,
        source_sha256=hashes, actual_input_unchanged=True, native_execution=False,
        valid_expert_labels=False)), indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: result[k] for k in ['accepted', 'selected', 'wall_seconds', 'records']}, default=plain), flush=True)


if __name__ == '__main__':
    main()
