"""Evaluate an explicit terminal continuation from completed native queries."""
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
    for name in ['run', 'bundle', 'validation', 'warm-plan', 'sphere-audit', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--ticks', type=int, nargs='+', default=[461])
    parser.add_argument('--substeps', type=int, choices=[16, 160], default=16)
    a = parser.parse_args()
    if a.output.exists():
        raise ValueError('Preserve previous evidence')
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    read = lambda p: json.loads(p.read_text())
    manifest = read(a.bundle/'manifest.json'); native = read(a.run/'sequence_native_result.json')
    job = read(a.run/'job.json')
    if job['status'] != 'SUCCEEDED' or native['start_tick'] != 460 or native['case'] != 3:
        raise ValueError('Require completed ground-start native source')
    archive = a.bundle/manifest['bundle_name']
    if sha(archive) != manifest['bundle_sha256'] or sha(a.sphere_audit) != native['sphere_audit_sha256']:
        raise ValueError('Changed model source')
    with np.load(archive, allow_pickle=False) as z:
        query = z['case_003_query_public'].copy()
    with np.load(a.run/'sequence_native_trace.npz', allow_pickle=False) as z:
        trace = {k: z[k].copy() for k in z.files}
    controller = build_controller(manifest['configuration'])
    policy, _ = load_verified_policy(controller.qp.model, controller, query, a.warm_plan, a.validation)
    geometry_path = ROOT/'configs/go2_piper_authored_ground_geometry_v1.json'
    geometry = AuthoredGroundClearance(geometry_path, controller.qp.model, margin_m=0.)
    protocol_path = ROOT/'configs/evaluation_protocol_v2.json'
    protocol = read(protocol_path)['recovery']; spheres = read(a.sphere_audit)['spheres']
    files = [Path(__file__), a.bundle/'manifest.json', archive, a.validation, a.warm_plan, a.sphere_audit,
             a.run/'sequence_native_result.json', a.run/'sequence_native_trace.npz', a.run/'job.json',
             geometry_path, protocol_path, controller.qp.model.path]
    files += [Path(v.__file__) for k, v in list(sys.modules.items())
              if k.startswith('scripts.') and getattr(v, '__file__', None)]
    hashes = {str(p.resolve()): sha(p) for p in files}; rows = []
    for tick in a.ticks:
        if not 460 <= tick < 800:
            raise ValueError('Original recovery phase required')
        state = state_at(controller.qp.model, trace, tick-460)
        result = evaluate_feedback_tail(controller, state, policy, spheres, geometry, protocol,
                                        control_tick=tick, substeps=a.substeps)
        rows.append(dict(tick=tick, initial_q=state.q, initial_v=state.v,
                         initial_contacts=state.contacts, previous_command=state.previous_command_target,
                         result=result))
        print(json.dumps(dict(tick=tick, **{k: result[k] for k in ['accepted', 'failure',
            'prefix_terminal', 'terminal', 'terminal_violations', 'trailing_model_kinematic_window_s',
            'completed_model_steps', 'wall_seconds']})), flush=True)
    for path, digest in hashes.items():
        if sha(path) != digest:
            raise ValueError('Changed input '+path)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(plain(dict(scope=__doc__, rows=rows, source_sha256=hashes,
        old_80_step_endpoint_replaced_by_explicit_continuation=True,
        task_thresholds_changed=False, native_execution=False, valid_expert_labels=False)),
        indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
