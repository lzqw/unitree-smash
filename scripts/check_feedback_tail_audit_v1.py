"""Exercise independent plan replay and fail-closed provenance checks."""
import argparse
import copy
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
from scripts.audit_feedback_tail_plan_v1 import check_plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['candidate', 'run', 'bundle', 'validation', 'warm-plan', 'sphere-audit', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); read = lambda p: json.loads(p.read_text())
    if a.output.exists():
        raise ValueError('Preserve earlier audit evidence')
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    candidate = read(a.candidate)
    for path, digest in candidate['source_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Changed candidate source '+path)
    m = read(a.bundle/'manifest.json'); c = build_controller(m['configuration'])
    with np.load(a.bundle/m['bundle_name'], allow_pickle=False) as z:
        query = z['case_003_query_public'].copy()
    with np.load(a.run/'sequence_native_trace.npz', allow_pickle=False) as z:
        trace = {k: z[k].copy() for k in z.files}
    policy, _ = load_verified_policy(c.qp.model, c, query, a.warm_plan, a.validation)
    c.sequence_geometry = AuthoredGroundClearance(ROOT/'configs/go2_piper_authored_ground_geometry_v1.json', c.qp.model, margin_m=0.)
    protocol = read(ROOT/'configs/evaluation_protocol_v2.json')['recovery']; spheres = read(a.sphere_audit)['spheres']
    state = state_at(c.qp.model, trace, candidate['tick']-460)
    plan = dict(tick=candidate['tick'], decision=candidate['result'])
    commands = np.asarray(plan['decision']['commands'])
    files = [Path(__file__), Path(sys.modules[check_plan.__module__].__file__), a.candidate]
    hashes = dict(candidate['source_sha256']); hashes.update({str(p.resolve()): sha(p) for p in files})
    summary, prediction = check_plan(c, state, commands, spheres, policy, protocol, plan)
    checks = []
    for name in ['wrong_query', 'wrong_phase', 'false_expert_label', 'missing_fine_verification']:
        forged = copy.deepcopy(plan)
        if name == 'wrong_query': forged['decision']['initial_q'][0] += .01
        if name == 'wrong_phase': forged['decision']['observed_control_tick'] += 1
        if name == 'false_expert_label': forged['decision']['valid_expert_labels'] = True
        if name == 'missing_fine_verification': forged['decision']['records'][-1]['verification']['accepted'] = False
        try:
            check_plan(c, state, commands, spheres, policy, protocol, forged)
        except ValueError:
            checks.append(name)
        else:
            raise ValueError('Forged evidence accepted: '+name)
    for path, digest in hashes.items():
        if sha(path) != digest:
            raise ValueError('Audit input changed '+path)
    result = dict(scope=__doc__, passed=True, independent_replay=summary,
        prefix_prediction_count=len(prediction), rejected_tampered_evidence=checks,
        source_sha256=hashes, native_execution=False, valid_expert_labels=False)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(passed=True, rejected_tampered_evidence=checks,
        trailing_window_s=summary['trailing_model_kinematic_window_s'])))


if __name__ == '__main__':
    main()
