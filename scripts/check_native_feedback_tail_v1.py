"""Exercise the real new adapter at a recorded rejected replan, without physics."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.native_feedback_tail_v1 import NativeFeedbackTailV1
from scripts.audit_native_frictional_recovery_v1 import state_at
from scripts.audit_joint_sequence_proposal_v1 import plain


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['bundle', 'warm-plan', 'validation', 'sphere-audit', 'native-run', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Preserve earlier adapter check')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    out = a.output.parent/'adapter_trace'; out.mkdir()
    read = lambda p: json.loads(p.read_text())
    manifest = read(a.bundle/'manifest.json')
    adapter = NativeFeedbackTailV1(a.bundle, 3, torch.tensor(manifest['configuration']['interface']['limits']),
        manifest['policy_sha256'], out, a.warm_plan, a.sphere_audit, a.validation)
    with np.load(a.native_run/'sequence_native_trace.npz', allow_pickle=False) as z:
        trace = {k: z[k].copy() for k in z.files}
    prior_path = a.native_run/'recovery_partial/plan_000460.json'; prior_plan = read(prior_path)
    adapter.plan = np.asarray(prior_plan['decision']['commands'])
    adapter.plan_start = 460; adapter.plan_contacts = np.asarray(prior_plan['observed_contacts'], bool)
    adapter.predicted_states = prior_plan['decision']['predicted_states']
    state = state_at(adapter.model, trace, 1)
    state.measured_force_world = trace['foot_force_world_n'][1].copy()
    student = trace['student_action'][1].copy(); previous = state.previous_command_target.copy()
    files = [Path(__file__), a.bundle/'manifest.json', a.warm_plan, a.validation, a.sphere_audit,
             a.native_run/'sequence_native_trace.npz', prior_path]
    files += [Path(v.__file__) for k, v in list(sys.modules.items())
              if k.startswith('scripts.') and getattr(v, '__file__', None)]
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    hashes = {str(p.resolve()): sha(p) for p in files}
    try:
        target, mode = adapter.feedback_target(461, state, student, previous)
    finally:
        adapter.planner.close()
    row = adapter.plans[-1]; decision = row['decision']
    if (not decision['accepted'] or decision['observed_control_tick'] != 461
            or adapter.planner.tick is not None or not row['contact_changed']
            or mode != 'frictional_receding_support_sequence'
            or not np.array_equal(target, decision['commands'][0])
            or not np.array_equal(row['observed_q'], state.q)
            or not np.array_equal(row['observed_v'], state.v)
            or not np.array_equal(row['actual_previous_target_rad'], previous)):
        raise ValueError('Wrong observed phase, state, history or selected target')
    encoded = torch.tensor(2.*(target-adapter.limits[:, 0])/np.diff(adapter.limits).ravel()-1., dtype=torch.float32)
    reconstructed = adapter.decode(encoded.numpy())
    mapping_error = float(np.max(abs(reconstructed-target)))
    if mapping_error > 1e-6 or np.max(abs(reconstructed[12:]-previous[12:])) > .100001:
        raise ValueError('Native float32 target mapping violates the command contract')
    for path, digest in hashes.items():
        if sha(path) != digest:
            raise ValueError('Input changed during adapter test')
    result = dict(scope=__doc__, passed=True, observed_tick=461, decision=decision,
        contact_change_caused_replan=True, current_query_and_history_used=True,
        actual_phase_binding_cleared=True, command_matches_selected_first_step=True,
        maximum_float32_mapping_error_rad=mapping_error,
        source_sha256=hashes, native_execution=False, valid_expert_labels=False)
    a.output.write_text(json.dumps(plain(result), indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(passed=True, selected=decision['selected'], wall_seconds=decision['wall_seconds'],
        maximum_float32_mapping_error_rad=mapping_error, native_execution=False)))


if __name__ == '__main__':
    main()
