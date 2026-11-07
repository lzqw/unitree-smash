"""Re-solve one recorded native failure from its measured state and actual history.

The proposed future remains a model result, not a restored native episode.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_native_frictional_recovery_v1 import state_at
from scripts.frictional_receding_recovery_v2 import FrictionalRecedingRecoveryV2
from scripts.audit_joint_sequence_proposal_v1 import plain


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['run', 'bundle', 'sphere-audit', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): raise ValueError('Preserve previous diagnosis')
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    result_path = a.run/'sequence_native_result.json'
    data = json.loads(result_path.read_text())
    manifest = json.loads((a.bundle/'manifest.json').read_text())
    if not data['planning_failed']: raise ValueError('No native failure to repair')
    failure = data['planning_records'][-1]
    if failure['decision']['accepted']: raise ValueError('Last query did not fail')
    previous = data['planning_records'][-2]
    if not previous['decision']['accepted']: raise ValueError('No last accepted plan')
    steps = failure['tick']-previous['tick']
    if not 1 <= steps <= 10: raise ValueError('Unexpected replan interval')
    old = np.asarray(previous['decision']['commands'])
    warm = np.vstack([old[steps:], np.repeat(old[-1:], steps, axis=0)])
    with np.load(a.run/'sequence_native_trace.npz', allow_pickle=False) as z:
        trace = {k:z[k].copy() for k in z.files}
    spheres = json.loads(a.sphere_audit.read_text())['spheres']
    planner = FrictionalRecedingRecoveryV2(manifest['configuration'], spheres)
    index = failure['tick']-data['start_tick']
    state = state_at(planner.controller.qp.model, trace, index)
    for field, value in [('observed_q', state.q), ('observed_v', state.v),
                         ('actual_previous_target_rad', state.previous_command_target)]:
        if not np.array_equal(failure[field], value): raise ValueError('Native query mismatch '+field)
    limits = planner.controller.qp.interface.limits
    student = limits[:,0]+.5*(trace['student_action'][index]+1.)*np.diff(limits).ravel()
    files = [Path(__file__), result_path, a.run/'sequence_native_trace.npz',
        a.bundle/'manifest.json', a.sphere_audit, planner.controller.qp.model.path]
    files += [Path(v.__file__) for k,v in list(sys.modules.items())
        if k.startswith('scripts.') and getattr(v,'__file__',None)]
    hashes = {str(f.resolve()):sha(f) for f in files}
    started = time.monotonic()
    try: decision = planner.solve(state, student, warm)
    finally: planner.close()
    for name, digest in hashes.items():
        if sha(name) != digest: raise ValueError('Input changed '+name)
    out = dict(scope=__doc__,case=data['case'],control_tick=failure['tick'],
        original_failure=failure['decision'],decision=decision,
        measured_native_initial_state=True,native_continuation_executed=False,
        valid_expert_labels=False,source_sha256=hashes,wall_seconds=time.monotonic()-started)
    a.output.write_text(json.dumps(plain(out),indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(tick=failure['tick'],accepted=decision['accepted'],
        wall_seconds=out['wall_seconds'],native_continuation_executed=False)),flush=True)


if __name__ == '__main__':
    main()
