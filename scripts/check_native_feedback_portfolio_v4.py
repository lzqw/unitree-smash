"""Exercise the real native adapter on a recorded state, without native physics."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.native_feedback_portfolio_v4 import NativeFeedbackPortfolioV4
from scripts.audit_native_frictional_recovery_v1 import state_at
from scripts.audit_joint_sequence_proposal_v1 import plain


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['bundle','warm-plan','validation','sphere-audit','native-run','output']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve earlier check')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    trace_out=a.output.parent/'adapter_trace';trace_out.mkdir()
    manifest=json.loads((a.bundle/'manifest.json').read_text())
    adapter=NativeFeedbackPortfolioV4(a.bundle,3,torch.tensor(manifest['configuration']['interface']['limits']),
        manifest['policy_sha256'],trace_out,a.warm_plan,a.sphere_audit,a.validation)
    with np.load(a.native_run/'sequence_native_trace.npz',allow_pickle=False) as z:
        trace={k:z[k].copy() for k in z.files}
    tick=473;index=tick-460;state=state_at(adapter.model,trace,index)
    state.measured_force_world=trace['foot_force_world_n'][index].copy()
    student=trace['student_action'][index].copy();prior=state.previous_command_target.copy()
    files=[Path(__file__),a.bundle/'manifest.json',a.warm_plan,a.validation,a.sphere_audit,
           a.native_run/'sequence_native_trace.npz']
    files += [Path(v.__file__) for k,v in list(sys.modules.items()) if k.startswith('scripts.') and getattr(v,'__file__',None)]
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    hashes={str(p.resolve()):sha(p) for p in files}
    try:target,mode=adapter.feedback_target(tick,state,student,prior)
    finally:adapter.planner.close()
    plan=adapter.plans[-1];decision=plan['decision']
    if (not decision['accepted'] or decision['observed_control_tick']!=tick or adapter.planner.tick is not None
            or mode!='frictional_receding_support_sequence' or not np.array_equal(target,decision['commands'][0])
            or not np.array_equal(plan['observed_q'],state.q) or not np.array_equal(plan['observed_v'],state.v)
            or not np.array_equal(plan['actual_previous_target_rad'],prior)):
        raise ValueError('Adapter changed phase, actual state or returned command')
    for path,digest in hashes.items():
        if sha(path)!=digest:raise ValueError('Source changed '+path)
    result=dict(scope=__doc__,passed=True,observed_tick=tick,decision=plain(decision),
        command_matches_accepted_first_step=True,actual_phase_binding_cleared=True,
        source_sha256=hashes,native_execution=False,valid_expert_labels=False)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=True,selected=decision['feedback_portfolio']['selected'],
                         wall_seconds=decision['wall_seconds'],native_execution=False)))


if __name__=='__main__':main()
