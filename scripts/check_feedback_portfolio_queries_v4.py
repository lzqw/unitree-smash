"""Verify a fixed feedback candidate portfolio through the unchanged MPC gates.

Recorded queries are measured native states. Future rollout remains model-only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_terminal_feedback_native_queries_v1 as queries
from scripts import stress_terminal_support_commands_v1 as stress
from scripts.arm_speed_reserve_feedback_v1 import ArmSpeedReserveFeedbackV1
from scripts.search_terminal_prefix_arm_margin_v1 import ArmPrefixCandidate
from scripts.frictional_feedback_portfolio_v4 import FrictionalFeedbackPortfolioV4
from scripts.audit_native_frictional_recovery_v1 import state_at
from scripts.native_frictional_recovery_v1 import shifted_frictional_seed
from scripts.audit_joint_sequence_proposal_v1 import plain


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['bundle','warm-plan','validation','sphere-audit','native-run','output']:
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): raise ValueError('Preserve prior evidence')
    queries.initialize(*map(str,[a.bundle,a.warm_plan,a.validation,a.sphere_audit,a.native_run]))
    c, base, _, _, spheres, _, _ = stress.CONTEXT
    manifest = json.loads((a.bundle/'manifest.json').read_text())
    native = json.loads((a.native_run/'sequence_native_result.json').read_text())
    policy = ArmSpeedReserveFeedbackV1(ArmPrefixCandidate(base,c.qp.interface,-.006),base,c.qp.interface)
    files = [Path(__file__),a.bundle/'manifest.json',a.warm_plan,a.validation,a.sphere_audit,
             a.native_run/'sequence_native_result.json',a.native_run/'sequence_native_trace.npz']
    files += [Path(v.__file__) for k,v in list(sys.modules.items()) if k.startswith('scripts.') and getattr(v,'__file__',None)]
    sha = lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    hashes = {str(p.resolve()):sha(p) for p in files}
    rows = []
    for tick in [460,473,478]:
        index = tick-queries.NATIVE_START
        state = state_at(c.qp.model,queries.TRACE,index)
        measured = next(r for r in native['planning_records'] if r['tick']==tick)
        for name,value in [('observed_q',state.q),('observed_v',state.v),
                           ('actual_previous_target_rad',state.previous_command_target)]:
            if not np.array_equal(measured[name],value): raise ValueError('Wrong native query '+name)
        previous = [r for r in native['planning_records'] if r['tick']<tick and r['decision']['accepted']]
        if previous:
            old = previous[-1]
            warm = shifted_frictional_seed(old['decision']['commands'],tick-old['tick'],80,state.previous_command_target)
        else:
            warm = json.loads(a.warm_plan.read_text())['final']['commands']
        limits = c.qp.interface.limits
        student = limits[:,0] + .5*(queries.TRACE['student_action'][index]+1.)*np.diff(limits).ravel()
        planner = FrictionalFeedbackPortfolioV4(manifest['configuration'],spheres,
            [('original_arm_margin',ArmPrefixCandidate(base,c.qp.interface,-.006)),('arm_speed_reserve',policy)],time_limit_s=60.)
        try: result = planner.solve(state,student,warm,control_tick=tick)
        finally: planner.close()
        row = dict(tick=tick,decision=plain(result));rows.append(row)
        print(json.dumps(dict(tick=tick,accepted=result['accepted'],wall_seconds=result['wall_seconds'],
                              selected_feedback=result['feedback_portfolio']['selected'])),flush=True)
        a.output.with_suffix('.partial.json').write_text(json.dumps(dict(rows=rows,complete=False),allow_nan=False)+'\n')
    for path,digest in hashes.items():
        if sha(path)!=digest: raise ValueError('Input changed '+path)
    a.output.write_text(json.dumps(dict(scope=__doc__,rows=rows,complete=True,source_sha256=hashes,
        native_execution=False,valid_expert_labels=False),indent=2,allow_nan=False)+'\n')


if __name__=='__main__': main()
