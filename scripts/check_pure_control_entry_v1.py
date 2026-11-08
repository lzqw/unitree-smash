"""Native-entry/data/interface checks, not a landing performance evaluation."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.check_constrained_recovery_suffix_v1 import source_context,state_from_row
from scripts.search_preserved_early_support_v1 import initialize
from scripts.search_landing_support_tail_v3 import _WORKER
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.pure_control_recovery_v1 import PureControlRecoveryV1
from scripts.event_recovery_manager_v1 import RecoveryCommandArbiterV1
from scripts.audit_joint_sequence_proposal_v1 import plain


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['entry','configuration','inputs','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();read=lambda p:json.loads(Path(p).read_text())
    if a.output.exists():raise ValueError('Fresh check output required')
    text=a.entry.read_text();ast.parse(text)
    if ('native_replay.case' in text or 'native_replay.start' in text or 'replay_state' in text
        or text.count('action=native_replay.action(tick,obs,action,finished)')!=1
        or text.count('native_replay.verify_applied(finished)')!=1):raise ValueError('Incorrect causal command wiring')
    block=text[text.index('action=native_replay.action(tick,obs,action,finished)'):text.index('observation, _, done, _ = wrapped.step(action)')]
    if 'action = ' in block or 'action=' in block.split('\n',1)[1]:raise ValueError('Another action writer after recovery')
    cfg=read(a.configuration);x=read(a.inputs);source=read(read(x['configuration']['model_plan'])['source'])
    *_,ctx=source_context(source['configuration']['source']);initialize(*ctx)
    original=state_from_row(x['query']);nominal=np.asarray(ctx[0]['posture_target'])
    # Use the evaluator's actual default posture, not an old hand-written tail.
    nominal=read(ROOT/'reports/native_joint_timed71_all128_20261108_02/native_command_inputs.json')['default_joint_position_rad']
    rows=[]
    for mode in ['measured','airborne','one_foot','angular_perturbation']:
        state=state_from_row(x['query']);before=state.q.copy()
        if mode=='airborne':state.contacts[:]=False
        if mode=='one_foot':state.contacts[:]=False;state.contacts[0]=True
        if mode=='angular_perturbation':state.v[3]+=.1
        c=PureControlRecoveryV1(PositionPDInterface(**cfg['interface']),nominal)
        target,d=c.command(state,'AIR_PREPARE' if mode=='airborne' else 'BUFFER_LOAD',3.)
        final,_=RecoveryCommandArbiterV1(c.interface.limits).select(None,target,state.previous_command_target,owns_recovery=True)
        if not np.array_equal(state.q,before) or not np.isfinite(final).all():raise ValueError('State mutated or nonfinite command')
        if np.max(abs(final[12:]-state.previous_command_target[12:]))>.100001:raise ValueError('Changed arm interface')
        rows.append(dict(mode=mode,target=final,diagnostic=d))
    if np.array_equal(rows[0]['target'],rows[-1]['target']):raise ValueError('No measured angular-rate feedback')
    result=dict(passed=True,checks=rows,entry_sha256=hashlib.sha256(a.entry.read_bytes()).hexdigest(),
        native_execution=False,physical_landing_passed=False,valid_expert_labels=False)
    a.output.write_text(json.dumps(plain(result),indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=True,solves=[r['diagnostic']['solver_accepted'] for r in rows],native_execution=False)))


if __name__=='__main__':main()
