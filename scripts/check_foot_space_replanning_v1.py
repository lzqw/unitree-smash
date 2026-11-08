"""Verify a current-state full-tail solve preserves the foot-space warm plan.

This uses a model state, explicitly not a native rescue. A separate recorded
native query may subsequently use the same planner without resetting physics.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.check_constrained_recovery_suffix_v1 import source_context,state_from_row
from scripts.recovery_family_queries_v1 import transfer_reference
from scripts.search_preserved_early_support_v1 import initialize
from scripts.current_state_recovery_planner_v3 import CurrentStateRecoveryPlannerV3,build_feedback_plan
from scripts.search_landing_support_tail_v3 import _WORKER
from scripts.audit_joint_sequence_proposal_v1 import plain


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['plan','search','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Fresh check output required')
    read=lambda path:json.loads(Path(path).read_text())
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    setup=read(a.plan);search=read(a.search)
    if not search['verified']:raise ValueError('Independently replayed full model solution required')
    source=read(setup['source']);_,_,_,geometry,context=source_context(source['configuration']['source'])
    initialize(*context)
    context,_=transfer_reference(context,state_from_row(search['query']))
    initialize(*context)
    warm=dict(contract='foot_space_current_state_plan_v1',original_knots_rad=search['original_knots'],
        transition_knots_rad=search['verified']['replay']['transition_knots'],braking=search['braking'],corrections=[])
    trajectory=search['verified']['replay']['result']['trajectory']
    tick=400;i=tick-280
    query={key:trajectory[i-1][key] for key in ['q','v','contacts','target']}
    state=state_from_row(query)
    policy=build_feedback_plan(_WORKER,warm)
    target,_=policy.action(80+tick-280,state,state.previous_command_target)
    if not np.array_equal(target,trajectory[i]['target']):raise ValueError('Warm policy or original clock changed')
    # This test deliberately changes the supplied current velocity while
    # retaining the same reference, to verify feedback is state-dependent.
    changed=state_from_row(query);changed.v=changed.v.copy();changed.v[4]+=.1
    target_changed,_=policy.action(80+tick-280,changed,changed.previous_command_target)
    response=float(np.max(abs(target_changed-target)))
    if response<=1e-8:raise ValueError('Current state does not affect the control request')
    pins=dict(search['source_sha256'])
    pins[str(a.search.resolve())]=sha(a.search)
    for name,module in list(sys.modules.items()):
        if name.startswith('scripts.') and getattr(module,'__file__',None):
            path=str(Path(module.__file__).resolve());pins[path]=sha(path)
    pins[str(Path(__file__).resolve())]=sha(__file__)
    for path,digest in pins.items():
        if sha(path)!=digest:raise ValueError('Changed input '+path)
    planner=CurrentStateRecoveryPlannerV3(context,geometry,workers=2,max_seconds=600.)
    try:result=planner.solve(state,tick,warm,preceding_stable_ticks=0)
    finally:planner.close()
    if not result['accepted'] or not result['independent_replay_exact']:
        raise ValueError('Verified nominal suffix was rejected')
    predicted=result['prediction']['trajectory']
    exact=(len(predicted)==len(trajectory[i:])==400 and all(np.array_equal(x[k],y[k])
        for x,y in zip(predicted,trajectory[i:]) for k in ['q','v','contacts','target']))
    if not exact:raise ValueError('Current-state solve changed the nominal full suffix')
    if result['plan']['corrections']:raise ValueError('Unexpected optimization on nominal warm plan')
    for path,digest in pins.items():
        if sha(path)!=digest:raise ValueError('Changed input during check '+path)
    output=dict(passed=True,case=87,control_tick=tick,full_remaining_steps=len(predicted),
        original_clock_preserved=True,reference_state_preserved=True,query_is_native=False,
        independent_replay_exact=True,nominal_suffix_exact=True,feedback_response_rad=response,
        solver_seconds=result['wall_seconds'],terminal=result['prediction']['terminal'],
        trailing_model_window_s=result['prediction']['trailing_model_window_s'],
        native_execution=False,valid_expert_labels=False,source_sha256=pins)
    a.output.write_text(json.dumps(plain(output),indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k!='source_sha256'}))


if __name__=='__main__':main()
