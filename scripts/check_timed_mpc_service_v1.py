"""Run a real measured-state suffix through the private CPU MPC service."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.check_constrained_recovery_suffix_v1 import source_context,state_from_row
from scripts.search_preserved_early_support_v1 import initialize
from scripts.search_landing_support_tail_v3 import _WORKER
from scripts.timed_whole_body_recovery_v1 import build_timed_plan
from scripts.current_state_timed_recovery_service_v4 import CurrentStateTimedRecoveryServiceV4
from scripts.audit_joint_sequence_proposal_v1 import plain


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['model-plan','run','bundle','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Fresh service check required')
    read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    m=read(a.bundle/'manifest.json');setup=read(a.model_plan)
    if read(a.run/'job.json')['status']!='SUCCEEDED':raise ValueError('Completed measured run required')
    parent=read(setup['source']);_,_,_,geometry,context=source_context(parent['configuration']['source'])
    context=list(copy.deepcopy(context));initial=m['query']
    if context[0]!=m['configuration']:raise ValueError('Changed physical interface')
    context[1].update(initial_q=initial['q'],initial_v=initial['v'],initial_contacts=initial['contacts'],actual_previous_command=initial['target'],tick=280)
    with np.load(a.bundle/m['bundle_name'],allow_pickle=False) as z:
        context[4]=z['tracking_reference_q'].copy();context[5]=z['tracking_reference_v'].copy()
    context[6]=m['tracking_gains'];initialize(*context)
    with np.load(a.run/'sequence_native_trace.npz',allow_pickle=False) as z:
        i=700-280;state=state_from_row(dict(q=z['q'][i],v=z['v'][i],contacts=z['contacts'][i],target=z['previous_target_rad'][i]))
    pins=dict(m['source_sha256'])
    for path in [a.model_plan,a.bundle/'manifest.json',a.run/'sequence_native_trace.npz',Path(__file__).resolve()]:pins[str(path.resolve())]=sha(path)
    for name,module in list(sys.modules.items()):
        if name.startswith('scripts.') and getattr(module,'__file__',None):pins[str(Path(module.__file__).resolve())]=sha(module.__file__)
    service=CurrentStateTimedRecoveryServiceV4(tuple(context),geometry,pins,workers=2,max_seconds=120.)
    try:
        answer=service.solve(state,700,m['proposal'])
        if not answer['accepted'] or not answer['independent_replay_exact']:raise ValueError('Actual suffix was not accepted')
        target,_=build_timed_plan(_WORKER,answer['plan']).action(80+700-280,state,state.previous_command_target)
        if not np.array_equal(target,answer['first_command']):raise ValueError('Service/native reconstruction differs')
        result=dict(passed=True,ready=service.ready,answer=answer,first_command_exact=True,source_sha256=pins,
            native_execution=False,valid_expert_labels=False)
    finally:service.close()
    if service.process.poll() is None:raise ValueError('CPU service remains live after close')
    result['service_exit_code']=service.process.returncode
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(plain(result),indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['passed','ready','first_command_exact','service_exit_code']}))


if __name__=='__main__':main()
