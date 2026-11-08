"""Check timing-wrapper equivalence over actual full-episode state/command inputs."""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pinocchio as pin
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.native_experimental_recovery_v2 import NativeExperimentalRecoveryV2
from scripts.loaded_pinocchio_model_v1 import PublicState
from scripts.search_preserved_early_support_v1 import joint_factory
from scripts.timed_whole_body_recovery_v1 import build_timed_plan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['bundle','run','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Fresh diagnostic required')
    m=json.loads((a.bundle/'manifest.json').read_text())
    native=json.loads((a.run/'sequence_native_result.json').read_text())
    if not native['full_feedback_sequence_executed'] or not native['query_match']['passed']:
        raise ValueError('Complete matched actual feedback inputs required')
    adapter=NativeExperimentalRecoveryV2(a.bundle,m['case'],torch.tensor(m['configuration']['interface']['limits']),m['policy_sha256'],a.output.parent)
    with np.load(a.bundle/m['bundle_name'],allow_pickle=False) as z:
        factory=partial(joint_factory,reference_q=z['tracking_reference_q'].copy(),reference_v=z['tracking_reference_v'].copy(),gains=m['tracking_gains'])
    worker=dict(controller=adapter.controller,initial=adapter.initial,factory=factory)
    plan=dict(m['proposal'],contract='timed_foot_space_recovery_v1',early_timing_s=[.08,.20,.50,.04])
    default=build_timed_plan(worker,plan)
    changed=build_timed_plan(worker,dict(plan,early_timing_s=[.04,.10,.25,.04]))
    with np.load(a.run/'sequence_native_trace.npz',allow_pickle=False) as z:
        trace={key:z[key].copy() for key in z.files}
    exact=True;maximum=0.;recorded=0.;changed_steps=[]
    for i,row in enumerate(native['records']):
        q=trace['q'][i];v=trace['v'][i];previous=trace['previous_target_rad'][i]
        state=PublicState(q,v,pin.Quaternion(q[3:7]).matrix(),q[:3].copy(),
            q[adapter.model.q_indices].copy(),v[adapter.model.v_indices].copy(),trace['contacts'][i],0.)
        state.previous_command_target=previous.copy();offset=80+row['tick']-280
        old,_=adapter.feedback.action(offset,state,previous)
        new,_=default.action(offset,state,previous)
        other,_=changed.action(offset,state,previous)
        exact &= np.array_equal(old,new)
        maximum=max(maximum,float(np.max(abs(old-new))))
        recorded=max(recorded,float(np.max(abs(old-trace['requested_target_rad'][i]))))
        if np.max(abs(other-old))>1e-6:changed_steps.append(row['tick'])
        if (np.any(other<adapter.limits[:,0]-1e-7) or np.any(other>adapter.limits[:,1]+1e-7)
                or np.max(abs(other[12:]-previous[12:]))>.100001):raise ValueError('Changed timing violates interface')
    if len(native['records'])!=520 or not exact or recorded>1e-6 or not changed_steps:
        raise ValueError('Timing wrapper equivalence or actual command reconstruction failed')
    paths=[a.bundle/'manifest.json',a.bundle/m['bundle_name'],a.run/'sequence_native_result.json',
        a.run/'sequence_native_trace.npz',Path(__file__).resolve(),ROOT/'scripts/timed_whole_body_recovery_v1.py']
    result=dict(scope=__doc__,passed=True,steps=520,default_exact=bool(exact),
        default_maximum_difference_rad=maximum,recorded_request_maximum_difference_rad=recorded,
        changed_timing_action_steps=changed_steps,changed_timing_interface_valid=True,
        new_native_execution=False,valid_expert_labels=False,
        source_sha256={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['source_sha256','changed_timing_action_steps']}))


if __name__=='__main__':main()
