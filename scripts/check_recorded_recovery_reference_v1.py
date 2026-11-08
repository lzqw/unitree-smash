"""Exercise recorded-command timing and the original actuator mapping on CPU."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pinocchio as pin

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.check_constrained_recovery_suffix_v1 import source_context
from scripts.search_preserved_early_support_v1 import initialize
from scripts.search_landing_support_tail_v3 import _WORKER
from scripts.recorded_recovery_reference_v1 import RecordedRecoveryReferenceV1,proposal_targets,initial_distance


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Fresh evidence required')
    source_path=ROOT/'reports/landing_early_references_20261108/recorded_state350_suffix_search_v1.json'
    source=json.loads(source_path.read_text());_,_,_,_,context=source_context(source['configuration']['source'])
    initialize(*context);w=_WORKER;model=w['controller'].qp.model;interface=w['controller'].qp.interface
    state=w['initial'];previous=state.previous_command_target.copy()
    history=np.tile(previous,(520,1));history[:,0]+=np.arange(520)*1e-5
    targets=proposal_targets(history)
    assert np.array_equal(targets[:-1],history[1:]) and np.array_equal(targets[-1],history[-1])
    policy=RecordedRecoveryReferenceV1(model,interface,state,targets)
    q,v=state.q.copy(),state.v.copy();first,_=policy.action(80,state,previous)
    assert np.array_equal(first,previous)
    last=previous.copy()
    for step in range(520):
        command,_=policy.action(80+step,state,last)
        assert np.all(command>=interface.limits[:,0]-1e-12) and np.all(command<=interface.limits[:,1]+1e-12)
        assert np.max(abs(command[12:]-last[12:]))<=.01*interface.arm_rate+1e-12
        last=command
    assert np.array_equal(q,state.q) and np.array_equal(v,state.v)
    outside=np.tile(previous,(520,1));outside[4:]=interface.limits[:,1]+10
    command,_=RecordedRecoveryReferenceV1(model,interface,state,outside).action(84,state,previous)
    assert np.all(command<=interface.limits[:,1]) and np.max(abs(command[12:]-previous[12:]))<=.01*interface.arm_rate+1e-12
    assert initial_distance(state,state)['score']==0
    donor=copy.deepcopy(state);c,s=np.cos(.73),np.sin(.73);rotation=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
    donor.q[:2]+=[2.,-3.];donor.q[3:7]=pin.Quaternion(rotation@pin.Quaternion(state.q[3:7]).matrix()).coeffs()
    score=initial_distance(state,donor)['score'];assert score<1e-24
    files=[Path(__file__),ROOT/'scripts/recorded_recovery_reference_v1.py',source_path]
    result=dict(passed=True,command_shift_exact=True,initial_handoff_exact=True,
        original_joint_and_arm_rate_limits_enforced=True,inputs_unchanged=True,
        planar_similarity_score=score,last_unobserved_command_explicit_hold=True,
        native_execution=False,model_recovery_proven=False,
        source_sha256={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
