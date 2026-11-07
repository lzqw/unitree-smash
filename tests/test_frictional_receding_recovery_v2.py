"""Admission fault injection; physics effectiveness is checked by separate rollouts."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from scripts.frictional_receding_recovery_v2 import FrictionalRecedingRecoveryV2


def fixture():
    path=Path(__file__).parent/'fixtures/frictional_short_plan_case3.npz'
    meta=json.loads(path.with_suffix('.json').read_text())
    planner=FrictionalRecedingRecoveryV2(meta['configuration'],meta['spheres'])
    with np.load(path) as z:
        state=SimpleNamespace(q=z['q'].copy(),v=z['v'].copy(),contacts=z['contacts'].copy(),
            previous_command_target=z['previous_command'].copy())
    return planner,state,np.repeat(state.previous_command_target[None],80,axis=0)


class FaultPool:
    def __init__(self,kind):self.kind=kind;self.calls=0;self.closed=False
    def close(self):self.closed=True
    def batch(self,requests):
        self.calls+=1;results=[]
        for r in requests:
            state=r['state'];previous=state.previous_command_target.copy()
            if self.kind=='stale':previous[0]+=.1
            results.append(dict(solved=True,accepted=not(self.kind=='verification' and self.calls==2),
                maximum_scaled_violation=0.,cost=0.,commands=r['seed'].copy(),states=[],
                input_q=state.q.copy(),input_v=state.v.copy(),input_previous_command=previous))
        return results


def test_independent_rejection_cannot_become_executable_plan():
    p,s,w=fixture();p.pool=FaultPool('verification')
    with pytest.raises(ValueError,match='independent verification'):p.solve(s,w[0],w)
    assert p.pool.calls==2


def test_wrong_actual_command_history_is_rejected():
    p,s,w=fixture();p.pool=FaultPool('stale')
    with pytest.raises(ValueError,match='stale input_previous_command'):p.solve(s,w[0],w)


def test_expired_budget_never_returns_candidate_as_action():
    p,s,w=fixture();pool=FaultPool('none');p.pool=pool;p.time_limit_s=1e-12
    result=p.solve(s,w[0],w)
    assert not result['accepted'] and result['commands'] is None and pool.closed
    assert pool.calls==0
