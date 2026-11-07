"""The experimental objective must match its merit and stay process-local."""
import hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import scripts.audit_joint_velocity_recovery_v1 as experiment
import scripts.landing_sequential_recovery_search_v1 as search
import scripts.landing_joint_sequence_proposal_v1 as proposal


def test_matched_merit_and_binding_restoration_after_failed_candidate(monkeypatch):
    source_path=Path(proposal.__file__);before=hashlib.sha256(source_path.read_bytes()).hexdigest()
    original_proposal=search.propose_sequence
    def original_rollout(*args):
        return dict(cost=3.,profile=[dict(following=SimpleNamespace(joint_velocity=np.ones(18))),
            dict(following=SimpleNamespace(joint_velocity=np.full(18,2.)))])
    monkeypatch.setattr(search,'rollout_for_search',original_rollout)
    class StopAfterMeritCheck(Exception):pass
    def stop(args):
        assert search.propose_sequence is not original_proposal
        result=search.rollout_for_search(None,None,None,None)
        assert result['joint_velocity_cost']==729.
        assert result['cost']==732.
        raise StopAfterMeritCheck()
    monkeypatch.setattr(experiment,'original_query',stop)
    with pytest.raises(StopAfterMeritCheck):
        experiment.query((None,None,{},2.,experiment.generated_proposal_source(2.)))
    assert search.propose_sequence is original_proposal
    assert search.rollout_for_search is original_rollout
    assert hashlib.sha256(source_path.read_bytes()).hexdigest()==before


@pytest.mark.parametrize('weight',[0.,-1.,float('nan'),float('inf')])
def test_invalid_objective_weight_is_rejected(weight):
    with pytest.raises(ValueError):experiment.generated_proposal_source(weight)
