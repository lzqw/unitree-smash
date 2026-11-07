import copy
import pytest
from scripts.evaluate_landing_reference_bank_v1 import select_prediction


def prediction(name, accepted=True, cost=1.):
    return dict(candidate=name, state_sha256='same-current-state', result=dict(accepted=accepted,
        required_model_steps=520, completed_model_steps=520, failure=None,
        terminal_violations={}, trailing_model_window_s=1., cost=cost))


def test_feasible_baseline_preserved_despite_alternative_lower_cost():
    baseline=prediction('baseline',cost=10.); other=prediction('alternative',cost=.1)
    assert select_prediction([baseline,other]) is baseline


def test_only_complete_feasible_alternative_can_be_selected():
    baseline=prediction('baseline',False); other=prediction('alternative')
    assert select_prediction([baseline,other]) is other
    other['result']['accepted']=False
    assert select_prediction([baseline,other]) is None
    other['result']['accepted']=True;other['result']['completed_model_steps']=100
    with pytest.raises(ValueError):select_prediction([baseline,other])


def test_mixed_states_and_duplicate_candidate_names_rejected():
    baseline=prediction('baseline');other=prediction('alternative')
    other['state_sha256']='different-state'
    with pytest.raises(ValueError):select_prediction([baseline,other])
    with pytest.raises(ValueError):select_prediction([baseline,copy.deepcopy(baseline)])
