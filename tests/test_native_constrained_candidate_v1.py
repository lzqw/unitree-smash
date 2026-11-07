"""Contract tests only: fixtures are not simulation evidence."""
from copy import deepcopy
import numpy as np
import pytest
from scripts.prepare_native_constrained_tracking_v1 import validated_candidate


def evidence():
    step = dict(q=[0.] * 25, v=[0.] * 24, target=[0.] * 18, contacts=[True] * 4)
    step['q'][6] = 1.
    result = dict(accepted=True, substeps=160, failure=None, completed_model_steps=520,
        required_model_steps=520, terminal_violations={}, trailing_model_window_s=.8,
        native_execution=False, valid_expert_labels=False, trajectory=[deepcopy(step) for _ in range(520)])
    row = dict(substeps=160, result=result, knots_rad=np.zeros((6, 18)).tolist())
    return dict(native_execution=False, valid_expert_labels=False, full_model_candidate_found=True,
        baseline_reproduced=True, best=row, independent_replay=deepcopy(row),
        fixed_early_knots_rad=np.zeros((3, 18)).tolist())


def test_coarse_or_partial_stability_cannot_be_exported():
    for field, value in [('substeps', 32), ('completed_model_steps', 519),
                         ('trailing_model_window_s', .79), ('accepted', False)]:
        search = evidence(); search['best']['result'][field] = value
        with pytest.raises(ValueError): validated_candidate(search)


def test_independent_replay_must_match_actual_commands_and_states():
    for field in ['q', 'v', 'target']:
        search = evidence(); search['independent_replay']['result']['trajectory'][17][field][0] += .001
        with pytest.raises(ValueError): validated_candidate(search)
    search = evidence(); search['independent_replay'] = None
    with pytest.raises(ValueError): validated_candidate(search)


def test_export_preserves_candidate_without_relabelling_native_success():
    search = evidence(); knots, trajectory = validated_candidate(search)
    assert knots.shape == (6, 18) and trajectory['target'].shape == (520, 18)
    assert search['native_execution'] is False and search['valid_expert_labels'] is False
    search['best']['result']['trajectory'][0]['target'][0] = float('nan')
    with pytest.raises(ValueError): validated_candidate(search)
