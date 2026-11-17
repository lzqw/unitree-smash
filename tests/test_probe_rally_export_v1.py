"""CPU tests for production small-panel first-episode scoring inputs."""
from copy import deepcopy

import pytest

from scripts.probe_rally_export_v1 import first_episode_probe_rows


def _inputs():
    cases = [{'case_id': 'TRAIN-0'}, {'case_id': 'TRAIN-64'}]
    indices = [0, 64]
    rally = [
        {'env_id': 0, 'episode_index': 0, 'case_id': 'TRAIN-0',
         'complete_smash': False},
        {'env_id': 1, 'episode_index': 0, 'case_id': 'TRAIN-64',
         'complete_smash': True},
        # Must never substitute a later episode for the first.
        {'env_id': 0, 'episode_index': 1, 'case_id': 'TRAIN-0',
         'complete_smash': True},
    ]
    jump = [
        {'env_id': 1, 'qualified_jump_landed': True,
         'stable_recovery': True},
        {'env_id': 0, 'qualified_jump_landed': False,
         'stable_recovery': False},
        {'env_id': 0, 'qualified_jump_landed': True,
         'stable_recovery': True},
    ]
    return rally, jump, cases, indices


def test_native_first_episode_requires_original_jump_and_rally_records():
    rally, jump, cases, indices = _inputs()
    before_rally, before_jump = deepcopy(rally), deepcopy(jump)
    rows = first_episode_probe_rows(rally, jump, cases, indices)
    assert [r['source_training_index'] for r in rows] == [0, 64]
    assert [r['source_case_index'] for r in rows] == [0, 64]
    assert [r['case_id'] for r in rows] == ['TRAIN-0', 'TRAIN-64']
    assert [r['episode_index'] for r in rows] == [0, 0]
    assert [r['jump']['qualified_jump_landed'] for r in rows] == [False, True]
    assert [r['ground_init'] for r in rows] == [True, True]
    assert not rows[0]['complete_smash']
    assert rally == before_rally and jump == before_jump


def test_missing_jump_fails_without_synthesizing_zero_or_false():
    rally, jump, cases, indices = _inputs()
    jump = [row for row in jump if row['env_id'] == 0]
    with pytest.raises(ValueError, match='jump='):
        first_episode_probe_rows(rally, jump, cases, indices)


def test_duplicate_first_rally_record_is_not_silently_replaced():
    rally, jump, cases, indices = _inputs()
    rally.insert(1, deepcopy(rally[0]))
    with pytest.raises(ValueError, match='repeated first-rally'):
        first_episode_probe_rows(rally, jump, cases, indices)


def test_bad_case_identity_or_missing_first_episode_is_rejected():
    rally, jump, cases, indices = _inputs()
    rally[0]['case_id'] = 'WRONG'
    with pytest.raises(ValueError, match='case identity'):
        first_episode_probe_rows(rally, jump, cases, indices)
    rally, jump, cases, indices = _inputs()
    rally = [row for row in rally if not
             (row['env_id'] == 1 and row['episode_index'] == 0)]
    with pytest.raises(ValueError, match='rally='):
        first_episode_probe_rows(rally, jump, cases, indices)
