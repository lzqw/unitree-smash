import copy
import json
from pathlib import Path
import pytest
from quadsmash.rl.supermimic_terminal_scoring import attach_reset_jump_records, score_terminal_episode

ROOT = Path(__file__).resolve().parents[1]


def source(case):
    rows = json.loads((ROOT/'tests/data/supermimic_terminal_source_cases.json').read_text())['rows']
    row = copy.deepcopy(next(r for r in rows if r['source_training_index'] == case))
    # Additional metadata required by the new capture. No simulated states are
    # invented; missing independent head audit deliberately remains unknown.
    row.update(episode_duration_s=8., episode_timeout=True,
               qualified_airborne_string_contact=True,
               ball_side_complete_except_recovery=case != 22,
               body_ground_seen=row['jump']['body_landing'],
               rally_fault=case == 22, complete5=case == 10)
    protocol = json.loads((ROOT/'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json').read_text())
    protocol['_root'] = str(ROOT)
    return row, protocol


@pytest.mark.parametrize('case,expected', [(10,(True,True,True)), (22,(False,False,False)), (82,(False,False,False))])
def test_real_historical_cases_preserve_independent_definition(case, expected):
    row, protocol = source(case)
    before = copy.deepcopy(row)
    out = score_terminal_episode(row,protocol)
    assert (out['recovery_success'],out['ball_side_complete_except_recovery'],out['complete5']) == expected
    assert out['head_contact'] is None and out['independent_nonfoot_contact'] is None
    assert json.dumps(row,sort_keys=True) == json.dumps(before,sort_keys=True)
    if case == 22:
        assert out['physical_recovery_success'] is True
        assert out['raw_environment_flags']['recovery_success'] is True
        assert 'no_rule_fault' in out['rejection_reasons']
    if case == 82:
        assert out['raw_environment_flags']['ball_side_complete_except_recovery'] is True
        assert 'released_downward_forward' in out['rejection_reasons']


def test_missing_jump_is_not_rebuilt_from_another_run():
    row, protocol = source(10)
    del row['jump']
    with pytest.raises(ValueError, match='Missing terminal jump'):
        score_terminal_episode(row,protocol)


def test_jump_episode_must_match():
    row, protocol = source(10)
    row['jump']['episode_index'] = 1
    with pytest.raises(ValueError, match='another episode'):
        score_terminal_episode(row,protocol)


def test_reset_batch_join_handles_different_order_and_copies():
    rallies = [dict(env_id=7,episode_index=4),dict(env_id=1,episode_index=9)]
    jumps = [dict(env_id=1,recovery_s=.8),dict(env_id=7,recovery_s=.9)]
    attach_reset_jump_records(rallies,jumps)
    assert rallies[0]['jump'] == dict(env_id=7,recovery_s=.9,episode_index=4)
    assert rallies[1]['jump']['episode_index'] == 9
    jumps[1]['recovery_s'] = 0
    assert rallies[0]['jump']['recovery_s'] == .9


@pytest.mark.parametrize('rallies,jumps', [
    ([dict(env_id=1,episode_index=0)],[]),
    ([dict(env_id=1,episode_index=0)],[dict(env_id=2)]),
    ([dict(env_id=1,episode_index=0)],[dict(env_id=1),dict(env_id=1)]),
])
def test_cannot_join_missing_or_ambiguous_evidence(rallies,jumps):
    with pytest.raises(ValueError):
        attach_reset_jump_records(rallies,jumps)
