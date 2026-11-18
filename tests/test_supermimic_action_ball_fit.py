from scripts.fit_supermimic_action_ball import (event_validation_selection_score,
    autoregressive_validation_selection_score)


def _metrics(pre, contact, post):
    return {str(offset): {'velocity_rmse': value}
            for offset, value in ((-1, pre), (0, contact), (1, post))}


def test_event_selection_rejects_postcontact_degradation():
    initial = _metrics(2.0, 1.0, .5)
    candidate = _metrics(1.8, .8, .6)
    result = event_validation_selection_score(initial, candidate)
    assert result['eligible'] is False
    assert result['reason'] == 'postcontact_degradation'


def test_event_selection_prefers_balanced_checkpoint():
    initial = _metrics(2.0, 1.0, .5)
    candidate = _metrics(1.8, .8, .5)
    result = event_validation_selection_score(initial, candidate)
    assert result['eligible'] is True
    assert result['score'] < 1.0


def test_event_selection_rejects_missing_offset():
    initial = _metrics(2.0, 1.0, .5)
    candidate = _metrics(1.8, .8, .5)
    del candidate['-1']
    result = event_validation_selection_score(initial, candidate)
    assert result['eligible'] is False
    assert result['reason'] == 'missing_event_offset'


def _ar_metrics(post):
    return {str(offset): {'errors': {str(horizon): {'velocity_rmse': post}
                                      for horizon in (4, 16, 32, 64)}}
            for offset in (-1, 0, 1)}


def test_autoregressive_selection_rejects_late_adjacent_drift():
    initial = _ar_metrics(.5)
    candidate = _ar_metrics(.5)
    candidate['1']['errors']['64']['velocity_rmse'] = .50001
    result = autoregressive_validation_selection_score(initial, candidate)
    assert result['eligible'] is False
    assert result['reason'] == 'adjacent_autoregressive_degradation_1_64'


def test_autoregressive_selection_accepts_stable_improvement():
    initial = _ar_metrics(1.0)
    candidate = _ar_metrics(.8)
    result = autoregressive_validation_selection_score(initial, candidate)
    assert result['eligible'] is True
    assert result['score'] < 1.0
