from scripts.fit_supermimic_action_ball import event_validation_selection_score


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
