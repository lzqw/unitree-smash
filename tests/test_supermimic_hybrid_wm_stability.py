import pytest

from scripts.report_supermimic_hybrid_wm_stability import summarize


def _comparison(contact):
    initial = {
        '-1': {'velocity_rmse': 2.0},
        '0': {'velocity_rmse': 1.0},
        '1': {'velocity_rmse': .5},
    }
    candidate = {key: dict(value) for key, value in initial.items()}
    candidate['0'] = {'velocity_rmse': contact}
    ar = {
        offset: {'errors': {str(h): {'velocity_rmse': 1.0}
                            for h in (4, 16, 32, 64)}}
        for offset in ('-1', '0', '1')
    }
    return dict(initial=initial, candidate=candidate, initial_autoregressive=ar,
                candidate_autoregressive=ar, candidate_sha256='a' * 64,
                actor_updates=0, world_updates=0, critic_updates=0)


def test_hybrid_stability_requires_contact_improvement():
    report = summarize(_comparison(1.0))
    assert report['accepted'] is False
    assert report['reason'] == 'contact_edge_not_improved'


def test_hybrid_stability_accepts_strict_contact_and_rollout_improvement():
    comparison = _comparison(.8)
    comparison['candidate_autoregressive'] = {
        offset: {'errors': {str(h): {'velocity_rmse': .8}
                            for h in (4, 16, 32, 64)}}
        for offset in ('-1', '0', '1')
    }
    report = summarize(comparison)
    assert report['accepted'] is True
    assert report['prediction_status'] == 'STABLE_CANDIDATE'


def test_hybrid_stability_rejects_missing_hash():
    comparison = _comparison(.8)
    comparison['candidate_sha256'] = None
    with pytest.raises(ValueError):
        summarize(comparison)
