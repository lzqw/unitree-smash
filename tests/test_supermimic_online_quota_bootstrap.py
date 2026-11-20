import pytest

from quadsmash.rl.supermimic_training_audit_v1 import collect_until_quota, sampling_promotion_gate


def test_sampler_controls_frozen_collection_and_does_not_recollect_ready_replay():
    ticks = []
    def ready():
        sampling = dict(source_shortfall={'older_online':int(sum(ticks)<300)}, fallback_count=0)
        return sampling_promotion_gate(sampling, 'promotion')['passed']
    assert collect_until_quota(ready,ticks.append,300,interval=128)==300
    assert ticks==[128,128,44]
    assert collect_until_quota(ready,ticks.append,300)==0


def test_bootstrap_rechecks_both_heads_at_limit_and_fails_before_fit():
    calls=[]
    def ready():
        return all(sampling_promotion_gate(dict(fallback_count=n), 'promotion')['passed'] for n in (0,1))
    with pytest.raises(RuntimeError,match='exhausted 3 ticks'):
        collect_until_quota(ready,calls.append,3,interval=2)
    assert calls==[2,1]
