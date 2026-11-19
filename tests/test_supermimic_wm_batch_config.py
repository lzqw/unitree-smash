import importlib.util
from pathlib import Path
import pytest
import scripts.train_supermimic_physical_block as m

@pytest.mark.parametrize('batch',[128,256,512,1024])
def test_preserve_exact_source_and_stage_ratios(batch):
    for q in [dict(recent_online=64,older_online=32,initial_archive=32),dict(uniform=32,pre_touchdown=24,touchdown=24,post_touchdown=24,recovery_failure=16,stable_recovery=8),dict(uniform=24,approach_active=32,hit_prepare=24,contact_release=24,approach_failure=16,post_hit=8)]:
        before=dict(q);r=m.scale_wm_quotas(q,batch)
        assert sum(r.values())==batch and q==before
        assert all(r[k]*128==q[k]*batch for k in q)

@pytest.mark.parametrize('batch',[0,127,129,-128,128.0,True])
def test_invalid_batch_rejected(batch):
    with pytest.raises(ValueError):m.scale_wm_quotas({'uniform':128},batch)

def test_missing_fraction_rejected_not_silently_normalized():
    with pytest.raises(ValueError):m.scale_wm_quotas({'uniform':127},512)

def test_parse_lane_list_supports_ranges_and_deduplicates():
    assert m.parse_lane_list('96-99,3,98,3') == [3, 96, 97, 98, 99]

def test_parse_lane_list_rejects_reverse_ranges():
    with pytest.raises(ValueError):
        m.parse_lane_list('9-4')
