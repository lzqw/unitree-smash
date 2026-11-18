import pytest
import torch
from quadsmash.rl.supermimic_response_fit_sources_v1 import draw_source_windows


def pools():
    return [{k:torch.tensor([[source*100+i,1] for i in range(40)]) for k in ('contact','pre_contact','post_contact','background')} for source in range(3)]


def test_source_event_quota_is_actual_not_posthoc_label():
    rows,report=draw_source_windows(pools(),dict(contact=256,pre_contact=128,post_contact=64,background=64),torch.Generator().manual_seed(3))
    assert len(rows)==512 and [x['count'] for x in report]==[256,128,128]
    assert [int(((rows[:,0]//100)==i).sum()) for i in range(3)]==[256,128,128]
    assert report[0]['stages']==dict(contact=128,pre_contact=64,post_contact=32,background=32)
    assert all(x['unique_windows']<=40 for x in report)


def test_original_only_has_no_forced_extra_source_or_fabrication():
    rows,report=draw_source_windows(pools()[:1],dict(contact=64,pre_contact=32,post_contact=16,background=16),torch.Generator().manual_seed(3))
    assert len(rows)==128 and len(report)==1 and rows[:,0].max()<100


def test_missing_stream_and_fractional_quota_rejected():
    with pytest.raises(ValueError):draw_source_windows(pools()[:2],dict(contact=128),torch.Generator())
    with pytest.raises(ValueError):draw_source_windows(pools(),dict(contact=3),torch.Generator())


def test_combination_preserves_lane_split_and_never_spans_stream_gap():
    from quadsmash.rl.supermimic_response_fit_sources_v1 import combine_sources
    from quadsmash.rl.supermimic_physical_dataset_v1 import valid_windows
    n=48
    def dataset():
        d=dict(state=torch.zeros(n,128,55),next_state=torch.zeros(n,128,55),episode=torch.zeros(n,128,dtype=torch.long),terminated=torch.zeros(n,128,dtype=torch.bool),timeout=torch.zeros(n,128,dtype=torch.bool),event_state=torch.zeros(n,128,8),next_event_state=torch.zeros(n,128,8))
        d['event_state'][21:,:,1]=1;d['next_event_state'][20:,:,1]=1
        return d
    initial=dataset();extra=dataset();ticks=list(range(n));ticks[40:]=list(range(100,100+n-40))
    v=valid_windows(extra['episode'],extra['terminated'],extra['timeout'],16)
    v&=(torch.tensor(ticks).unfold(0,16,1).diff(dim=-1)==1).all(-1)[:,None]
    package=dict(tensors=extra,native_metadata=[dict(native_tick=t) for t in ticks],window16_valid=v)
    merged,pools,ids=combine_sources(initial,[package,package],list(initial),list(range(96)),16)
    assert len(merged['state'])==n*3 and ids.tolist()==[0]*n+[1]*n+[2]*n
    assert merged['episode'][0,0]!=merged['episode'][n,0] and merged['episode'][n,0]!=merged['episode'][2*n,0]
    for index,source in enumerate(pools):
        for pairs in source.values():
            assert (pairs[:,1]<96).all()
            for tick,lane in pairs.tolist():
                assert tick//n==index and (tick+15)//n==index
                if index:assert v[tick%n,lane]
    assert initial['episode'].sum()==0
