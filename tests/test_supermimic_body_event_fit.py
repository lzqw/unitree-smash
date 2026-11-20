import torch
from scripts.fit_supermimic_mixed_body_rollout import event_pools


def test_native_event_windows_preserve_split_and_separate_support_from_strike():
    episode = torch.zeros(12,2,dtype=torch.long)
    done = torch.zeros_like(episode,dtype=torch.bool)
    done[9] = True
    episode[10:] = 1
    events = torch.zeros(12,2,8)
    events[...,0] = 1
    events[5:10,:,1] = 1
    feet = torch.zeros(12,2,4,dtype=torch.bool)
    feet[6:8] = True
    data = dict(episode=episode,terminated=done,timeout=torch.zeros_like(done),
                event_state=events,next_event_state=events.roll(-1,0),feet=feet,next_feet=feet.roll(-1,0))
    pools=event_pools(data,torch.tensor([True,False]),2,True)
    groups={k:set(map(tuple,v.tolist())) for k,v in pools.items()}
    assert groups['racket']=={(3,0),(4,0)}
    assert groups['recovery']=={(5,0),(6,0),(7,0)}
    assert not groups['natural'].intersection(groups['recovery']|groups['racket'])
    assert all(lane==0 and tick!=9 for group in groups.values() for tick,lane in group)
