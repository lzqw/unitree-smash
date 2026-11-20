import torch

from scripts.audit_supermimic_recovery_transfer import event_starts


def test_post_strike_panel_requires_complete_same_episode_at_both_support_edges():
    ticks,lanes=100,3
    before=torch.zeros(ticks,lanes,4,dtype=torch.bool)
    before[2:5]=True
    before[21:26]=True
    after=torch.cat([before[1:],before[-1:]])
    contact=torch.zeros(ticks,lanes,dtype=torch.bool)
    contact[9,:2]=True
    episode=torch.zeros(ticks,lanes,dtype=torch.long)
    episode[87:,1]=1
    terminated=torch.zeros_like(contact)
    terminated[86,1]=True
    starts=event_starts(dict(episode=episode,contact=contact,feet=before,next_feet=after,
                             terminated=terminated,timeout=torch.zeros_like(contact)))
    # Lane one's touchdown has a full window. Its subsequent support loss crosses its reset.
    assert starts==dict(touchdown=[(20,0),(20,1)],subsequent_support_loss=[(25,0)])
