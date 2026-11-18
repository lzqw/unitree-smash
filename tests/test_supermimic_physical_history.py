import copy
import pytest
import torch
from quadsmash.rl.supermimic_physical_dataset_v1 import STATE_FIELDS
from quadsmash.rl.supermimic_physical_history_v1 import compact_physical_row,historical_valid_windows


def row():
    p={k:torch.zeros(2,n) for k,n in STATE_FIELDS}
    p.update(final_action=torch.zeros(2,18),foot_contact=torch.zeros(2,4),ball_present=torch.ones(2,dtype=torch.bool),elapsed_s=torch.ones(2))
    b=dict(physical=p,episode_index=torch.zeros(2,dtype=torch.long),control_tick=10,event_state=torch.zeros(2,8),auxiliary=dict(command_limits=torch.tensor([[-1.,1.]]*18)))
    a=copy.deepcopy(b);a['control_tick']=11;a['physical']['root_position_w']+=.01
    return dict(before=b,after=a,final_action=torch.zeros(2,18),final_joint_targets=torch.zeros(2,18),terminated=torch.tensor([False,True]),timeout=torch.zeros(2,dtype=torch.bool))


def test_physical_only_does_not_invent_unknown_policy_or_rtg():
    r=row();d,meta=compact_physical_row(r,num_envs=2)
    assert meta['policy_version'] is None and meta['stream_id'] is None
    assert 'reward' not in d and 'values' not in d and 'executor' not in d
    assert d['terminated'][1] and d['next_state'][1,0]==pytest.approx(.01)
    r['after']['physical']['root_position_w'].zero_()
    assert d['next_state'][1,0]==pytest.approx(.01)


@pytest.mark.parametrize('issue',['reset','target','missing','nonfinite'])
def test_incomplete_or_inconsistent_supervision_rejected(issue):
    r=row()
    if issue=='reset':r['after']['episode_index'][0]+=1
    elif issue=='target':r['final_joint_targets'][0,0]=.1
    elif issue=='missing':del r['before']['physical']['final_action']
    else:r['after']['physical']['ball_velocity_w'][0,0]=float('nan')
    with pytest.raises((KeyError,ValueError)):compact_physical_row(r,num_envs=2)


def test_window_never_crosses_tick_gap_or_reset():
    data=dict(state=torch.zeros(5,2,55),episode=torch.zeros(5,2,dtype=torch.long),terminated=torch.zeros(5,2,dtype=torch.bool),timeout=torch.zeros(5,2,dtype=torch.bool))
    data['terminated'][1,0]=True;data['episode'][2:,0]=1
    valid=historical_valid_windows(data,[0,1,2,4,5],2)
    assert valid.tolist()==[[True,True],[False,True],[False,False],[True,True]]
