import torch
from quadsmash.rl.supermimic_ball_sequence_v1 import ball_window_pools,ball_error


def test_event_windows_do_not_cross_reset_or_heldout_lane():
    episode=torch.tensor([[0,0],[0,0],[0,0],[1,0],[1,0],[1,0]])
    terminal=torch.zeros_like(episode,dtype=torch.bool);terminal[2,0]=True
    data={'episode':episode,'terminated':terminal,'timeout':torch.zeros_like(terminal)}
    masks={k:torch.zeros_like(terminal) for k in ('new_racket','new_ground','new_net')}
    masks['new_racket'][2,:]=True;masks['new_net'][4,1]=True
    pools=ball_window_pools(data,masks,torch.tensor([0]),3)
    assert pools['uniform'].tolist()==[[0,0],[3,0]]
    assert pools['new_racket'].tolist()==[[0,0]]
    assert 'new_net' not in pools


def test_physical_scales_do_not_remove_gradients_or_clip_outliers():
    prediction=torch.tensor([[.1,0.,0.,2.,0.,0.]],requires_grad=True)
    loss=ball_error(prediction,torch.zeros_like(prediction))
    torch.testing.assert_close(loss,torch.tensor(1/3))
    loss.backward();assert prediction.grad[0,0]>0 and prediction.grad[0,3]>0
    assert ball_error(2*prediction.detach(),torch.zeros_like(prediction))>loss.detach()
