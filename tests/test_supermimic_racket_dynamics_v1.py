import json
import torch
from quadsmash.rl.supermimic_racket_dynamics_v1 import racket_state,racket_loss,robot_loss,approach_window_pools
from quadsmash.rl.supermimic_full_body_geometry_v1 import FullBodyGeometryV1,MODEL


def test_racket_gradient_tracks_base_and_arm_without_ball_leak():
    g=FullBodyGeometryV1(json.loads(MODEL.read_text())['asset_sha256'],dtype=torch.float64)
    state=torch.zeros(2,55,dtype=torch.float64);state[:,3]=1.;state[:,13:31]=.2
    target=racket_state(state,g).detach()
    pred=state.clone();pred[:,0]=.03;pred[:,25]+=.1;pred.requires_grad_()
    features=racket_state(pred,g);loss=racket_loss(features,target)
    grad=torch.autograd.grad(loss,pred)[0]
    assert loss>0 and grad.isfinite().all()
    assert grad[:,:3].abs().sum()>0 and grad[:,25:31].abs().sum()>0
    assert grad[:,49:].eq(0).all()
    assert racket_loss(target,target)==0


def test_robot_loss_respects_quaternion_double_cover_and_excludes_ball():
    pred=torch.zeros(1,55);pred[:,3]=1.;target=pred.clone();target[:,3]=-1.;target[:,49:]=100.
    assert robot_loss(pred,target)==0
    pred[:,13]=.1
    torch.testing.assert_close(robot_loss(pred,target),torch.tensor(1/49))


def test_approach_sampling_keeps_prefix_reset_and_lane_boundaries():
    data={'episode':torch.zeros(12,2,dtype=torch.long),'terminated':torch.zeros(12,2,dtype=torch.bool),
          'timeout':torch.zeros(12,2,dtype=torch.bool)}
    masks={'new_racket':torch.zeros(12,2,dtype=torch.bool)};masks['new_racket'][6]=True
    data['episode'][9:]=1
    pools=approach_window_pools(data,masks,torch.tensor([0]),4)
    assert pools['racket_approach'].tolist()==[[3,0],[4,0]]
    assert pools['uniform'][:,1].eq(0).all()
    assert not ((pools['uniform'][:,0]<9)&(pools['uniform'][:,0]+3>=9)).any()
