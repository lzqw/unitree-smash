from types import SimpleNamespace

import torch

from quadsmash.rl.supermimic_actuator_identification import actuator_contract, drive_features
from quadsmash.rl.supermimic_physical_observation_v1 import JOINT_NAMES


def test_native_drive_rows_follow_joint_names_and_keep_per_lane_gains():
    ids = list(reversed(range(18)))
    d = SimpleNamespace(**{k:torch.ones(3,18) for k in
        ('joint_stiffness','joint_damping','joint_effort_limits','joint_vel_limits')},
        joint_pos_limits=torch.tensor([-1.,1.]).expand(3,18,2))
    d.joint_stiffness[1,17] = 7.
    env = SimpleNamespace(joint_ids=ids, num_envs=3, step_dt=.01,
        robot=SimpleNamespace(data=d,joint_names=list(reversed(JOINT_NAMES)),actuators={}),
        cfg=SimpleNamespace(sim=SimpleNamespace(dt=.001),decimation=10))
    result=actuator_contract(env,{'asset':{'sha256':'example'}})
    assert len(result['profiles'])==2
    assert [result['profiles'][i]['kp'][0] for i in result['lane_profile']]==[1,7,1]


def test_drive_torque_uses_target_error_damping_and_per_joint_saturation():
    state = torch.zeros(2,55)
    state[0,31:49] = .5
    target = torch.full((2,18),.1,requires_grad=True)
    kp = torch.full((18,),10.); kd=torch.ones(18)
    p=dict(kp=kp,kd=kd,effort_limit=torch.ones(18)*.75,velocity_limit=torch.ones(18))
    torque, x=drive_features(state,target,torch.zeros_like(target),torch.zeros_like(target),
                             torch.ones_like(target),p)
    torch.testing.assert_close(torque[:,0],torch.tensor([.5,.75]))
    torch.testing.assert_close(x[:,0,7],torch.tensor([0.,1.]))
    torque.sum().backward()
    torch.testing.assert_close(target.grad[:,0],torch.tensor([10.,0.]))
