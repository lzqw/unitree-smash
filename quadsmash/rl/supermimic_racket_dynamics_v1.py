"""Task-space supervision for robot dynamics; no changes to native physics."""
import torch
from .supermimic_physical_dataset_v1 import decode_physical, valid_windows


def racket_state(state, geometry):
    p=decode_physical(state)
    b=geometry(p['joint_position'],p['joint_velocity'],p['root_position_w'],p['root_quaternion_w'],
               p['root_link_velocity_w'],p['root_angular_velocity_w'])
    i=geometry.ids['Racket']
    return torch.cat((b['position'][:,i],b['rotation'][:,i].flatten(1),
                      b['velocity'][:,i],b['angular_velocity'][:,i]),-1)


def racket_loss(predicted, target):
    # Physical-unit training scales; never collision or success tolerances.
    scales=predicted.new_tensor([.03]*3+[.1]*9+[1.]*3+[3.]*3)
    error=((predicted-target)/scales).square()
    return sum(error[:,a:b].mean() for a,b in ((0,3),(3,12),(12,15),(15,18)))/4


def robot_loss(predicted,target):
    # Ball predictions must not directly dominate a robot-specific fit.
    same=(predicted[:,3:7]*target[:,3:7]).sum(-1,keepdim=True)>=0
    aligned=torch.cat((target[:,:3],torch.where(same,target[:,3:7],-target[:,3:7]),target[:,7:49]),-1)
    scales=predicted.new_tensor([.05]*3+[.1]*4+[1.]*3+[3.]*3+[.1]*18+[3.]*18)
    return ((predicted[:,:49]-aligned)/scales).square().mean()


def approach_window_pools(data,masks,lanes,steps=32):
    if steps<2:raise ValueError('Need a nonempty approach prefix')
    valid=valid_windows(data['episode'],data['terminated'],data['timeout'],steps)
    allowed=torch.zeros(valid.shape[1],dtype=torch.bool,device=valid.device);allowed[lanes]=True
    valid=valid&allowed[None]
    # Native labels only choose supervised training windows; never model inputs.
    approach=masks['new_racket'].unfold(0,steps,1)[...,steps//2:].any(-1)&valid
    pools={'uniform':valid.nonzero(),'racket_approach':approach.nonzero()}
    if any(not len(v) for v in pools.values()):raise ValueError('Empty approach/uniform pool')
    return pools
