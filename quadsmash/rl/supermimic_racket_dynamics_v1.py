"""Task-space supervision for robot dynamics; no changes to native physics."""
import torch
from .supermimic_full_body_geometry_v1 import racket_kinematics
from .supermimic_physical_dataset_v1 import decode_physical, valid_windows


def racket_state(state, geometry):
    center, rotation, velocity, omega = racket_kinematics(decode_physical(state), geometry)
    return torch.cat((center, rotation.flatten(1), velocity, omega), -1)


def racket_loss_components(predicted, target, *, per_row=False):
    """Return task-space position, velocity, normal and angular losses.

    The first racket rotation column is the authored racket-face normal used
    by the task geometry.  Keeping its cosine loss separate prevents a large
    orientation matrix error from hiding a normal-direction error.
    """
    if predicted.shape != target.shape or predicted.shape[-1] != 18:
        raise ValueError('Racket state must be [batch,18] with matching target')
    position=((predicted[:,:3]-target[:,:3])/.03).square().mean(-1)
    rotation_pred=predicted[:,3:12].reshape(-1,3,3)
    rotation_target=target[:,3:12].reshape(-1,3,3)
    normal_pred=torch.nn.functional.normalize(rotation_pred[:,:,0],dim=-1)
    normal_target=torch.nn.functional.normalize(rotation_target[:,:,0],dim=-1)
    normal=1.-(normal_pred*normal_target).sum(-1)
    velocity=((predicted[:,12:15]-target[:,12:15])/1.).square().mean(-1)
    angular=((predicted[:,15:18]-target[:,15:18])/3.).square().mean(-1)
    parts=dict(position=position, normal=normal, velocity=velocity, angular=angular)
    return parts if per_row else {name: value.mean() for name, value in parts.items()}


def racket_loss(predicted, target):
    # Physical-unit training scales; never collision or success tolerances.
    components=racket_loss_components(predicted,target)
    return sum(components.values())/len(components)


def robot_loss(predicted,target,*,per_row=False):
    # Ball predictions must not directly dominate a robot-specific fit.
    same=(predicted[:,3:7]*target[:,3:7]).sum(-1,keepdim=True)>=0
    aligned=torch.cat((target[:,:3],torch.where(same,target[:,3:7],-target[:,3:7]),target[:,7:49]),-1)
    scales=predicted.new_tensor([.05]*3+[.1]*4+[1.]*3+[3.]*3+[.1]*18+[3.]*18)
    error=((predicted[:,:49]-aligned)/scales).square()
    return error.mean(-1) if per_row else error.mean()


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
