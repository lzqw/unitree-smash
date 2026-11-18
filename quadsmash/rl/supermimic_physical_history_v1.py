"""Read-only compact physics supervision, not complete imagination/critic data.

Recorded final commands make older policies usable as dynamics observations;
unknown policy identity remains unknown and never becomes current-policy RTG.
"""
import torch
from .supermimic_physical_dataset_v1 import encode_physical,valid_windows


def compact_physical_row(row, *, num_envs, command_tolerance=2e-6):
    before,after=row['before'],row['after']
    if after['control_tick']!=before['control_tick']+1:
        raise ValueError('Nonconsecutive native transition')
    if not torch.equal(before['episode_index'],after['episode_index']):
        raise ValueError('Reset state substituted for terminal observation')
    p,q=before['physical'],after['physical']
    limits=before['auxiliary']['command_limits']
    if limits.shape!=(18,2) or not torch.isfinite(limits).all() or (limits[:,1]<=limits[:,0]).any():
        raise ValueError('Missing original command limits')
    action=row['final_action'];target=row['final_joint_targets']
    if action.shape!=(num_envs,18) or target.shape!=action.shape:
        raise ValueError('Final18-joint command shape differs')
    decoded=limits[:,0]+.5*(action.clamp(-1,1)+1)*(limits[:,1]-limits[:,0])
    delta=(decoded-target).abs().max()
    if not torch.isfinite(delta) or delta>command_tolerance:
        raise ValueError('Recorded final command does not match original absolute target mapping')
    data=dict(state=encode_physical(p),next_state=encode_physical(q),targets=target,
        final_action=action,previous_action=p['final_action'],feet=p['foot_contact'],next_feet=q['foot_contact'],
        present=p['ball_present'],elapsed=p['elapsed_s'],event_state=before['event_state'],next_event_state=after['event_state'],
        episode=before['episode_index'],terminated=row['terminated'],timeout=row['timeout'])
    if any(not isinstance(v,torch.Tensor) or v.shape[0]!=num_envs or not torch.isfinite(v).all() for v in data.values()):
        raise ValueError('Missing, nonfinite or mismatched physical transition')
    if data['event_state'].shape!=(num_envs,8) or data['next_event_state'].shape!=(num_envs,8):
        raise ValueError('Original event schema required')
    data={k:v.detach().cpu().clone() for k,v in data.items()}
    return data,dict(native_tick=int(before['control_tick']),policy_version=row.get('policy_version'),
        stream_id=row.get('stream_id'),command_max_diff=float(delta))


def historical_valid_windows(data, native_ticks, steps):
    mask=valid_windows(data['episode'],data['terminated'],data['timeout'],steps)
    ticks=torch.as_tensor(native_ticks,dtype=torch.long)
    if ticks.shape!=(len(data['state']),):raise ValueError('Tick identity missing')
    sequential=(ticks.unfold(0,steps,1).diff(dim=-1)==1).all(-1)
    return mask & sequential[:,None]
