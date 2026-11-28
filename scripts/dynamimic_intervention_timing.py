"""Case-bound timing of training-only native interventions."""
import hashlib
import json

import torch


def schedule_from_baseline(folder, cases, checkpoint, companion, *, air_preparation=False, first_support=False):
    if air_preparation and first_support:
        raise ValueError("Choose a single intervention phase")
    storage = folder/'trajectories'
    payload = (storage/'manifest.json').read_bytes()
    manifest = json.loads(payload)
    if (manifest['schema'] != 'dynamimic-native-recovery-v1' or manifest['dt'] != .01
            or manifest['cases'] != cases or manifest['checkpoint_sha256'] != checkpoint
            or manifest['companion_sha256'] != companion):
        raise ValueError('Intervention timing requires the same baseline behavior and launches')
    fields = {k:[] for k in ('active','body','landed','live')}
    for name in manifest['chunks']:
        chunk = torch.load(storage/name, weights_only=True, mmap=True, map_location='cpu')
        for key in ('active','body','landed','live'):
            fields[key].append(chunk[key])
        if first_support:
            fields.setdefault('feet', []).append(chunk['feet'])
        if air_preparation or first_support:
            fields.setdefault('release_age',[]).append(chunk['obs'][...,775] - chunk['release_time'])
    data = {k:torch.cat(parts) for k,parts in fields.items()}
    if first_support:
        starts = first_support_starts(data)
    elif air_preparation:
        starts = air_preparation_starts(data)
    else:
        starts = prefall_starts(data)
    if not (starts >= 0).any():
        raise ValueError('No eligible native intervention windows')
    result = dict(start_ticks=starts.tolist(), duration_ticks=16,
                  baseline_manifest_sha256=hashlib.sha256(payload).hexdigest())
    if air_preparation:
        result['phase'] = 'air_preparation'
    if first_support:
        result['phase'] = 'first_support'
    return result


def air_preparation_starts(data):
    # The current tick's landed signal is observed after executing its action.
    active = data['active']
    end = len(active)
    previously_landed = torch.zeros_like(data['landed'])
    previously_landed[1:] = data['landed'][:-1]
    eligible = active & data['live'] & ~previously_landed & (data['release_age'] >= .05)
    ticks = torch.arange(end, device=active.device)[:,None]
    start = torch.where(eligible,ticks,end).amin(0)
    return torch.where(start+16 < end,start,-1)


def first_support_starts(data):
    # Landed and feet describe the boundary after applying that tick's command.
    landed = data['landed'] & data['live']
    onset = landed.clone()
    onset[1:] &= ~landed[:-1]
    onset &= data['feet'].bool().any(-1)
    eligible = torch.zeros_like(landed)
    eligible[1:] = onset[:-1]
    eligible &= data['active'] & data['live'] & (data['release_age'] >= .04)
    end = len(landed)
    ticks = torch.arange(end, device=landed.device)[:, None]
    start = torch.where(eligible, ticks, end).amin(0)
    return torch.where(start + 16 < end, start, -1)


def prefall_starts(data):
    active = data['active']
    end = len(active)
    ticks = torch.arange(end)[:,None]
    landing = torch.where(data['live'] & data['landed'],ticks,end).amin(0)
    fault = torch.where(active & data['body'],ticks,end).amin(0)
    # Intervene before the recorded fall. Compare with other eligible lanes
    # close to landing. Future outcome selects data, never the policy input.
    start = torch.where(fault < end, fault-32, landing+16)
    valid = (start >= 0) & (start+16 < end)
    idx = start.clamp(0,end-1)
    lane = torch.arange(len(idx))
    valid &= active[idx,lane]
    return torch.where(valid,start,-1)


def pulse_ticks(pulse, count):
    if 'start_ticks' in pulse:
        ticks = pulse['start_ticks']
        if not isinstance(ticks, list) or any(type(t) is not int for t in ticks):
            raise ValueError('Per-lane intervention ticks must be integers')
        start = torch.tensor(ticks, dtype=torch.long)
        if start.shape != (count,) or (start < -1).any() or 'start_tick' in pulse:
            raise ValueError('Malformed per-lane intervention timing')
        return start
    return torch.full((count,),int(pulse['start_tick']),dtype=torch.long)
