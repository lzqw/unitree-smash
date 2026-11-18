"""Multiple verified physical-supervision streams for a bounded WM repair.

No historical action/return becomes an Actor input or a current-policy label.
All original lane ids survive; window continuity includes explicit stream gaps.
"""
import hashlib,json
from pathlib import Path
import torch
from .supermimic_physical_dataset_v1 import valid_windows
from .supermimic_action_ball_v1 import contact_neighborhood_pools


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def load_history(directory,contract,joint_names,split_sha):
    root=Path(directory);receipt=json.loads((root/'completed.json').read_text())
    if receipt['status']!='PACKED_VERIFIED_DYNAMICS_ONLY' or receipt['split_sha256']!=split_sha:
        raise ValueError('Historical source completion or split changed')
    result=[];provenance=[]
    if len(receipt['sources'])!=2:raise ValueError('This repair requires the two predeclared historical streams')
    for item in receipt['sources']:
        path=Path(item['path'])
        if sha(path)!=item['sha256']:raise ValueError('Historical compact source hash changed')
        p=torch.load(path,map_location='cpu',weights_only=True)
        if p['schema']!='historical-physical-supervision-only-v1' or p['split_manifest_sha256']!=split_sha:
            raise ValueError('Unknown or incompatible historical data schema')
        m=p['source_manifest']
        if m['task_contract']!=contract or m['joint_names']!=joint_names or m['num_envs']!=128:
            raise ValueError('Historical physical execution contract differs')
        if p['eligibility']['actor_starts'] or p['eligibility']['critic_rtg']:
            raise ValueError('Physical-only package eligibility changed')
        result.append(p)
        provenance.append(dict(path=str(path),sha256=item['sha256'],source_archive=p['source_archive'],
            source_manifest_sha256=p['source_manifest_sha256'],unknown_policy_version_ticks=item['unknown_policy_version_ticks'],
            eligibility=p['eligibility']))
    return result,provenance


def combine_sources(initial,extras,keys,train_lanes,steps=16):
    packages=[dict(tensors=initial,native_metadata=None)]+extras
    parts=[];valids=[];source_ids=[];offset=0;episode_offset=0;pool_by_source=[]
    train=torch.zeros(128,dtype=torch.bool);train[train_lanes]=True
    for index,p in enumerate(packages):
        d={k:p['tensors'][k].clone() for k in keys}
        n=len(d['state'])
        if n<steps or d['state'].shape[1]!=128:raise ValueError('Original128 contiguous windows required')
        local=valid_windows(d['episode'],d['terminated'],d['timeout'],steps)
        if p.get('native_metadata') is not None:
            ticks=torch.tensor([x['native_tick'] for x in p['native_metadata']])
            if len(ticks)!=n:raise ValueError('Missing historical native tick identity')
            local&=(ticks.unfold(0,steps,1).diff(dim=-1)==1).all(-1)[:,None]
            if not torch.equal(local,p['window16_valid']):raise ValueError('Historical window validity changed')
        eligible=torch.zeros_like(d['episode'],dtype=torch.bool);eligible[:len(local)]=local;eligible&=train[None]
        edges=(d['next_event_state'][...,1]>d['event_state'][...,1])|(d['next_event_state'][...,6]>d['event_state'][...,6])
        pools=contact_neighborhood_pools(edges,d['episode'],d['terminated'],d['timeout'],eligible)
        if any(not len(v) for v in pools.values()):raise ValueError('Historical source lacks required contact/neighbourhood support')
        # Window indices may be concatenated; episode labels are offset to keep
        # distinct source streams distinct. Original identities remain in source files.
        local_episode=d['episode'].long();d['episode']=local_episode-local_episode.min()+episode_offset
        episode_offset=int(d['episode'].max())+1
        pool_by_source.append({k:v+torch.tensor([offset,0]) for k,v in pools.items()})
        parts.append(d);source_ids.append(torch.full((n,),index,dtype=torch.long));offset+=n
    merged={k:torch.cat([d[k] for d in parts]) for k in keys}
    return merged,pool_by_source,torch.cat(source_ids)


def draw_source_windows(pools,stage_counts,generator):
    """Fixed initial/history50/25/25; original-only preserves previous draws."""
    if len(pools) not in (1,3):raise ValueError('Only original or original plus two declared history sources')
    weights=(4,) if len(pools)==1 else (2,1,1)
    selected=[];counts=[]
    for index,(source,weight) in enumerate(zip(pools,weights)):
        allocation={k:n*weight//4 for k,n in stage_counts.items()}
        if any(n*weight%4 for n in stage_counts.values()):raise ValueError('Exact source/event quotas require divisible counts')
        rows=torch.cat([source[k][torch.randint(len(source[k]),(n,),generator=generator)] for k,n in allocation.items()])
        selected.append(rows);counts.append(dict(source=index,count=len(rows),stages=allocation,unique_windows=len(torch.unique(rows,dim=0))))
    return torch.cat(selected),counts
