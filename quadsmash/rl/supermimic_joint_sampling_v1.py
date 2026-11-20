"""Exact source/stage quota flow on unique window membership signatures.

Episode caps are jointly solved with quotas using bounded integer allocation.
Infeasible quotas yield explicit shortfalls; labels and duplicates are never invented.
"""
from collections import deque, Counter
import torch

class Flow:
    def __init__(self): self.edges={}
    def add(self,a,b,capacity):
        edge=[b,int(capacity),None,int(capacity)];reverse=[a,0,edge,0];edge[2]=reverse
        self.edges.setdefault(a,[]).append(edge);self.edges.setdefault(b,[]).append(reverse)
        return edge
    def solve(self,s,t):
        total=0
        while True:
            parent={s:None};queue=deque([s])
            while queue and t not in parent:
                a=queue.popleft()
                for e in self.edges.get(a,[]):
                    if e[1]>0 and e[0] not in parent:parent[e[0]]=(a,e);queue.append(e[0])
            if t not in parent:return total
            node=t;amount=10**9
            while node!=s:node0,e=parent[node];amount=min(amount,e[1]);node=node0
            node=t
            while node!=s:node0,e=parent[node];e[1]-=amount;e[2][1]+=amount;node=node0
            total+=amount

def capped_allocation(sources, pattern, identities, names, sq, tq, batch, cap, source_quota):
    """Integer counts per (source, membership, episode), not per-transition binaries."""
    import numpy as np
    from scipy.optimize import milp, Bounds, LinearConstraint
    from scipy.sparse import csc_matrix
    groups={}
    for i,(source,signature,episode) in enumerate(zip(sources.tolist(),pattern.tolist(),identities)):
        groups.setdefault((source,signature,episode),[]).append(i)
    groups=list(groups.items()); variables=[]; rows={}; ri=[];ci=[];values=[];limits=[]
    def constraint(key,limit):
        if key not in rows:rows[key]=len(limits);limits.append(limit)
        return rows[key]
    # Exact source targets when complete. Partial configured source quotas keep
    # the existing wildcard semantics and are resolved by the flow path only.
    if source_quota and sum(sq.values())!=batch:
        raise ValueError('Episode-capped source quotas must sum to batch')
    for gi,((source,signature,episode),ids) in enumerate(groups):
        for j,name in enumerate(names):
            if not signature&(1<<j):continue
            col=len(variables);variables.append((gi,name))
            for key,limit in ((('source',source),sq.get(source,0)),(('stage',name),tq[name]),
                              (('group',gi),len(ids)),(('episode',episode),cap)):
                ri.append(constraint(key,limit));ci.append(col);values.append(1.)
    if not variables:return [torch.tensor(ids) for _,ids in groups],[],0
    matrix=csc_matrix((values,(ri,ci)),shape=(len(limits),len(variables)))
    result=milp(-np.ones(len(variables)),integrality=np.ones(len(variables)),
        bounds=Bounds(0,batch),constraints=LinearConstraint(matrix,0,np.asarray(limits)),
        options={'time_limit':10.0,'mip_rel_gap':0.0})
    if not result.success:raise RuntimeError(f'Joint quota allocator did not establish optimum: {result.message}')
    allocation=np.rint(result.x).astype(int)
    if not np.allclose(result.x,allocation,atol=1e-6) or (matrix@allocation>np.asarray(limits)+1e-6).any():
        raise RuntimeError('Invalid integer quota allocation')
    return ([torch.tensor(ids,dtype=torch.long) for _,ids in groups],
            [(gi,name,int(count)) for (gi,name),count in zip(variables,allocation) if count],int(allocation.sum()))

def sample_joint(stage_pools,source_labels,batch,generator,*,source_quota=None,stage_quota=None,episode_keys=None,max_per_episode=8):
    if batch<1 or max_per_episode<1:raise ValueError('Positive batch and episode cap required')
    sq=dict(source_quota or {});tq=dict(stage_quota or {'uniform':batch})
    if any(v<0 for v in list(sq.values())+list(tq.values())) or sum(sq.values())>batch or sum(tq.values())>batch:raise ValueError('Invalid quotas')
    pools={k:v for k,v in stage_pools.items() if len(v)}
    if not pools:raise ValueError('No valid windows available for stratified sampling')
    pairs=torch.unique(torch.cat(list(pools.values())),dim=0);width=source_labels.shape[1]
    codes=pairs[:,0]*width+pairs[:,1]
    pools.setdefault('uniform',pairs)
    if sum(tq.values())<batch:tq['uniform']=tq.get('uniform',0)+batch-sum(tq.values())
    names=list(tq)
    if len(names)>20:raise ValueError('Too many stage signatures')
    memberships=torch.stack([torch.isin(codes,pools.get(k,pairs[:0])[:,0]*width+pools.get(k,pairs[:0])[:,1]) for k in names],1)
    sources=source_labels[pairs[:,0],pairs[:,1]]
    if not sq:sq={int(s):batch for s in sources.unique()}
    # Free source quota remainder is an explicit wildcard, not a false source.
    pattern=(memberships.long()*(2**torch.arange(len(names)))).sum(1)
    grouped=torch.stack((sources,pattern),1);unique,inverse=torch.unique(grouped,dim=0,return_inverse=True)
    flow=Flow();S=('root',);T=('sink',);links=[]
    for source,count in sq.items():flow.add(S,('source',source),count)
    if source_quota and sum(sq.values())<batch:
        flow.add(S,('free',),batch-sum(sq.values()))
        for source in sources.unique().tolist():flow.add(('free',),('source',source),batch)
    group_rows=[]
    for gi,(source,signature) in enumerate(unique.tolist()):
        ids=torch.where(inverse==gi)[0];group_rows.append(ids)
        flow.add(('source',source),('group',gi),len(ids))
        for j,name in enumerate(names):
            if signature&(1<<j):links.append((gi,name,flow.add(('group',gi),('stage',name),len(ids))))
    for stage,count in tq.items():flow.add(('stage',stage),T,count)
    allocated=flow.solve(S,T)
    selected=[];allocated_stage=Counter();used=set();ep_counts=Counter()
    identities=episode_keys[pairs[:,0],pairs[:,1]].tolist() if episode_keys is not None else codes.tolist()
    assignments=[(gi,stage,edge[3]-edge[1]) for gi,stage,edge in links]
    if episode_keys is not None:
        group_rows,assignments,allocated=capped_allocation(sources,pattern,identities,names,sq,tq,batch,max_per_episode,source_quota)
    def draw(ids,want):
        order=ids[torch.randperm(len(ids),generator=generator)].tolist();chosen=[]
        # First favor less represented episodes, using shuffle for seeded ties.
        order.sort(key=lambda i:ep_counts[identities[i]])
        for i in order:
            ep=identities[i]
            if i not in used and ep_counts[ep]<max_per_episode:
                used.add(i);ep_counts[ep]+=1;selected.append(i);chosen.append(i)
                if len(chosen)==want:break
        return len(chosen)
    # Scarce event groups first. Flow has already rerouted flexible quotas.
    for gi,stage,want in sorted(assignments,key=lambda x:len(group_rows[x[0]])):
        if want:allocated_stage[stage]+=draw(group_rows[gi],want)
    before_fallback=len(selected)
    if len(selected)<batch:draw(torch.arange(len(pairs)),batch-len(selected))
    if len(selected)!=batch:
        raise ValueError(f'Insufficient unique windows under episode cap: {len(selected)}/{batch}; cap={max_per_episode}')
    chosen=pairs[torch.tensor(selected)];selected_sources=sources[selected]
    counts={int(s):int((selected_sources==s).sum()) for s in sources.unique().tolist()}
    coverage={k:int(torch.isin(chosen[:,0]*width+chosen[:,1],v[:,0]*width+v[:,1]).sum()) for k,v in dict(stage_pools,**{'uniform':pools['uniform']}).items()}
    actual_stage_counts={k:int(coverage.get(k,0)) for k in names}
    return chosen,dict(source_counts=counts,stage_counts={k:allocated_stage[k] for k in names},
        stage_actual_counts=actual_stage_counts,
        source_requested=source_quota or {},stage_requested=stage_quota or tq,
        source_shortfall={k:max(0,v-counts.get(k,0)) for k,v in (source_quota or {}).items()},
        stage_shortfall={k:max(0,v-actual_stage_counts.get(k,0)) for k,v in tq.items()},
        actual_event_coverage=coverage,fallback_count=batch-before_fallback,
        joint_flow_count=allocated,joint_quota_feasible=allocated==batch,allocator='grouped_integer' if episode_keys is not None else 'max_flow',
        effective_weight={
            'source': {int(k): float(v)/float(batch) for k,v in counts.items()},
            'stage': {str(k): float(v)/float(batch) for k,v in actual_stage_counts.items()},
        },
        max_per_episode=max_per_episode,actual_max_per_episode=max(ep_counts.values()),unique_window_count=batch)
