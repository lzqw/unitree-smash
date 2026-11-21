"""Pool causal Recovery starts without mixing G07 plans or per-record controller clocks."""
import torch


def recovery_batch(pools, target, attempt):
    """Round-robin stages and records, one lane per episode key per proposal."""
    if target < 1 or attempt < 1:
        raise ValueError('Positive batch size and proposal index required')
    candidates=[]
    for stage, rows in pools.items():
        if rows:
            shift=(attempt-1)%len(rows)
            candidates.append((stage,rows[shift:]+rows[:shift]))
    if not candidates:
        raise ValueError('No eligible Recovery episode')
    seen=set(); batches=[]; metadata=[]
    # Take one start per record before returning to fill the remaining capacity.
    for depth in range(max(len(items) for _,items in candidates)):
        for stage,items in candidates:
            if depth>=len(items):
                continue
            original,mask=items[depth]
            selected=torch.zeros_like(mask)
            episodes=original['before']['episode_index'].reshape(-1).tolist()
            ticks=original['before']['elapsed_tick'].reshape(-1).tolist()
            keys=[]
            lanes=torch.where(mask)[0].tolist()
            shift=(attempt-1)%max(len(lanes),1)
            for lane in lanes[shift:]+lanes[:shift]:
                key=(int(original.get('_lane_id',lane)),int(episodes[lane]))
                if key in seen:
                    continue
                selected[lane]=True;seen.add(key)
                keys.append([*key,int(ticks[lane])])
                # Divide a target across the stage families before filling from one dense record.
                if len(seen)==target or len(keys)>=max(1,(target+len(candidates)-1)//len(candidates)):
                    break
            if keys:
                batches.append(dict(original,_actor_lane_mask=selected))
                metadata.append(dict(stage=stage, lane_episode_tick=keys))
            if len(seen)==target:
                return batches,metadata
    # Fill unused capacity from denser records after sparse phases received a turn.
    for batch,entry in zip(batches,metadata,strict=True):
        for original,mask in pools[entry['stage']]:
            if batch['before'] is not original['before']:
                continue
            lanes=torch.where(mask)[0].tolist()
            shift=(attempt-1)%max(len(lanes),1)
            for lane in lanes[shift:]+lanes[:shift]:
                ep=int(original['before']['episode_index'][lane])
                key=(int(original.get('_lane_id',lane)),ep)
                if key in seen:
                    continue
                seen.add(key);batch['_actor_lane_mask'][lane]=True
                entry['lane_episode_tick'].append([*key,int(original['before']['elapsed_tick'][lane])])
                if len(seen)==target:
                    return batches,metadata
            break
    return batches,metadata


def _gradient_value(term, params):
    """Save a scalar and its first derivative while releasing the per-record BPTT graph."""
    if not torch.is_grad_enabled() or not term.requires_grad:
        return term.detach()
    derivatives=torch.autograd.grad(term,params,allow_unused=True,retain_graph=True)
    return term.detach()+sum(((p-p.detach())*g.detach()).sum() for p,g in zip(params,derivatives) if g is not None)


def pooled_objective(trainer, starts, effective_weights=None, baseline_toi_weights=None):
    toi = trainer.block_name=='approach' and getattr(trainer,'approach_objective','task')=='toi_racket_only'
    if not starts or not (toi or trainer.block_name=='recovery'):
        raise ValueError('Pooled starts require Recovery or TOI racket supervision')
    for group in (effective_weights,baseline_toi_weights):
        if group is not None and len(group)!=len(starts):
            raise ValueError('Backtracking weights must cover the same start records')
    toi_weights=[]
    preserved=True
    params=[p for p in trainer.actor.parameters() if p.requires_grad]
    terms={k:[] for k in ('task','tracking','anchor')}
    counts={};covered={};weights=[];returns=[];data=[];usage={};by_record=[]
    for i,state in enumerate(starts):
        kwargs=dict(effective_weights=None if effective_weights is None else effective_weights[i])
        if toi:
            kwargs['baseline_toi_weights']=None if baseline_toi_weights is None else baseline_toi_weights[i]
        row=trainer._rollout_objective(state,**kwargs)
        if toi:
            toi_weights.append(row['toi_weights'])
            preserved=preserved and row['toi_coverage_preserved']
        n=row['gate_lanes']; w=row['tracking_effective_weight_sum']
        for key in terms:
            terms[key].append((_gradient_value(row[key],params), w if key=='tracking' else n))
        for key in ('gate_steps','gate_lanes','tracking_steps','tracking_active_lane_steps','tracking_effective_weight_sum',
                    'tracking_nonzero_gradient_lanes','matched_references'):
            counts[key]=counts.get(key,0)+row[key]
        if 'outgoing_impact_lanes' in row:
            counts['outgoing_impact_lanes']=counts.get('outgoing_impact_lanes',0)+row['outgoing_impact_lanes']
        report=row['reference_coverage']
        by_record.append(dict(gate_lanes=n,matched=row['matched_references'],tracking_lane_steps=row['tracking_active_lane_steps']))
        if torch.is_grad_enabled():
            from .supermimic_approach_data_v1 import gradient_parts
            by_record[-1].update(gradient_parts({k:terms[k][-1][0] for k in terms},params))
        for key in ('selected_start_lanes','gate_active_selected_start_lanes'):
            covered[key]=covered.get(key,0)+report[key]
        for digest,count in report.get('reference_usage_by_sha256',{}).items():
            usage[digest]=usage.get(digest,0)+count
        weights.append(row['effective_weights'])
        returns.append(row['ret'][row['effective_weights'].bool()].detach())
        x,r,d,last,mask=row['critic_data']
        keep=mask.any(0)
        data.append((x[:,keep],r[:,keep],d[:,keep],last[keep],mask[:,keep]))
        del row
    # Pad the LEFT side only. Recursion from the genuine tail retains the correct GAE bootstrap.
    horizon=max(len(x[0]) for x in data)
    def join(k):
        if k==3:
            return torch.cat([v[k] for v in data])
        def pad(a):
            prefix=a.new_zeros((horizon-len(a),*a.shape[1:]))
            return torch.cat((prefix,a))
        return torch.cat([pad(v[k]) for v in data],dim=1)
    scalars={k:sum(x*n for x,n in items)/max(1.e-8,sum(n for _,n in items)) for k,items in terms.items()}
    loss=trainer.tracking_coefficient(trainer.block_name)*scalars['tracking']+trainer.anchor_coeff*scalars['anchor']
    if not toi:
        loss=loss+scalars['task']
    elif getattr(trainer,'contact_aux_weight',0.):
        loss=loss+trainer.contact_aux_weight*scalars['task']
    coverage=dict(report,**covered)
    coverage.update(matched_lanes=counts['matched_references'],effective_weight_sum=counts['tracking_effective_weight_sum'],
                    tracking_applied=bool(counts['tracking_active_lane_steps']),reference_usage_by_sha256=usage,per_record=by_record)
    ret=torch.cat(returns)
    return dict(**scalars,**counts, loss=loss,ret=ret if ret.numel() else loss.detach().reshape(1),
                effective_weights=weights,toi_weights=toi_weights if toi else None,toi_coverage_preserved=preserved,reference_coverage=coverage,
                critic_data=tuple(join(k) for k in range(5)),tracking_objective='toi_racket_only' if toi else 'recovery_reference')
