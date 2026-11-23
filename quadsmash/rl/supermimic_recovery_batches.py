"""Pool causal Recovery starts without mixing G07 plans or per-record controller clocks."""
import math

import torch
from .supermimic_physical_dataset_v1 import decode_physical
from .supermimic_physical_loop_replay_v1 import NATIVE_START_ID_FIELDS


def recovery_motion(physical):
    fields = decode_physical(physical)
    w, x, y, z = fields['root_quaternion_w'].unbind(-1)
    roll_sin = 2 * (w*x + y*z)
    roll_cos = 1 - 2 * (x*x + y*y)
    singular = (roll_sin.square() + roll_cos.square()) < 1e-12
    roll = torch.atan2(torch.where(singular, 0., roll_sin), torch.where(singular, 1., roll_cos))
    pitch = torch.asin((2 * (w*y - z*x)).clamp(-1 + 1e-6, 1 - 1e-6))
    tilt = torch.maximum(roll.abs(), pitch.abs())
    horizontal_speed = fields['root_link_velocity_w'][..., :2].norm(dim=-1)
    return tilt, horizontal_speed, fields['root_angular_velocity_w'], fields['joint_velocity']


def recovery_stability_cost(physical, limits):
    """Per-step normalized excess over the scored Recovery motion limits."""
    tilt, horizontal_speed, angular, joints = recovery_motion(physical)
    angular_speed = angular.norm(dim=-1)
    joint_speed = joints.abs().amax(dim=-1)
    ratios = torch.stack((
        tilt / math.radians(limits['maximum_abs_roll_pitch_deg']),
        horizontal_speed / limits['maximum_horizontal_root_speed_m_s'],
        angular_speed / limits['maximum_root_angular_speed_rad_s'],
        joint_speed / limits['maximum_abs_joint_velocity_rad_s'],
    ), dim=-1)
    return (ratios - 1).relu().mean(dim=-1)


def relaxed_recovery_step(physical, feet, limits, run, dt, *, active):
    tilt, horizontal, angular, joints = recovery_motion(physical)
    support = feet.clamp(0, 1).sum(-1)
    ratios = torch.stack((
        tilt / math.radians(limits['maximum_abs_roll_pitch_deg']),
        angular[..., :2].norm(dim=-1) / limits['maximum_roll_pitch_speed_rad_s'],
        horizontal / limits['maximum_horizontal_root_speed_m_s'],
        joints[..., :12].abs().amax(-1) / limits['maximum_abs_leg_joint_velocity_rad_s'],
    ), -1) / .8
    support_deficit = (limits['minimum_support_feet'] - support).relu()
    cost = (ratios - 1).relu().square().sum(-1) + support_deficit.square()
    cost = cost + (.18 - physical[..., 2]).relu() + .01 * joints[..., 12:].square().mean(-1)
    stable = torch.sigmoid((1 - ratios) / .02).prod(-1)
    stable = stable * torch.sigmoid((support - limits['minimum_support_feet'] + .5) / .05)
    next_run = torch.where(active, stable * (run + dt), 0.)
    window = limits['continuous_stable_window_s']
    progress = (next_run.clamp(max=window) - run.clamp(max=window)) / window
    return cost, next_run, progress


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
                native_origin = {key: original[key] for key in NATIVE_START_ID_FIELDS if key in original}
                metadata.append(dict(stage=stage, lane_episode_tick=keys, native_origin=native_origin or None))
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


def late_recovery_batch(pools, target, attempt, *, relaxed=False):
    # Quotas operate on unique native lane/episodes across all families.
    groups = (('risk', ('pre_body_ground', 'post_landing_support_loss', 'recovery_failure'), target//2),
              ('landing', ('touchdown', 'post_touchdown'), target//4),
              ('stable', ('stable_recovery',), target-target//2-target//4))
    pools = dict(pools)
    if relaxed:
        landing_count, risk_count = round(.60 * target), round(.25 * target)
        groups = (('landing', ('touchdown', 'post_touchdown'), landing_count),
                  ('risk', ('pre_body_ground',), risk_count),
                  ('stable', ('stable_recovery',), target - landing_count - risk_count))
        for name in ('post_touchdown', 'stable_recovery'):
            filtered = []
            for record, mask in pools.get(name, ()):
                before = record['before']
                landing = before['auxiliary']['contact_recovery']['landing_time'].reshape(-1)
                elapsed = before['physical']['elapsed_s'].reshape(-1)
                valid = mask & (landing > 0) & (elapsed - landing <= (.30 if name == 'post_touchdown' else 2.))
                if valid.any():
                    filtered.append((record, valid))
            pools[name] = filtered
    # A failure can contribute three lead times but only one start per proposal.
    # Rotate the chosen lead across episodes and draws rather than always taking 20.
    prefixes = pools.get('pre_body_ground', [])
    preferred = {}
    for idx, (record, mask) in enumerate(prefixes):
        for lane in torch.where(mask)[0].tolist():
            key = (int(record.get('_lane_id', lane)), int(record['before']['episode_index'][lane]))
            desired = 20 * (1 + (key[0] + key[1] + attempt - 1) % 3)
            candidate = (abs(record.get('_failure_lead_ticks', 20) - desired), idx, lane)
            if key not in preferred or candidate < preferred[key]:
                preferred[key] = candidate
    masks = [torch.zeros_like(mask) for _, mask in prefixes]
    for _, idx, lane in preferred.values():
        masks[idx][lane] = True
    pools['pre_body_ground'] = [(record, mask) for (record, _), mask in zip(prefixes, masks) if mask.any()]
    seen, rows, layout = set(), [], []

    def draw(names, count):
        available = {}
        for name in names:
            for record, mask in pools.get(name, []):
                mask = mask.clone()
                for lane in torch.where(mask)[0].tolist():
                    if (int(record.get('_lane_id', lane)), int(record['before']['episode_index'][lane])) in seen:
                        mask[lane] = False
                if mask.any():
                    available.setdefault(name, []).append((record, mask))
        if not available or count <= 0:
            return
        selected, details = recovery_batch(available, count, attempt)
        rows.extend(selected)
        layout.extend(details)
        for record, group in zip(selected, details, strict=True):
            before = record['before']
            group['samples'] = []
            for lane, ep, tick in group['lane_episode_tick']:
                seen.add((lane, ep))
                i = 0 if '_lane_id' in record else lane
                jump = before['auxiliary']['jump']
                landing = float(before['auxiliary']['contact_recovery']['landing_time'][i])
                source = 'first_qualified_landing'
                if landing <= 0:
                    source = 'latest_observed_jump_landing'
                    landing = float(jump['landing_time'][i])
                    task = before.get('task', {})
                    if 'contact_time' in task and not landing >= float(task['contact_time'][i]):
                        landing = 0.
                phys = before['physical']
                q = phys['root_quaternion_w'][i]
                group['samples'].append(dict(lane=lane, episode=ep, tick=tick,
                    since_landing_s=float(phys['elapsed_s'][i])-landing if landing > 0 else None,
                    landing_clock_source=source if landing > 0 else None,
                    support_feet=int(phys['foot_contact'][i].sum()),
                    tilt_rad=float(torch.acos((1-2*q[1:3].square().sum()).clamp(-1,1))),
                    failure_window_ticks=record.get('_failure_lead_ticks')))

    for _, names, quota in groups:
        draw(names, quota)
    if len(seen) < target:
        draw(pools.keys(), target-len(seen))
    if not rows:
        raise ValueError('No eligible Recovery episode')
    return rows, layout


def _gradient_value(term, params, derivatives=None):
    """Save a scalar and its first derivative while releasing the per-record BPTT graph."""
    if not torch.is_grad_enabled() or not term.requires_grad:
        return term.detach()
    if derivatives is None:
        derivatives=torch.autograd.grad(term,params,allow_unused=True,retain_graph=True)
    return term.detach()+sum(((p-p.detach())*g.detach()).sum() for p,g in zip(params,derivatives) if g is not None)


def update_recovery_landing(landed, previous_task, next_feet):
    """Carry native post-contact landing into the rollout via the same support event."""
    rally = previous_task['rally']
    eligible = (rally['had_contact'].reshape(-1) > .5) & (rally['released'].reshape(-1) > .5)
    in_flight = previous_task['auxiliary']['jump']['flight_candidate'].reshape(-1) > .5
    return landed | (eligible & in_flight & (next_feet > .5).any(-1))


def pooled_objective(trainer, starts, effective_weights=None, baseline_toi_weights=None, baseline_landing_weights=None, *, collect_critic_data=True):
    toi = trainer.block_name=='approach' and getattr(trainer,'approach_objective','task')=='toi_racket_only'
    if not starts or not (toi or trainer.block_name=='recovery'):
        raise ValueError('Pooled starts require Recovery or TOI racket supervision')
    for group in (effective_weights,baseline_toi_weights,baseline_landing_weights):
        if group is not None and len(group)!=len(starts):
            raise ValueError('Backtracking weights must cover the same start records')
    toi_weights=[]
    landing_weights=[]
    preserved=True
    params=[p for p in trainer.actor.parameters() if p.requires_grad]
    terms={k:[] for k in ('task','tracking','anchor')}
    counts={};covered={};weights=[];returns=[];data=[];usage={};by_record=[]
    for i,state in enumerate(starts):
        kwargs=dict(effective_weights=None if effective_weights is None else effective_weights[i])
        if not collect_critic_data:
            kwargs['collect_critic_data'] = False
        if baseline_landing_weights is not None:
            kwargs['baseline_landing_weights'] = baseline_landing_weights[i]
        if toi:
            kwargs['baseline_toi_weights']=None if baseline_toi_weights is None else baseline_toi_weights[i]
        if torch.is_grad_enabled():
            kwargs['tracking_parameters'] = params
        row=trainer._rollout_objective(state,**kwargs)
        if toi:
            toi_weights.append(row['toi_weights'])
            preserved=preserved and row['toi_coverage_preserved']
        if row.get('landing_weights') is not None:
            landing_weights.append(row['landing_weights'])
        n=row['gate_lanes']; w=row['tracking_effective_weight_sum']
        for key in terms:
            derivatives = row.get('tracking_derivatives') if key == 'tracking' else None
            terms[key].append((_gradient_value(row[key],params,derivatives), w if key=='tracking' else n))
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
        for key in ('predicted_first_string_lanes','predicted_first_frame_lanes','predicted_first_margin_sum_m',
                    'newly_landed_lanes','newly_landed_lane_steps'):
            if key in report:
                covered[key]=covered.get(key,0)+report[key]
        for key in ('selected_start_lanes','gate_active_selected_start_lanes'):
            covered[key]=covered.get(key,0)+report[key]
        for digest,count in report.get('reference_usage_by_sha256',{}).items():
            usage[digest]=usage.get(digest,0)+count
        weights.append(row['effective_weights'])
        returns.append(row['ret'][row['effective_weights'].bool()].detach())
        if collect_critic_data:
            x,r,d,last,mask=row['critic_data']
            keep=mask.any(0)
            data.append((x[:,keep],r[:,keep],d[:,keep],last[keep],mask[:,keep]))
        del row
    # Pad the LEFT side only. Recursion from the genuine tail retains the correct GAE bootstrap.
    horizon=max((len(x[0]) for x in data), default=0)
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
                effective_weights=weights,landing_weights=landing_weights or None,toi_weights=toi_weights if toi else None,toi_coverage_preserved=preserved,reference_coverage=coverage,
                critic_data=tuple(join(k) for k in range(5)) if collect_critic_data else None,tracking_objective='toi_racket_only' if toi else 'recovery_reference')
