"""Held-out recorded-command autoregression; no native simulation or updates."""
import torch
from .supermimic_ball_sequence_v1 import ball_transition
from .supermimic_physical_dataset_v1 import decode_physical
from .supermimic_physical_dataset_v1 import valid_windows
from .supermimic_event_hazard_v1 import advance_event_latches

def event_aligned_windows(data,lanes,*,horizon=64,offsets=(-4,-1,0,1,4),maximum=128):
    """Offsets use the native transition that first creates contact/frame latch.

    Offset zero starts immediately before that transition; labels are offline
    selection only. No interpolation of PhysX substeps is performed.
    """
    valid=valid_windows(data['episode'],data['terminated'],data['timeout'],horizon)
    if 'valid_lengths' in data:valid &= torch.arange(len(valid))[:,None]+horizon<=data['valid_lengths'][None]
    allowed=torch.zeros(valid.shape[1],dtype=torch.bool);allowed[list(lanes)]=True
    before,after=data['event_state'],data['next_event_state']
    contacts=((after[:,:,1]>before[:,:,1])|(after[:,:,6]>before[:,:,6]))&allowed[None]
    events=contacts.nonzero();result={}
    for offset in offsets:
        rows=[];ticks=[]
        for tick,lane in events.tolist():
            start=tick+offset
            if start<0 or start>=len(valid) or not valid[start,lane]:continue
            if data['episode'][start,lane]!=data['episode'][tick,lane]:continue
            if data['terminated'][min(start,tick):max(start,tick),lane].any() or data['timeout'][min(start,tick):max(start,tick),lane].any():continue
            rows.append((start,lane));ticks.append(tick)
            if len(rows)>=maximum:break
        result[str(offset)]=dict(starts=torch.tensor(rows,dtype=torch.long).reshape(-1,2),event_ticks=ticks,
                                offset_control_ticks=offset,source='native qualified-contact/frame latch edge, not substep timestamp')
    return result

@torch.no_grad()
def validate_windows(adapter,window,launch,horizons=(1,4,16,32,64)):
    if len(window['state'])<max(horizons):raise ValueError('Validation window too short')
    state=window['state'][0];feet=window['feet'][0].float();present=window['present'][0]
    values,valid=window['values'][0],window['valid'][0];events=window['event_state'][0].float()
    result={}; predicted_first=torch.full((len(state),),-1,device=state.device,dtype=torch.long)
    true_first=predicted_first.clone()
    for t in range(max(horizons)):
        next_state,newfeet,present=ball_transition(adapter.physical_model,adapter.ball_model,state,window['targets'][t],
            window['previous_action'][t],feet,present,window['elapsed'][t],launch)
        task=adapter.task_model.forward_values(state,next_state,window['targets'][t],window['previous_action'][t],feet,
            window['elapsed'][t],values,valid)
        events=advance_event_latches(events,task['hazards'])
        ptd=(~(feet>.5).any(-1))&(newfeet>.5).any(-1)&(events[:,0]>.5)
        ttd=(~window['feet'][t].bool().any(-1))&window['next_feet'][t].bool().any(-1)&(window['event_state'][t,:,0]>.5)
        predicted_first=torch.where((predicted_first<0)&ptd,t+1,predicted_first)
        true_first=torch.where((true_first<0)&ttd,t+1,true_first)
        state,feet=next_state,newfeet;values=task['numeric'];valid=task['valid_logits'].sigmoid()
        if t+1 in horizons:
            target=window['next_state'][t];delta=state-target
            q=torch.nn.functional.normalize(state[:,3:7],dim=-1);qt=torch.nn.functional.normalize(target[:,3:7],dim=-1)
            body=events[:,5];truth=window['next_event_state'][t,:,5].float()
            matched=(predicted_first>=0)&(true_first>=0)
            metrics=dict(state_rmse=float(delta.square().mean().sqrt()),root_position_rmse=float(delta[:,:3].square().mean().sqrt()),
                orientation_rad=float((2*torch.acos((q*qt).sum(-1).abs().clamp(0,1))).mean()),
                angular_velocity_rmse=float(delta[:,10:13].square().mean().sqrt()),joint_velocity_rmse=float(delta[:,31:49].square().mean().sqrt()),
                ball_position_rmse=float(delta[:,49:52].square().mean().sqrt()),ball_velocity_rmse=float(delta[:,52:55].square().mean().sqrt()),
                ball_forward_sign_error=float(((state[:,52]>0)!=(target[:,52]>0)).float().mean()),
                ball_forward_sign_accuracy=float(((state[:,52]>0)==(target[:,52]>0)).float().mean()),
                foot_contact_brier=float((feet-window['next_feet'][t].float()).square().mean()),
                body_ground_brier=float((body-truth).square().mean()),task_event_brier=float((events-window['next_event_state'][t]).square().mean()),
                touchdown_matched=int(matched.sum()),touchdown_presence_mismatch=int(((predicted_first>=0)!=(true_first>=0)).sum()),
                touchdown_timing_mae_s=float((predicted_first[matched]-true_first[matched]).abs().float().mean()*.01) if matched.any() else None)
            for label,mask in (('positive',truth>.5),('negative',truth<.5)):
                metrics['body_'+label+'_count']=int(mask.sum())
                metrics['body_'+label+'_brier']=float((body[mask]-truth[mask]).square().mean()) if mask.any() else None
            geometry=getattr(adapter.physical_model,'geometry',None)
            if geometry is not None:
                def fk(s):
                    p=decode_physical(s)
                    return geometry(p['joint_position'],p['joint_velocity'],p['root_position_w'],p['root_quaternion_w'],
                                    p['root_link_velocity_w'],p['root_angular_velocity_w'])
                pred_fk,true_fk=fk(state),fk(target);rid=geometry.ids['Racket']
                metrics['racket_position_rmse']=float((pred_fk['position'][:,rid]-true_fk['position'][:,rid]).square().mean().sqrt())
                metrics['racket_velocity_rmse']=float((pred_fk['velocity'][:,rid]-true_fk['velocity'][:,rid]).square().mean().sqrt())
                dot=(pred_fk['rotation'][:,rid,:,0]*true_fk['rotation'][:,rid,:,0]).sum(-1).clamp(-1,1)
                metrics['racket_normal_error_rad']=float(torch.acos(dot).mean())
            else:
                metrics['missing_fields']=['racket_kinematics']
            if not all(torch.isfinite(x).all() for x in (state,feet,events)):raise FloatingPointError('Nonfinite validation rollout')
            result[str(t+1)]=metrics
    return result

def validation_stages(window):
    """Overlapping native-label groups for reporting, never rollout inputs."""
    before,after=window['event_state'],window['next_event_state']
    feet=window['feet'].bool().any(-1);next_feet=window['next_feet'].bool().any(-1)
    return dict(all=torch.ones(before.shape[1],dtype=torch.bool,device=before.device),
        contact=(after[:,:,1]>before[:,:,1]).any(0),
        touchdown=((~feet)&next_feet&(before[:,:,0]>.5)).any(0),
        recovery_failure=(after[:,:,5]>before[:,:,5]).any(0),
        stable_recovery=(after[:,:,4]>before[:,:,4]).any(0),
        contact_ball_side_pending=((after[:,:,1]>.5)&((after[:,:,2]<.5)|(after[:,:,3]<.5))).any(0))

@torch.no_grad()
def command_sensitivity(adapter,window,launch,*,epsilon=.001,joints=(12,13,14,15,16,17)):
    """Fixed perturbation model diagnostic; NOT a PhysX counterfactual."""
    rows=[]
    for joint in joints:
        predictions=[]
        for sign in (-1.,1.):
            targets=window['targets'][0].clone();targets[:,joint]+=sign*epsilon
            state,feet,present=ball_transition(adapter.physical_model,adapter.ball_model,window['state'][0],targets,
                window['previous_action'][0],window['feet'][0].float(),window['present'][0],window['elapsed'][0],launch)
            predictions.append(state)
        fd=(predictions[1]-predictions[0])/(2*epsilon)
        rows.append(dict(joint=int(joint),robot_derivative_norm=float(fd[:,:49].norm()),ball_derivative_norm=float(fd[:,49:].norm()),
                         scope='model only; perturbs final targets; native slew/contact-cache replay not performed'))
    return rows

@torch.no_grad()
def runtime_contact_check(adapter,data,lanes,launch,upper_bounds,*,maximum=16):
    """Fixed event-aligned holdout check; caller excludes these lanes from WM fit."""
    pairs=event_aligned_windows(data,lanes,maximum=maximum)['-1']['starts']
    if not len(pairs):raise ValueError('No held-out event-aligned contact windows; do not substitute standing loss')
    t,ids=pairs.unbind(1);indices=t[None]+torch.arange(64)[:,None]
    device=next(adapter.parameters()).device
    window={k:v[indices,ids[None]].to(device) for k,v in data.items() if v.ndim>=2 and v.shape[:2]==data['episode'].shape}
    results=validate_windows(adapter,window,launch[ids.to(launch.device)],horizons=(1,4,16,64))
    failures={}
    for path,limit in upper_bounds.items():
        horizon,key=path.split('/',1);value=results.get(horizon,{}).get(key)
        if value is None or not isinstance(value,(int,float)) or not torch.isfinite(torch.tensor(value)) or value>limit:
            failures[path]=dict(value=value,upper_bound=limit)
    return dict(errors=results,failures=failures,passed=not failures,window_starts=pairs.tolist(),
                scope='event-aligned held-out recorded-command prediction; not native policy score')


def recovery_event_windows(data,lanes,*,horizon=64,maximum=16):
    """Recovery event origins, using observed jump/contact and native foot edges."""
    valid=valid_windows(data['episode'],data['terminated'],data['timeout'],horizon)
    if 'valid_lengths' in data:valid &= torch.arange(len(valid))[:,None]+horizon<=data['valid_lengths'][None]
    allowed=torch.zeros(valid.shape[1],dtype=torch.bool);allowed[list(lanes)]=True
    event=data['event_state'];nxt=data['next_event_state'];feet=data['feet'].bool().any(-1);after=data['next_feet'].bool().any(-1)
    posthit=(event[:,:,1]>.5)&(event[:,:,0]>.5)
    groups={'touchdown':(~feet)&after&posthit,'support_loss':feet&(~after)&posthit,
            'body_ground':(nxt[:,:,5]>event[:,:,5])&posthit,'stable':(nxt[:,:,4]>event[:,:,4])&posthit}
    result={}
    for name,mask in groups.items():
        rows=[]
        for tick,lane in (mask&allowed[None]).nonzero().tolist():
            start=max(0,tick-1)
            if start>=len(valid) or not valid[start,lane]:continue
            rows.append((start,lane))
            if len(rows)>=maximum:break
        result[name]=torch.tensor(rows,dtype=torch.long).reshape(-1,2)
    return result


def independent_event_support(data,pairs):
    identities=set()
    for tick,lane in pairs.tolist():
        stream=int(data['stream_id'][tick,lane]) if 'stream_id' in data else None
        identities.add((stream,lane,int(data['episode'][tick,lane])))
    return len(identities)


def approach_event_windows(data,lanes,maximum=16):
    aligned=event_aligned_windows(data,lanes,offsets=(-1,),maximum=max(128,maximum))['-1']
    groups={k:[] for k in ('legal_contact','illegal_contact','ball_side_success','ball_side_failure')}
    before,after=data['event_state'],data['next_event_state']
    for (start,lane),tick in zip(aligned['starts'].tolist(),aligned['event_ticks']):
        if after[tick,lane,1]>before[tick,lane,1]:groups['legal_contact'].append((start,lane))
        if after[tick,lane,6]>before[tick,lane,6]:groups['illegal_contact'].append((start,lane))
        # Only explicit native terminal scores may establish these outcomes.
        # They are offline window labels, never forwarded to the model/actor.
        if 'ball_side_terminal' in data:
            ep=data['episode'][start,lane]
            mask=(data['episode'][:,lane]==ep)&(data['terminated'][:,lane]|data['timeout'][:,lane])
            labels=data['ball_side_terminal'][:,lane][mask]
            if len(labels)==1 and torch.isfinite(labels[0]):
                groups['ball_side_success' if labels[0]>.5 else 'ball_side_failure'].append((start,lane))
    # Diverse episodes first: repeated edges in one episode are not independent evidence.
    result={}
    for name,rows in groups.items():
        selected=[];seen=set()
        for start,lane in rows:
            identity=(lane,int(data['episode'][start,lane]))
            if identity in seen:continue
            seen.add(identity);selected.append((start,lane))
            if len(selected)>=maximum:break
        result[name]=torch.tensor(selected,dtype=torch.long).reshape(-1,2)
    return result


@torch.no_grad()
def runtime_head_check(adapter,data,lanes,launch,head,upper_bounds,*,maximum=16,minimum_event_episodes=None):
    from .supermimic_dual_head_quality_v1 import compare_bounds,REQUIRED_EVENTS
    if head=='approach':groups=approach_event_windows(data,lanes,maximum=maximum)
    elif head=='recovery':groups=recovery_event_windows(data,lanes,maximum=maximum)
    else:raise ValueError('Unknown WM validation head')
    # Default one episode is a coverage floor, not a calibrated reliability claim.
    minimum_event_episodes=minimum_event_episodes or {e:1 for e in REQUIRED_EVENTS[head]}
    measured={};failures={};starts={};support={}
    for name,pairs in groups.items():
        starts[name]=pairs.tolist();support[name]=independent_event_support(data,pairs)
        minimum=minimum_event_episodes.get(name)
        if type(minimum) is not int or minimum<1 or support[name]<minimum:
            failures['support/'+name]=dict(independent_episodes=support[name],minimum=minimum)
        if not len(pairs):continue
        t,ids=pairs.unbind(1);at=t[None]+torch.arange(64)[:,None];device=next(adapter.parameters()).device
        window={k:v[at,ids[None]].to(device) for k,v in data.items() if isinstance(v,torch.Tensor) and v.ndim>=2 and v.shape[:2]==data['episode'].shape}
        result=validate_windows(adapter,window,launch[ids.to(launch.device)],horizons=(1,4,16,32,64))
        measured[name]=result
        flat={h+'/'+k:v for h,m in result.items() for k,v in m.items()}
        failed=compare_bounds(flat,upper_bounds)
        if failed:failures[name]=failed
    return dict(head=head,passed=not failures,errors=measured,failures=failures,window_starts=starts,
                independent_event_episodes=support,minimum_event_episodes=minimum_event_episodes,
                missing_events=[k for k,v in starts.items() if not v],
                scope='held-out recorded-command prediction; native action-response evidence is separate',
                contact_source='native qualified-contact/frame/foot flags; other illegal contact modes unverified')
