"""Held-out recorded-command autoregression; no native simulation or updates."""
import torch
from .supermimic_ball_sequence_v1 import ball_transition
from .supermimic_physical_dataset_v1 import decode_physical
from .supermimic_event_hazard_v1 import advance_event_latches

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
