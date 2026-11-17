"""Per-head quality contracts and independent native value diagnostics.

Thresholds must be supplied from calibration evidence. This module does not
run calibration experiments or treat model metrics as task success.
"""
import math
import torch

HEADS=('recovery','approach')


def quality_spec(receipt):
    q=receipt.get('dual_head_quality')
    if not isinstance(q,dict):raise ValueError('Missing calibrated dual_head_quality; contact-only gate cannot certify Recovery')
    lanes=q.get('heldout_lane_ids',[])
    if not lanes or any(type(i) is not int or not 0<=i<128 for i in lanes) or len(set(lanes))!=len(lanes) or len(lanes)>=128:
        raise ValueError('Invalid whole-lane holdout split')
    for head in HEADS:
        probe=q.get('native_action_probes',{}).get(head,{})
        if probe.get('same_prefix_verified') is not True or probe.get('paired_samples',0)<1:
            raise ValueError('Missing native action-response evidence for '+head)
        errors=probe.get('errors',{});limits=probe.get('upper_bounds',{})
        if not errors or errors.keys()!=limits.keys():raise ValueError('Missing action-response bounds for '+head)
        if compare_bounds(errors,limits):raise ValueError('Native action-response bounds failed for '+head)
        wm=q.get('wm_upper_bounds',{}).get(head,{})
        critic=q.get('critic_upper_bounds',{}).get(head,{})
        required=('ball_velocity_rmse','ball_forward_sign_error') if head=='approach' else ('angular_velocity_rmse','foot_contact_brier','body_ground_brier')
        if not all(any(k.endswith('/'+m) for k in wm) for m in required):raise ValueError('Missing head-specific WM bounds for '+head)
        if not {'mae','rmse'}<=critic.keys():raise ValueError('Missing native critic bounds for '+head)
        for name,value in dict(wm,**critic).items():
            if not isinstance(value,(int,float)) or not math.isfinite(value) or value<0 or any(x in name for x in ('accuracy','_count','_matched')):
                raise ValueError('Invalid error upper bound: '+name)
    return q


def compare_bounds(metrics,limits):
    failures={}
    for name,limit in limits.items():
        value=metrics.get(name)
        if (not isinstance(value,(int,float)) or not math.isfinite(value) or value<0
                or not isinstance(limit,(int,float)) or not math.isfinite(limit) or limit<0 or value>limit):
            failures[name]=dict(value=value,upper_bound=limit)
    return failures


def reference_coverage(bank,head,weight):
    entries=[] if bank is None else [e for e in bank.entries if e['mode']==head]
    verified=[e for e in entries if e.get('terminal_verified') is True and e.get('score_schema')=='historical-independent-complete5-v1']
    enabled=bool(weight>0 and verified)
    return dict(head=head,reference_count=len(entries),verified_reference_count=len(verified),tracking_weight=float(weight),
                success_guided_available=enabled,mode='success-guided' if enabled else 'task-bptt-only',
                reason=None if enabled else ('tracking-disabled' if weight<=0 else 'no-verified-head-reference'))


def split_native_value_rows(bank,holdout_lanes):
    identities=bank['identity']
    if not bank.get('policy_id') or not bank.get('actor_sha256'):raise ValueError('Missing frozen policy identity')
    n=len(bank['rtg'])
    if any(len(bank['head_masks'][h])!=n for h in HEADS):raise ValueError('Native head mask alignment mismatch')
    if not torch.isfinite(bank['inputs']).all() or not torch.isfinite(bank['rtg']).all():raise ValueError('Nonfinite native value data')
    if len(identities)!=n or len(bank['inputs'])!=n:raise ValueError('Native value row alignment mismatch')
    if len({tuple(x) for x in identities})!=n:raise ValueError('Duplicate native value identity')
    # identity = (lane, episode, tick), all under the bank's frozen policy_id.
    val=torch.tensor([int(x[0]) in holdout_lanes for x in identities],dtype=torch.bool)
    if not val.any() or val.all():raise ValueError('Native value split needs disjoint train and validation episodes')
    return ~val,val


@torch.no_grad()
def critic_holdout(critic,bank,head,holdout_lanes,*,current_actor_sha256,current_execution=None,limits=None):
    _,val=split_native_value_rows(bank,holdout_lanes)
    mask=val&bank['head_masks'][head].cpu().bool()
    ids=torch.where(mask)[0]
    base=dict(head=head,policy_id=bank['policy_id'],label_actor_sha256=bank['actor_sha256'],
              current_policy_matches_labels=(current_actor_sha256==bank['actor_sha256'] and current_execution is not None and bank.get('policy_execution')==current_execution),
              label_policy_execution=bank.get('policy_execution'),label_execution_known=bank.get('policy_execution') is not None,sample_count=len(ids),
              scope='frozen-label-policy value calibration; not updated-policy exact value',
              identity=[bank['identity'][i] for i in ids.tolist()])
    if not len(ids):return dict(base,passed=False,missing_fields=['head-specific-complete-native-validation-states'])
    device=next(critic.parameters()).device
    pred=torch.cat([critic(bank['inputs'][part].to(device)).detach().cpu().reshape(-1) for part in ids.split(128)])
    truth=bank['rtg'][ids].cpu().reshape(-1);error=pred-truth
    if not torch.isfinite(error).all():raise FloatingPointError('Nonfinite native value validation')
    metrics=dict(mae=float(error.abs().mean()),rmse=float(error.square().mean().sqrt()),bias=float(error.mean()))
    concordant=pairs=0
    # Compare only equal start ticks; never rank 0.64s returns against full episodes.
    times=torch.tensor([bank['identity'][i][2] for i in ids.tolist()])
    for tick in times.unique():
        p=pred[times==tick];y=truth[times==tick]
        difference=y[:,None]-y[None];selected=torch.triu(torch.ones_like(difference,dtype=torch.bool),diagonal=1)&(difference!=0)
        pairs+=int(selected.sum());concordant+=int(((p[:,None]-p[None])*difference>0)[selected].sum())
    metrics['rtg_ordering_accuracy']=concordant/pairs if pairs else None
    metrics['rtg_ordering_pairs']=pairs
    failures=compare_bounds(metrics,limits) if limits is not None else {}
    return dict(base,metrics=metrics,failures=failures,passed=not failures if limits is not None else None,
                protection_enabled=limits is not None)
