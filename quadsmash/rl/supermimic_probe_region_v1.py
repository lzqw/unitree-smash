"""Re-evaluate current dynamics on immutable native paired-prefix evidence.

This is a bounded empirical validity region, not a physical safety guarantee.
No new optimizer step, native run, threshold fitting or implicit evidence renewal.
"""
import hashlib, math, json
from pathlib import Path
import torch
from .supermimic_ball_sequence_v1 import ball_transition
from .supermimic_physical_loop_loading_v1 import tensor_digest

STATIC=('execution_code_sha256','schema','task_contract_sha256','policy_execution')
METRICS=('native_repeat_max','native_response_min','response_rmse_max','response_cosine_min',
         'response_relative_error_max','final_normalized_command_delta_max','final_joint_target_delta_rad_max','accepted_update_age_max')

def contact_mode_groups(modes,count,head,timing=None):
    """Mode switches are finite differences, not a smooth Jacobian oracle."""
    required=('had_contact','released','frame_hit','string_hit') if head=='approach' else ('foot_contact','body_ground_seen')
    groups=[]
    for i in range(count):
        rows=[]
        for branch in ('nominal_a','nominal_b','minus','plus'):
            row=modes.get(branch) if isinstance(modes,dict) else None
            if isinstance(row,list):row=row[i] if len(row)==count else None
            elif count!=1:row=None
            if not isinstance(row,dict) or any(k not in row or row[k] is None for k in required):break
            rows.append({k:row[k] for k in required})
        if len(rows)!=4:groups.append('unknown')
        elif rows[0]!=rows[1]:groups.append('nominal-mode-mismatch')
        elif not all(row==rows[0] for row in rows):groups.append('mode-switch')
        elif timing is not None:
            timing_rows=[]
            for branch in ('nominal_a','nominal_b','minus','plus'):
                value=timing.get(branch) if isinstance(timing,dict) else None
                if isinstance(value,torch.Tensor):value=value[i].item()
                elif isinstance(value,list):value=value[i]
                if value is None:break
                timing_rows.append(int(value))
            if len(timing_rows)!=4:
                groups.append('unknown')
            elif timing_rows[0]!=timing_rows[1]:
                groups.append('nominal-timing-mismatch')
            elif len(set(timing_rows))!=1:
                groups.append('contact-time-switch')
            else:
                groups.append('smooth-mode')
        else:groups.append('smooth-mode')
    return groups

def load_region(spec,receipt_path):
    path=Path(receipt_path).parent/spec['path']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=spec['sha256']:raise ValueError('Native probe bundle hash mismatch')
    bundle=torch.load(path,map_location='cpu',weights_only=True)
    if bundle.get('schema')!='native-paired-prefix-region-v1':raise ValueError('Unknown native probe region schema')
    limits=spec['limits']
    for key in METRICS:
        if not isinstance(limits.get(key),(int,float)) or not math.isfinite(limits[key]):raise ValueError('Missing finite calibrated bound: '+key)
    if limits['native_response_min']<=0 or limits['native_repeat_max']<0 or not -1<=limits['response_cosine_min']<=1:
        raise ValueError('Invalid native probe bounds')
    if any(limits[k]<0 for k in METRICS if k.endswith('_max')):raise ValueError('Negative validity bound')
    calibration=spec.get('calibration')
    if not calibration:raise ValueError('Missing predeclared limit calibration provenance')
    raw=(Path(receipt_path).parent/calibration['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=calibration['sha256']:raise ValueError('Calibration provenance hash mismatch')
    evidence=json.loads(raw)
    if evidence.get('limits')!=limits or evidence.get('probe_sha256')!=spec['sha256']:
        raise ValueError('Bounds must match recorded independent calibration')
    if not evidence.get('heldout_episode_ids') or not evidence.get('calibration_episode_ids') or set(evidence['heldout_episode_ids'])&set(evidence['calibration_episode_ids']):
        raise ValueError('Independent calibration/holdout identity required')
    if not bundle.get('sample_ids') or not set(bundle['sample_ids'])<=set(evidence['heldout_episode_ids']):
        raise ValueError('Native probe samples not bound to heldout episode identities')
    if type(spec.get('minimum_samples')) is not int or spec['minimum_samples']<1:raise ValueError('Declare minimum native probe support')
    if len(bundle['pairs']['state'])<spec['minimum_samples']:raise ValueError('Insufficient native probe coverage')
    if len(set(bundle['sample_ids']))<spec['minimum_samples']:raise ValueError('Insufficient independent probe episodes')
    return bundle,limits

def paired_metrics(adapter,pairs,head,limits):
    """All commands here are measured FINAL commands, not requested perturbations.

    nominal_a/b are independent from-ground same-prefix replays. Branch start
    state and execution-memory hashes must match; a 55D snapshot is insufficient.
    """
    device=next(adapter.parameters()).device
    p={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in pairs.items()}
    required=('state','targets_minus','targets_plus','previous_action','feet','present','elapsed','launch',
              'native_minus','native_plus','native_nominal_a','native_nominal_b')
    if any(k not in p for k in required):raise ValueError('Incomplete measured native probe pairs')
    if not len(p['state']) or any(not torch.isfinite(p[k]).all() for k in required):raise ValueError('Empty/nonfinite native probe')
    hashes=p.get('prefix_sha256',{});mem=p.get('executor_before_sha256',{})
    for values in (hashes,mem):
        if set(values)!=set(('nominal_a','nominal_b','minus','plus')):
            raise ValueError('Same-prefix/controller repeatability missing; do not substitute snapshot restore')
        rows=[]
        for value in values.values():
            rows.append(value if isinstance(value,list) else [value]*len(p['state']))
        if any(len(row)!=len(p['state']) for row in rows) or any(not all(col) or len(set(col))!=1 for col in zip(*rows)):
            raise ValueError('Same-prefix/controller repeatability missing; do not substitute snapshot restore')
    if p.get('restore_mode')!='from-ground-identical-prefix':raise ValueError('Unsupported native restoration claim')
    temporal = all(k in p for k in (
        'temporal_horizon', 'step_targets_nominal_a',
        'step_targets_nominal_b', 'step_targets_minus', 'step_targets_plus',
        'step_final_actions_nominal_a', 'step_final_actions_nominal_b',
        'step_final_actions_minus', 'step_final_actions_plus'))
    if temporal:
        for key in ('step_targets_nominal_a','step_targets_nominal_b',
                    'step_targets_minus','step_targets_plus',
                    'step_final_actions_nominal_a','step_final_actions_nominal_b',
                    'step_final_actions_minus','step_final_actions_plus'):
            if not torch.isfinite(p[key]).all():
                raise ValueError('Nonfinite native temporal probe tensor: '+key)
    substep_summary = p.get('native_substep_summary')
    native_contact_timing = None
    if substep_summary is not None:
        if set(substep_summary) != set(('nominal_a', 'nominal_b', 'minus', 'plus')):
            raise ValueError('Incomplete native substep summary')
        timing = {branch: substep_summary[branch]['first_contact_substep'].to(device)
                  for branch in ('nominal_a', 'nominal_b', 'minus', 'plus')}
        native_contact_timing = {
            'first_contact_substep': {k: v.tolist() for k, v in timing.items()},
            'nominal_repeat_max': int((timing['nominal_a'] - timing['nominal_b']).abs().max()),
            'perturbation_span_max': int(torch.stack(tuple(timing.values())).amax(0).sub(
                torch.stack(tuple(timing.values())).amin(0)).max()),
            'physics_dt_s': float(substep_summary['nominal_a']['physics_dt_s']),
        }
    def rollout(targets, actions):
        state=p['state']; previous=p['previous_action']; feet=p['feet']
        present=p['present']; elapsed=p['elapsed']
        for i in range(targets.shape[1]):
            state,contact_prob,present=ball_transition(
                adapter.physical_model,adapter.ball_model,state,targets[:,i],
                previous,feet,present,elapsed,p['launch'],
                action_model=getattr(adapter,'action_ball_model',None))
            previous=actions[:,i]
            feet=contact_prob[:,:feet.shape[1]]
            elapsed=elapsed+elapsed.new_tensor(0.01)
        return state
    predictions=[]
    if temporal:
        with torch.no_grad():
            for name in ('minus','plus'):
                predictions.append(rollout(p['step_targets_'+name],p['step_final_actions_'+name]))
    else:
        with torch.no_grad():
            for name in ('targets_minus','targets_plus'):
                state,feet,_=ball_transition(adapter.physical_model,adapter.ball_model,p['state'],p[name],p['previous_action'],p['feet'],p['present'],p['elapsed'],p['launch'],action_model=getattr(adapter,'action_ball_model',None))
                predictions.append(state)
    selection=slice(52,55) if head=='approach' else slice(10,13)
    native=(p['native_plus']-p['native_minus'])[:,selection]
    predicted=(predictions[1]-predictions[0])[:,selection]
    repeat=float((p['native_nominal_a']-p['native_nominal_b']).abs().max())
    target_delta=(p['step_targets_plus'][:,0]-p['step_targets_minus'][:,0]).norm(dim=-1) if temporal else (p['targets_plus']-p['targets_minus']).norm(dim=-1)
    norms=native.norm(dim=-1);supported=(norms>=limits['native_response_min'])&(target_delta>0)
    missing=int((~supported).sum());failures=[]
    if native_contact_timing is not None and native_contact_timing['nominal_repeat_max'] > 0:
        failures.append('native-substep-nominal-timing-not-repeatable')
    timing_for_groups=None
    if substep_summary is not None:
        timing_for_groups={branch: substep_summary[branch]['first_contact_substep']
                           for branch in ('nominal_a','nominal_b','minus','plus')}
    groups=contact_mode_groups(p.get('native_contact_modes'),len(native),head,timing_for_groups)
    if 'unknown' in groups:failures.append('unknown-native-contact-mode')
    if 'nominal-mode-mismatch' in groups:failures.append('native-contact-mode-repeatability')
    if 'mode-switch' in groups:failures.append('non-smooth-contact-boundary-needs-separate-validation')
    if 'contact-time-switch' in groups:failures.append('contact-timing-switch-needs-separate-validation')
    if 'nominal-timing-mismatch' in groups:failures.append('native-substep-nominal-timing-not-repeatable')
    if repeat>limits['native_repeat_max']:failures.append('native-repeatability')
    if missing:failures.append('saturated-or-native-response-below-support')
    # No silent filtering of hard or saturated probes to inflate pass rate.
    cosine=(native*predicted).sum(-1)/(norms*predicted.norm(dim=-1)).clamp_min(1e-12)
    relative=(predicted-native).norm(dim=-1)/norms.clamp_min(1e-12)
    rmse=float((predicted-native).square().mean().sqrt())
    if rmse>limits['response_rmse_max']:failures.append('response-rmse')
    if float(cosine.min())<limits['response_cosine_min']:failures.append('response-direction')
    if float(relative.max())>limits['response_relative_error_max']:failures.append('response-amplitude')
    # True local model Jacobian projected onto measured executable perturbation.
    with torch.enable_grad():
        if temporal:
            # Differentiate the same complete executed-command interpolation
            # represented by the measured plus/minus rollout.  Using a
            # midpoint for only tick zero while replaying nominal commands
            # afterwards is a different perturbation when actuator memory
            # carries the action into later ticks.
            alpha=p['state'].new_zeros(len(p['state']),requires_grad=True)
            minus_targets=p['step_targets_minus']; plus_targets=p['step_targets_plus']
            minus_actions=p['step_final_actions_minus']; plus_actions=p['step_final_actions_plus']
            state=p['state']; previous=p['previous_action']; feet=p['feet']
            present=p['present']; elapsed=p['elapsed']
            for i in range(minus_targets.shape[1]):
                mix=alpha[:,None]
                current_target=minus_targets[:,i]+mix*(plus_targets[:,i]-minus_targets[:,i])
                current_action=minus_actions[:,i]+mix*(plus_actions[:,i]-minus_actions[:,i])
                state,contact_prob,present=ball_transition(
                    adapter.physical_model,adapter.ball_model,state,current_target,
                    previous,feet,present,elapsed,p['launch'],
                    action_model=getattr(adapter,'action_ball_model',None))
                previous=current_action
                feet=contact_prob[:,:feet.shape[1]]
                elapsed=elapsed+elapsed.new_tensor(0.01)
            grads=[torch.autograd.grad(state[:,selection][:,j].sum(),alpha,retain_graph=True,allow_unused=True)[0] for j in range(3)]
            jac=torch.stack([torch.zeros_like(alpha) if g is None else g for g in grads],1)
            command_delta=(p['step_targets_plus']-p['step_targets_minus']).reshape(len(state),-1).norm(dim=-1)
        else:
            target=((p['targets_plus']+p['targets_minus'])/2).detach().requires_grad_(True)
            y,_,_=ball_transition(adapter.physical_model,adapter.ball_model,p['state'],target,p['previous_action'],p['feet'],p['present'],p['elapsed'],p['launch'],action_model=getattr(adapter,'action_ball_model',None))
            grads=[torch.autograd.grad(y[:,selection][:,j].sum(),target,retain_graph=True,allow_unused=True)[0] for j in range(3)]
            jac=torch.stack([torch.zeros_like(target) if g is None else g for g in grads],1)
            command_delta=p['targets_plus']-p['targets_minus']
        projected=jac if temporal else (jac*command_delta[:,None]).sum(-1)
    if not torch.isfinite(projected).all():failures.append('nonfinite-jacobian')
    jac_cos=(projected*native).sum(-1)/(projected.norm(dim=-1)*norms).clamp_min(1e-12)
    smooth=torch.tensor([g=='smooth-mode' for g in groups],device=device)
    if smooth.any() and float(jac_cos[smooth].min())<limits['response_cosine_min']:failures.append('local-jacobian-direction')
    return dict(native_contact_modes=p.get('native_contact_modes','unknown'),passed=not failures,failures=failures,count=len(native),unsupported_count=missing,
        native_repeat_max=repeat,response_rmse=rmse,response_cosine=cosine.tolist(),relative_error=relative.tolist(),
        contact_mode_groups=groups,jacobian_cosine=[float(jac_cos[i].detach()) if groups[i]=='smooth-mode' else None for i in range(len(groups))],native_delta=native.tolist(),model_delta=predicted.tolist(),
        projected_jacobian=projected.detach().tolist(),temporal_horizon=int(p.get('temporal_horizon',1)),
        action_tick=int(p.get('action_tick',p.get('event_tick',0))),event_tick=int(p.get('event_tick',0)),
        native_contact_timing=native_contact_timing,
        scope='measured final-command pair on repeated native prefix; temporal rollout used when action precedes event; contact-mode switches and optional substep timing reported in raw evidence')

def assess_region(spec,receipt_path,current,adapter,command_preview,head):
    """command_preview(raw_current_start) restores a separate G07 branch each call."""
    bundle,limits=load_region(spec,receipt_path);origin=bundle['runtime_identity']
    if bundle.get('head')!=head:raise ValueError('Native probe belongs to different head')
    from .supermimic_evidence_binding_v1 import static_identity_failures, native_code_compatibility
    failures=static_identity_failures(origin,current)
    code_scope=native_code_compatibility(origin.get('execution_code_sha256'),current.get('execution_code_sha256'))
    age=current['policy_version']-origin['policy_version']
    if age<0 or age>limits['accepted_update_age_max']:failures.append('policy-age')
    if tensor_digest(bundle['actor_state'].items())!=origin['actor_state_sha256']:raise ValueError('Probe actor anchor identity mismatch')
    starts=bundle.get('command_starts',[])
    if not starts:raise ValueError('Missing full current-state/executor command drift library')
    normalized=targets=0.
    with torch.no_grad():
        for item in starts:
            final,target=command_preview(item['start'])
            mask=item.get('valid_lanes',torch.ones(len(final),dtype=torch.bool)).cpu().bool()
            if mask.shape!=(len(final),) or not mask.any():raise ValueError('Invalid native command comparison lane mask')
            normalized=max(normalized,float((final.cpu()-item['final_normalized'])[mask].abs().max()))
            targets=max(targets,float((target.cpu()-item['final_joint_targets'])[mask].abs().max()))
    if normalized>limits['final_normalized_command_delta_max']:failures.append('executed-normalized-command-drift')
    if targets>limits['final_joint_target_delta_rad_max']:failures.append('executed-joint-target-drift')
    measured=paired_metrics(adapter,bundle['pairs'],head,limits)
    failures+=measured['failures']
    return dict(passed=not failures,failures=failures,current_identity=current,evidence_sha256=spec['sha256'],
        origin_identity=origin,code_scope=code_scope,accepted_update_age=age,final_normalized_command_delta=normalized,
        final_joint_target_delta_rad=targets,metrics=measured,limits=limits,
        scope='current model rechecked against immutable native responses plus bounded policy drift; not an exact current-policy value guarantee')
