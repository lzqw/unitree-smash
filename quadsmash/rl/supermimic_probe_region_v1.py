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
        if set(values)!=set(('nominal_a','nominal_b','minus','plus')) or any(not x for x in values.values()) or len(set(values.values()))!=1:
            raise ValueError('Same-prefix/controller repeatability missing; do not substitute snapshot restore')
    if p.get('restore_mode')!='from-ground-identical-prefix':raise ValueError('Unsupported native restoration claim')
    predictions=[]
    with torch.no_grad():
        for name in ('targets_minus','targets_plus'):
            state,feet,_=ball_transition(adapter.physical_model,adapter.ball_model,p['state'],p[name],p['previous_action'],p['feet'],p['present'],p['elapsed'],p['launch'],action_model=getattr(adapter,'action_ball_model',None))
            predictions.append(state)
    selection=slice(52,55) if head=='approach' else slice(10,13)
    native=(p['native_plus']-p['native_minus'])[:,selection]
    predicted=(predictions[1]-predictions[0])[:,selection]
    repeat=float((p['native_nominal_a']-p['native_nominal_b']).abs().max())
    target_delta=(p['targets_plus']-p['targets_minus']).norm(dim=-1)
    norms=native.norm(dim=-1);supported=(norms>=limits['native_response_min'])&(target_delta>0)
    missing=int((~supported).sum());failures=[]
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
        target=((p['targets_plus']+p['targets_minus'])/2).detach().requires_grad_(True)
        y,_,_=ball_transition(adapter.physical_model,adapter.ball_model,p['state'],target,p['previous_action'],p['feet'],p['present'],p['elapsed'],p['launch'],action_model=getattr(adapter,'action_ball_model',None))
        grads=[torch.autograd.grad(y[:,selection][:,j].sum(),target,retain_graph=True,allow_unused=True)[0] for j in range(3)]
        jac=torch.stack([torch.zeros_like(target) if g is None else g for g in grads],1)
        projected=(jac*(p['targets_plus']-p['targets_minus'])[:,None]).sum(-1)
    if not torch.isfinite(projected).all():failures.append('nonfinite-jacobian')
    jac_cos=(projected*native).sum(-1)/(projected.norm(dim=-1)*norms).clamp_min(1e-12)
    if float(jac_cos.min())<limits['response_cosine_min']:failures.append('local-jacobian-direction')
    return dict(native_contact_modes=p.get('native_contact_modes','unknown'),passed=not failures,failures=failures,count=len(native),unsupported_count=missing,
        native_repeat_max=repeat,response_rmse=rmse,response_cosine=cosine.tolist(),relative_error=relative.tolist(),
        jacobian_cosine=jac_cos.detach().tolist(),native_delta=native.tolist(),model_delta=predicted.tolist(),
        projected_jacobian=projected.detach().tolist(),scope='measured final-command pair on repeated native prefix; contact-mode switches reported in raw evidence')

def assess_region(spec,receipt_path,current,adapter,command_preview,head):
    """command_preview(raw_current_start) restores a separate G07 branch each call."""
    bundle,limits=load_region(spec,receipt_path);origin=bundle['runtime_identity']
    if bundle.get('head')!=head:raise ValueError('Native probe belongs to different head')
    failures=[key for key in STATIC if origin.get(key)!=current.get(key)]
    age=current['policy_version']-origin['policy_version']
    if age<0 or age>limits['accepted_update_age_max']:failures.append('policy-age')
    if tensor_digest(bundle['actor_state'].items())!=origin['actor_state_sha256']:raise ValueError('Probe actor anchor identity mismatch')
    starts=bundle.get('command_starts',[])
    if not starts:raise ValueError('Missing full current-state/executor command drift library')
    normalized=targets=0.
    with torch.no_grad():
        for item in starts:
            final,target=command_preview(item['start'])
            normalized=max(normalized,float((final.cpu()-item['final_normalized']).abs().max()))
            targets=max(targets,float((target.cpu()-item['final_joint_targets']).abs().max()))
    if normalized>limits['final_normalized_command_delta_max']:failures.append('executed-normalized-command-drift')
    if targets>limits['final_joint_target_delta_rad_max']:failures.append('executed-joint-target-drift')
    measured=paired_metrics(adapter,bundle['pairs'],head,limits)
    failures+=measured['failures']
    return dict(passed=not failures,failures=failures,current_identity=current,evidence_sha256=spec['sha256'],
        origin_identity=origin,accepted_update_age=age,final_normalized_command_delta=normalized,
        final_joint_target_delta_rad=targets,metrics=measured,limits=limits,
        scope='current model rechecked against immutable native responses plus bounded policy drift; not an exact current-policy value guarantee')
