"""Fail-closed run budgets and evidence provenance. Does not run experiments."""
import hashlib,json,math
from pathlib import Path

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def validate_budgets(recovery,approach,recovery_proposals,approach_proposals):
    for head,updates,proposals in (('recovery',recovery,recovery_proposals),('approach',approach,approach_proposals)):
        if updates<0 or proposals<0 or proposals<updates:raise ValueError(f'{head}: proposal budget must be >= effective update target')

def verify_contact_receipt(path,model_paths,task_contract):
    """Verify evidence identity and predeclared bounds, not manufacture trust.

    This gate does not claim the receipt is an independent PhysX experiment.
    Raw evidence is retained/hash checked and must come from a separate approved run.
    """
    path=Path(path);r=json.loads(path.read_text())
    if r.get('schema')!='supermimic-contact-validation-v1':raise ValueError('Contact validation schema mismatch')
    expected={k:sha256(v) for k,v in model_paths.items()}
    if r.get('model_sha256')!=expected:raise ValueError('Contact evidence belongs to different model checkpoints')
    contract_hash=hashlib.sha256(json.dumps(task_contract,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if r.get('task_contract_sha256')!=contract_hash:raise ValueError('Contact evidence execution/physics contract mismatch')
    train=set(r['training_episode_ids']);val=set(r['validation_episode_ids'])
    if not val or train&val:raise ValueError('Contact validation must hold out complete episodes')
    native=r['native_action_probe']
    if native.get('same_prefix_verified') is not True or native.get('paired_samples',0)<1:raise ValueError('Missing repeatable native action-response evidence')
    if native.get('response_direction_verified') is not True:
        raise ValueError('Native action-response direction was not verified')
    if native.get('independent_episode_count', 0) < 1:
        raise ValueError('Native action-response evidence lacks independent episode support')
    if r.get('thresholds_declared_before_validation') is not True:raise ValueError('Contact bounds must be declared before validation')
    for group in ('event_aligned_errors','action_response_errors'):
        values=r[group];limits=r['upper_bounds'][group]
        if not values or set(values)!=set(limits):raise ValueError('Missing measured contact metrics/bounds')
        for key,value in values.items():
            if not isinstance(value,(int,float)) or not math.isfinite(value) or value<0 or not math.isfinite(limits[key]) or value>limits[key]:
                raise ValueError(f'Contact validation failed: {group}/{key}')
    if not r.get('evidence_files'):raise ValueError('Missing contact evidence artifacts')
    for item in r['evidence_files']:
        source=path.parent/item['path']
        if sha256(source)!=item['sha256']:raise ValueError('Contact evidence artifact hash mismatch')
    return dict(receipt_sha256=sha256(path),model_sha256=expected,task_contract_sha256=contract_hash,
                validation_episodes=len(val),current_runtime_validity=None,scope='static source-file and task-contract evidence only; actual runtime weights checked separately')

def verify_native_promotion(metrics_path,checkpoint_path,*,expected_execution=None):
    m=json.loads(Path(metrics_path).read_text());n=m.get('native_episode_metrics',m)
    if expected_execution is not None and m.get('policy_execution')!=expected_execution:raise ValueError('Native evaluation policy execution contract mismatch')
    if m.get('evaluated_checkpoint_sha256')!=sha256(checkpoint_path):raise ValueError('Native evaluation checkpoint hash mismatch')
    if m.get('num_envs')!=128 or m.get('control_ticks')!=800:raise ValueError('Incomplete native evaluation protocol')
    if not math.isfinite(n['native_episode_return_mean']):raise ValueError('Nonfinite native return')
    if not (n['native_episode_return_mean']>-12.52678435 and n['ball_side_complete_except_recovery']>=53 and n['complete5']>=27 and m['numerical_failures']==0):raise ValueError('Native candidate not accepted')
    return dict(checkpoint_sha256=sha256(checkpoint_path),evaluation_sha256=sha256(metrics_path))

def verify_runtime_limits(receipt):
    lanes=receipt.get('heldout_lane_ids',[])
    limits=receipt.get('runtime_upper_bounds',{})
    if not lanes or len(set(lanes))!=len(lanes) or min(lanes)<0 or max(lanes)>=128 or len(lanes)>=128:
        raise ValueError('Large profile needs explicit held-out native lane identities')
    if not limits:raise ValueError('Large profile needs predeclared runtime WM bounds')
    for path,value in limits.items():
        if '/' not in path or any(word in path for word in ('accuracy','_count','_matched')) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:raise ValueError('Invalid runtime WM bound')
    return lanes,limits
