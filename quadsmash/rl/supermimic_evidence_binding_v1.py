"""Separate static source provenance, current weights and calibrated trust scope."""
import hashlib,json,math
from pathlib import Path
from .supermimic_physical_loop_loading_v1 import tensor_digest


def json_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def runtime_identity(actor,adapter,task_contract,execution,policy_version):
    root=Path(__file__).parent
    code_hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
        ('supermimic_policy_execution_v1.py','supermimic_v1.py','supermimic_physical_imagination_adapter_v1.py',
         'supermimic_probe_region_v1.py','supermimic_native_pair_capture_v1.py','supermimic_ball_sequence_v1.py','supermimic_action_ball_v1.py','supermimic_g07_memory_preview_v1.py','supermimic_execution_preview_v1.py')}
    for name in ('g07_shared_target_residual_v1.py','full_racket_residual_v4.py','full_racket_rally_controller_v4.py','full_racket_arm_target_rate_v4.py','persistent_recovery_assistance_v1.py'):
        code_hashes['control/'+name]=hashlib.sha256((root.parent/'control'/name).read_bytes()).hexdigest()
    return dict(execution_code_sha256=code_hashes,schema='supermimic-runtime-identity-v1',actor_state_sha256=tensor_digest(actor.state_dict().items()),
        world_state_sha256=tensor_digest(adapter.state_dict().items()),task_contract_sha256=json_sha(task_contract),
        policy_execution=execution,policy_version=int(policy_version))


def probe_applicability(probe,current,region=None):
    declared=probe.get('runtime_identity')
    required=('execution_code_sha256','schema','actor_state_sha256','world_state_sha256','task_contract_sha256','policy_execution','policy_version')
    mismatch={k:dict(evidence=None if declared is None else declared.get(k),current=current[k])
              for k in required if declared is None or declared.get(k)!=current[k]}
    static=('execution_code_sha256','schema','task_contract_sha256','policy_execution')
    bounded=(region is not None and region.get('passed') is True
             and region.get('current_identity')==current and region.get('origin_identity')==declared
             and region.get('evidence_sha256')==probe.get('validity_region',{}).get('sha256')
             and not any(k in mismatch for k in static))
    valid=(not mismatch or bounded) and (region is None or region.get('passed') is True)
    return dict(current_version_valid=valid,
                scope='current-exact-version' if not mismatch else ('bounded-native-response-rechecked' if bounded else 'historical-reference-only'),
                mismatch=mismatch,evidence_identity=declared,current_identity=current,region=region,
                revalidation_required=not valid)


def current_value_trust(bank,*,actor_sha,execution,accepted_updates,command_delta=None,limits=None):
    if bank.get('policy_execution') is None:
        return dict(status='unknown',trusted=False,reason='label-execution-unknown')
    if bank['policy_execution']!=execution:
        return dict(status='untrusted',trusted=False,reason='execution-contract-changed')
    if bank.get('actor_sha256')==actor_sha:
        return dict(status='same-frozen-policy',trusted=True,reason=None)
    if limits is None or bank.get('accepted_updates') is None or command_delta is None:
        return dict(status='unknown',trusted=False,reason='missing-calibrated-policy-drift-evidence')
    age=accepted_updates-int(bank['accepted_updates'])
    values=dict(accepted_update_age=age,**command_delta)
    required=('accepted_update_age','final_normalized_command_delta','final_joint_target_delta_rad')
    failures={k:values.get(k) for k in required if not isinstance(values.get(k),(int,float))
        or not math.isfinite(values[k]) or values[k]<0 or not isinstance(limits.get(k),(int,float))
        or not math.isfinite(limits[k]) or limits[k]<0 or values[k]>limits[k]}
    return dict(status='bounded-reference-surrogate' if not failures else 'untrusted',trusted=not failures,
        reason='calibrated-drift-heuristic-not-exact-current-policy-value',measured=values,limits=limits,failures=failures)


def verify_fit_provenance(receipt,receipt_path):
    """A supplied fit manifest must match artifacts and exclude validation identities.

    This verifies recorded provenance, not authenticity of an unsupported declaration.
    """
    item=receipt.get('wm_fit_provenance')
    if not item:return dict(verified=False,reason='missing-historical-WM-fit-manifest')
    p=Path(receipt_path).parent/item['path'];raw=p.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=item['sha256']:raise ValueError('WM fit provenance hash mismatch')
    m=json.loads(raw)
    if m.get('model_sha256')!=receipt['model_sha256']:raise ValueError('WM fit manifest belongs to different base files')
    training=set(m['training_episode_ids']);validation=set(receipt['validation_episode_ids'])
    if not training or not validation or training&validation:raise ValueError('Historical fit/validation episode overlap or missing coverage')
    if set(receipt['training_episode_ids'])!=training:raise ValueError('Receipt and fit-manifest training identities differ')
    return dict(verified=True,manifest_sha256=item['sha256'],training_episodes=len(training),validation_episodes=len(validation),
                scope='recorded manifest identity check; native generalization not proved')


def attach_terminal_ball_side(data,records):
    """Join original terminal scores by lane/episode; absent labels stay NaN."""
    import torch
    labels=torch.full(data['episode'].shape,float('nan'))
    seen=set()
    for record in records:
        if record.get('scoring_schema')!='historical-independent-complete5-v1':raise ValueError('Unknown terminal scoring schema')
        lane=int(record['env_id']);ep=int(record['episode_index']);key=(lane,ep)
        if key in seen:raise ValueError('Duplicate native terminal identity')
        seen.add(key)
        score=record.get('ball_side_complete_except_recovery')
        if type(score) is not bool:raise ValueError('Missing native ball-side label')
        mask=(data['episode'][:,lane]==ep)&(data['terminated'][:,lane]|data['timeout'][:,lane])
        times=torch.where(mask)[0]
        if len(times)!=1:raise ValueError('Native terminal score not paired with exactly one transition')
        labels[times[0],lane]=float(score)
    return dict(data,ball_side_terminal=labels)


def loaded_fit_provenance(initial_provenance,resumed,world_sha,heldout_lanes):
    if not initial_provenance.get('verified'):
        return dict(verified=False,reason='initial-fit-independence-unverified',initial=initial_provenance)
    if resumed is None:return dict(verified=True,scope='verified base fit manifest plus enforced online exclusion')
    saved=resumed.get('wm_fit_provenance',{})
    if saved.get('verified') is not True or saved.get('world_state_sha256')!=world_sha:
        return dict(verified=False,reason='resumed-WM-fit-provenance-missing-or-weight-mismatch')
    if set(saved.get('excluded_lanes',[]))!=set(heldout_lanes):
        return dict(verified=False,reason='resumed-WM-holdout-split-changed')
    if saved.get('initial_manifest_sha256')!=initial_provenance.get('manifest_sha256'):
        return dict(verified=False,reason='resumed-WM-base-fit-provenance-changed')
    return dict(verified=True,scope='verified saved provenance chain; independent physical accuracy checked separately')


def complete_native_rtg(data,gamma):
    """Only observed terminal suffixes get MC labels; reset/unfinished tails are invalid."""
    import torch
    reward=data['reward'].double();episode=data['episode'];done=data['terminated']|data['timeout']
    if reward.shape!=episode.shape or done.shape!=reward.shape:raise ValueError('RTG data shape mismatch')
    if not torch.isfinite(reward).all():raise ValueError('Nonfinite native reward')
    rtg=torch.zeros_like(reward);complete=torch.zeros_like(done)
    value=torch.zeros(reward.shape[1],dtype=reward.dtype,device=reward.device)
    known=torch.zeros(reward.shape[1],dtype=torch.bool,device=reward.device)
    for t in range(len(reward)-1,-1,-1):
        if t+1<len(reward):
            same=episode[t]==episode[t+1]
            known&=same;value=torch.where(same,value,torch.zeros_like(value))
        known=done[t]|known
        value=reward[t]+gamma*value*(~done[t])
        rtg[t]=torch.where(known,value,torch.zeros_like(value));complete[t]=known
    return rtg,complete
