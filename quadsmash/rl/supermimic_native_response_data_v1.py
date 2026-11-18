"""Immutable from-ground paired transitions, separate from rollout inputs.

These labels identify executable local responses. Native next states never
become inputs of a deployed prediction or an imagined policy step.
"""
import hashlib,json
from pathlib import Path
import torch
from .supermimic_native_pair_capture_v1 import BRANCHES,tree_digest,probe_lanes,validate_spec
from .supermimic_physical_dataset_v1 import encode_physical
from .supermimic_data_split_v1 import explicit_lane_split


def load_response_data(directory,split_path):
    directory=Path(directory)
    status=json.loads((directory/'status.json').read_text())
    if status.get('status')!='COLLECTION_FINISHED' or status.get('actor_updates')!=0:
        raise ValueError('Native four-branch collection is not complete')
    records={r['branch']:r for r in status['completed']}
    if len(status['completed'])!=4 or set(records)!=set(BRANCHES):raise ValueError('Missing or duplicated native branch')
    runs={};hashes={}
    for branch in BRANCHES:
        path=directory/branch/'native_probe.pt';digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if records[branch]['exit_code']!=0 or records[branch]['probe_sha256']!=digest:raise ValueError('Native branch failed or hash changed')
        hashes[branch]=digest;runs[branch]=torch.load(path,map_location='cpu',weights_only=True)
    base=runs['nominal_a'];validate_spec(base['spec'])
    lanes=probe_lanes(base['spec'])
    if lanes!=list(range(128)):raise ValueError('Response fitting requires original128 ordered measured interventions')
    train,validation=explicit_lane_split(json.loads(Path(split_path).read_text()),128)
    start=base['raw']['start'];physical=start['before']['physical']
    for branch,run in runs.items():
        if run.get('schema')!='native-paired-prefix-run-v1' or run.get('branch')!=branch:raise ValueError('Wrong branch schema')
        for key in ('spec','seed','task_contract','policy_execution','physics','prefix_sha256','executor_before_sha256','execution_code_sha256','restore_mode'):
            if run[key]!=base[key]:raise ValueError('Native prefix/contract mismatch: '+key)
        if any(run.get(k)!=0 for k in ('actor_updates','world_updates','critic_updates')):raise ValueError('Updated probe policy/model')
        if tree_digest(run['actor_state'])!=tree_digest(base['actor_state']) or tree_digest(run['raw']['start'])!=tree_digest(start):
            raise ValueError('Native full start/actor mismatch')
        if not run['raw']['command_valid_lanes'].all():raise ValueError('Measured native transition crosses reset')
    if base['restore_mode']!='from-ground-identical-prefix':raise ValueError('Snapshot restoration is not same-prefix evidence')
    repeat=float((runs['nominal_a']['raw']['next_state']-runs['nominal_b']['raw']['next_state']).abs().max())
    # Measured deterministic probe was specified to repeat exactly; no new
    # tolerance is inferred from the observed response magnitude.
    if repeat!=0:raise ValueError('Native repeatability failed')
    suite=Path(base['task_contract']['suite']['path'])
    if not suite.is_absolute():suite=Path(__file__).resolve().parents[2]/suite
    if hashlib.sha256(suite.read_bytes()).hexdigest()!=base['task_contract']['suite']['sha256']:raise ValueError('Suite changed')
    cases=json.loads(suite.read_text())['cases']
    tensors=dict(state=encode_physical(physical),previous_action=physical['final_action'],feet=physical['foot_contact'].float(),
                 present=physical['ball_present'],elapsed=physical['elapsed_s'],launch=torch.tensor([c['launch']['position_m']+c['launch']['velocity_mps'] for c in cases]))
    for branch,run in runs.items():
        tensors['target_'+branch]=run['raw']['final_joint_targets']
        tensors['next_'+branch]=run['raw']['next_state']
    if any(not torch.isfinite(v).all() for v in tensors.values()):raise ValueError('Nonfinite measured data')
    return tensors,dict(source_sha256=hashes,spec=base['spec'],train_lanes=train.tolist(),validation_lanes=validation.tolist(),
        split_sha256=hashlib.sha256(Path(split_path).read_bytes()).hexdigest(),task_contract=base['task_contract'],
        policy_execution=base['policy_execution'],native_repeat_max=repeat,actor_state_sha256=tree_digest(base['actor_state']),
        scope='fixed-time executable paired labels; no fitted trust bounds; neither success labels nor future actor inputs')


def normalized_response_loss(prediction_minus,prediction_plus,native_minus,native_plus,targets_minus,targets_plus,scale,*,minimum_command_delta=1e-6):
    """Finite-difference supervision per measured final-command radian.

    Saturated interventions have no identifiable directional derivative. They
    remain in pointwise supervision and in diagnostics, rather than inventing
    a derivative or silently reporting them as supported probes.
    """
    if prediction_minus.shape!=prediction_plus.shape or native_minus.shape!=prediction_minus.shape or native_plus.shape!=prediction_minus.shape:
        raise ValueError('Native/predicted response shape mismatch')
    amplitude=(targets_plus-targets_minus).norm(dim=-1)
    supported=amplitude>=minimum_command_delta
    error=((prediction_plus-prediction_minus)-(native_plus-native_minus))/amplitude.clamp_min(minimum_command_delta)[:,None]/scale
    per=error.square().mean(-1)
    loss=(per*supported.to(per.dtype)).sum()/supported.sum().clamp_min(1)
    return loss,dict(supported=int(supported.sum()),saturated=int((~supported).sum()))
