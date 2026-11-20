"""Strict loading/provenance for one physical imagination path."""
import hashlib,json,types
from pathlib import Path
import torch
from .supermimic_v1 import load_ppo_initialized_actor
from .supermimic_physical_world_v2 import PhysicalWorldV2
from .supermimic_ball_residual_v2 import BallResidualV2, verify_event_calibration
from .supermimic_action_ball_v1 import action_ball_from_checkpoint
from .supermimic_action_ball_v1 import TemporalCausalRacketModeImpulseResidualBallV13
from .supermimic_ball_impulse_prior_v1 import (
    BallImpulsePriorV1, attach_hybrid_impulse_prior, hybrid_ball_manifest)
from .supermimic_task_memory_schema_v1 import TaskMemorySchemaV1,MemoryField
from .supermimic_task_transition_model_v1 import task_model_for_schema
from .supermimic_physical_imagination_adapter_v1 import PhysicalImaginationAdapter
from .supermimic_g07_memory_preview_v1 import G07MemoryPreviewV1
from quadsmash.control.g07_shared_target_residual_v1 import make_g07_residual_controller


SOURCE_SHA='8f510fe9f218beef791ac21675a19ed25d11f510a83fe247cddf2893bee08399'
APPROACH_NAMES=tuple('actor.approach.'+layer+'.'+kind for layer in ('0','2','4') for kind in ('weight','bias'))
RECOVERY_NAMES=tuple('actor.recovery.'+layer+'.'+kind for layer in ('0','2','4') for kind in ('weight','bias'))
HEAD_NAMES=APPROACH_NAMES+RECOVERY_NAMES


def set_actor_trainable(actor, names):
    names=set(names)
    parameters=dict(actor.named_parameters())
    unknown=sorted(names-set(parameters))
    if unknown:
        raise ValueError(f'Unknown actor parameters in whitelist: {unknown}')
    actor.requires_grad_(False)
    for name in names:
        parameters[name].requires_grad_(True)
    if not names:
        raise ValueError('Empty actor whitelist')
    return tuple(sorted(names))


def freeze_execution_modules(obj,seen=None):
    seen=set() if seen is None else seen
    if id(obj) in seen:return
    seen.add(id(obj))
    if isinstance(obj,torch.nn.Module):
        obj.eval()
        for parameter in obj.parameters():parameter.requires_grad_(False)
    elif isinstance(obj,dict):
        for v in obj.values():freeze_execution_modules(v,seen)
    elif isinstance(obj,(tuple,list)):
        for v in obj:freeze_execution_modules(v,seen)
    elif hasattr(obj,'__dict__') and not isinstance(obj,(type,torch.Tensor,types.ModuleType,types.FunctionType,types.MethodType)):
        for v in vars(obj).values():freeze_execution_modules(v,seen)


def tensor_digest(named):
    h=hashlib.sha256()
    for name,tensor in sorted(named):
        h.update(name.encode());h.update(str(tensor.shape).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_physical_loop(*,archive,physical_checkpoint,ball_checkpoint,task_checkpoint,safety_mapping,device='cuda:0',action_ball_checkpoint=None,execution_root=None):
    manifest=json.loads((Path(archive)/'manifest.json').read_text())
    paths=dict(source=Path(manifest['source_checkpoint']),physical=Path(physical_checkpoint),
               ball=Path(ball_checkpoint),task=Path(task_checkpoint))
    if action_ball_checkpoint is not None:
        paths['action_ball']=Path(action_ball_checkpoint)
    hashes={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in paths.items()}
    if hashes['source']!=SOURCE_SHA or manifest['source_sha256']!=SOURCE_SHA:raise ValueError('Wrong source actor')
    actor,actor_meta=load_ppo_initialized_actor(paths['source'],torch.device(device))
    names=dict(actor.named_parameters())
    if not set(HEAD_NAMES)<=names.keys():raise ValueError('Actor head parameter contract mismatch')
    set_actor_trainable(actor, APPROACH_NAMES)
    if execution_root is None:
        recorded = manifest.get('g07_execution_root')
        if recorded is None:
            recorded = manifest.get('init_policy_checkpoint', {}).get('g07_execution_root')
        if recorded is None:
            raise ValueError(
                'G07 execution root is required. Pass execution_root explicitly or record '
                'g07_execution_root in the archive manifest.'
            )
        execution_root = Path(recorded)
    execution_root = Path(execution_root).expanduser().resolve()
    if not execution_root.is_dir():
        raise FileNotFoundError(f'G07 execution root does not exist: {execution_root}')
    controller=make_g07_residual_controller(torch.load(paths['source'],map_location='cpu',weights_only=False),
        manifest['num_envs'],torch.device(device),execution_root)
    freeze_execution_modules(controller)
    executor=G07MemoryPreviewV1(controller);executor.mixer.to(device)
    physical_state=torch.load(paths['physical'],map_location='cpu',weights_only=True)
    physical=PhysicalWorldV2(manifest['task_contract']['asset']['sha256']).to(device)
    if physical_state['schema']!=physical.schema:raise ValueError('Wrong physical model schema')
    physical.load_state_dict(physical_state['model'],strict=True)
    ball_state=torch.load(paths['ball'],map_location='cpu',weights_only=True)
    if 'event_calibration' in ball_state:
        verify_event_calibration(ball_state,physical_state['calibration']['ball_sha256'],hashes['physical'],physical_state)
    ball=BallResidualV2().to(device);ball.load_state_dict(ball_state['model'],strict=True)
    action_ball=None
    if action_ball_checkpoint is not None:
        candidate=torch.load(paths['action_ball'],map_location='cpu',weights_only=True)
        action_ball=action_ball_from_checkpoint(candidate).to(device)
        if candidate.get('schema') == TemporalCausalRacketModeImpulseResidualBallV13.schema:
            expected_version = TemporalCausalRacketModeImpulseResidualBallV13.V13_SEMANTICS_VERSION
            if int(candidate.get('v13_semantics_version', -1)) != expected_version:
                raise ValueError(
                    'V13 candidate semantics version is not the current runtime contract')
        if (candidate.get('physical_sha256')!=hashes['physical'] or
                candidate.get('legacy_ball_sha256')!=hashes['ball'] or
                candidate.get('asset_sha256')!=manifest['task_contract']['asset']['sha256'] or
                int(candidate.get('updates',0))<1):
            raise ValueError('Action-conditioned ball candidate checkpoint/source mismatch')
        if (candidate.get('geometry_hashes')!=action_ball.preview.geometry_hashes or
                candidate.get('softness_m')!=action_ball.softness_m):
            raise ValueError('Action-ball candidate racket/net geometry or support width mismatch')
        hybrid_recorded=candidate.get('hybrid_ball_transition')
        calibration=(hybrid_recorded or {}).get('calibration', {})
        expected_hybrid=hybrid_ball_manifest(**calibration)
        if hybrid_recorded != expected_hybrid:
            raise ValueError('Action-ball candidate is not bound to the current Hybrid transition')
        if candidate.get('schema') == 'causal-action-conditioned-racket-sweep-v2':
            if (candidate.get('closing_softness_m') != action_ball.closing_softness_m or
                    candidate.get('contact_margin_m') != action_ball.contact_margin_m):
                raise ValueError('Causal action-ball gate contract mismatch')
        action_ball.load_state_dict(candidate['model'],strict=True)
        # Every runtime path uses the same registered frozen Hybrid prior.
        # Its physical identity is stored in the adapter manifest and is
        # reconstructed rather than copied into the residual checkpoint.
        attach_hybrid_impulse_prior(action_ball, device, calibration=calibration)
    task_state=torch.load(paths['task'],map_location='cpu',weights_only=True)
    schema=TaskMemorySchemaV1([MemoryField(**dict(f,shape=tuple(f['shape']))) for f in task_state['task_schema']['fields']])
    output_key=max((key for key in task_state['model'] if key.startswith('net.net.') and key.endswith('.bias')),
                   key=lambda key:int(key.split('.')[2]))
    width=task_state['model'][output_key].numel()-2*schema.width-10
    task=task_model_for_schema(task_state.get('schema'),schema,width).to(device)
    task.load_state_dict(task_state['model'],strict=True)
    safety=json.loads(Path(safety_mapping).read_text())
    if safety['asset_sha256']!=manifest['task_contract']['asset']['sha256']:raise ValueError('Safety geometry asset mismatch')
    adapter=PhysicalImaginationAdapter(physical,ball,task,executor,safety_body_names=safety['safety_body_names'],
        action_ball_model=action_ball,
        manifest=dict(checkpoint_paths={k:str(v) for k,v in paths.items()},checkpoint_sha256=hashes,
                      g07_execution_root=str(execution_root),
                      task_schema=schema.manifest(),actor_whitelist=APPROACH_NAMES,
                      actor_heads={'approach':APPROACH_NAMES,'recovery':RECOVERY_NAMES})).to(device)
    if action_ball is not None:
        adapter.manifest['hybrid_ball_transition'] = hybrid_ball_manifest(**calibration)
    suite=Path(__file__).resolve().parents[2]/manifest['task_contract']['suite']['path']
    if hashlib.sha256(suite.read_bytes()).hexdigest()!=manifest['task_contract']['suite']['sha256']:raise ValueError('Suite hash mismatch')
    cases=json.loads(suite.read_text())['cases']
    launch=torch.tensor([c['launch']['position_m']+c['launch']['velocity_mps'] for c in cases],device=device)
    return actor,adapter,controller,launch,manifest
