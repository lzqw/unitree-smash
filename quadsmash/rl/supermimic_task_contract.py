"""Shared source-policy task contract, checked before native construction.

No Isaac import at module scope: provenance and scoring are CPU-auditable.
Old SuperMimic replays without this contract cannot resume into a new task.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VERSION = 'supermimic-source-task-contract-v2'
# The full-racket source factory sets 4096; validate its effective result below.
# This is an audit contract, not a PhysX contact-response parameter.
SOURCE_SWEEP_AUDIT = dict(maximum_iterations=4096, tolerance_m=1e-6)


def validate_sweep_contract(cfg, contract, requested=None):
    """Inherit the source setting; never use zero to silently disable audit."""
    audit = contract.get('sweep_audit')
    if audit != SOURCE_SWEEP_AUDIT:
        raise ValueError('Missing or incompatible source sweep audit contract')
    expected = audit['maximum_iterations']
    if requested is not None and requested != expected:
        raise ValueError(f'Sweep override {requested} differs from source {expected}')
    if cfg.sweep_audit_iterations != expected:
        raise ValueError(f'Effective sweep iterations {cfg.sweep_audit_iterations} differ from source {expected}')
    return expected


def zero_panel_reset_noise(count, seed):
    """Use the historical ndarray contract, including partial-reset indexing."""
    from quadsmash.control.full_racket_train_evaluation_v4 import training_noise, NOISE_SHAPES
    return training_noise(count, {name: 0. for name in NOISE_SHAPES}, seed)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_task_contract(source, source_path, *, execution_root=None,
                          precision_receipt=None, root=ROOT):
    root, source_path = Path(root), Path(source_path).resolve()
    g07 = source.get('schema') == 'g07-shared-target-residual-ppo-v1'
    if not g07 and source.get('schema') != 'full-racket-residual-ppo-v4':
        raise ValueError('Unsupported source execution schema')
    protocol_path = root / ('configs/g07_full_racket_eval_v5_contact_recovery_origin0.json'
                            if g07 else 'configs/g07_full_racket_eval_v4_r3.json')
    protocol = json.loads(protocol_path.read_text())
    if g07 and source.get('protocol_sha256') != sha(protocol_path):
        raise ValueError('G07 source protocol hash differs; no inferred migration')
    if source.get('asset_sha256') != protocol['asset']['sha256']:
        raise ValueError('Source asset does not match protocol')
    for name, expected in {protocol['asset']['path']: protocol['asset']['sha256'],
                           **protocol['physical_configs'],
                           protocol['training_suite']['path']: protocol['training_suite']['sha256']}.items():
        if sha(root / name) != expected:
            raise ValueError('Frozen task file changed: ' + name)
    cases = json.loads((root / protocol['training_suite']['path']).read_text())['cases']
    if len(cases) != 128 or protocol['training_suite']['count'] != 128:
        raise ValueError('Expected original ordered TRAIN128')
    ground = [4096., 4096., .1]
    receipt, receipt_sha, association = None, None, None
    if g07:
        from quadsmash.tasks.stand.full_racket_origin0_v4 import validate_admission
        from quadsmash.tasks.stand.contact_recovery_origin0_v5 import validate_recovery_protocol
        validate_recovery_protocol(protocol)
        metadata = source['residual_training_metadata']
        association = protocol['recovery_association']['schema']
        if source.get('recovery_association') != protocol['recovery_association']:
            raise ValueError('Source recovery association differs')
        if precision_receipt is None:
            if execution_root is None:
                raise ValueError('G07 requires precision receipt or explicit execution root')
            precision_receipt = Path(execution_root) / 'reports/G07_full_racket_v4/origin_precision_after_aim_queue01/PRECISION_ADMISSION.json'
        receipt = str(Path(precision_receipt).resolve())
        receipt_sha = sha(receipt)
        if receipt_sha != metadata['origin_precision_receipt_sha256']:
            raise ValueError('Origin precision receipt hash differs from source')
        validate_admission(receipt)
        geometry = metadata.get('ground_extent_diagnostic')
        if geometry is not None:
            verification = source_path.parent / 'ground_source_verification.json'
            if sha(verification) != metadata.get('ground_source_verification_sha256'):
                raise ValueError('Ground verification hash differs from source')
            if json.loads(verification.read_text())['ground_extent_diagnostic'] != geometry:
                raise ValueError('Ground verification does not describe source extent')
            ground = geometry['requested_m']
    contract = dict(version=VERSION, source_policy_sha256=sha(source_path),
        protocol_path=str(protocol_path.relative_to(root)), protocol_sha256=sha(protocol_path),
        suite=protocol['training_suite'], asset=protocol['asset'],
        environment_spacing_m=0. if g07 else json.loads((root/'configs/court_net_v4.json').read_text())['environment_spacing_m'],
        ground_size_m=ground, contact_flight_recovery=association,
        control_dt_s=protocol['policy_dt_s'], physics_dt_s=protocol['physics_dt_s'],
        episode_s=protocol['episode_seconds'], precision_receipt_sha256=receipt_sha,
        sweep_audit=dict(SOURCE_SWEEP_AUDIT),
        extra_online_mpc=False,
        historical_base_and_recovery_assistance_preserved=g07)
    return contract, protocol, cases, receipt


def make_task_cfg(count, contract, protocol, receipt):
    if contract['contact_flight_recovery'] is not None:
        from quadsmash.tasks.stand.contact_recovery_origin0_v5 import contact_recovery_origin0_cfg
        cfg = contact_recovery_origin0_cfg(count, protocol, Path(receipt))
    else:
        from quadsmash.tasks.stand.full_racket_env_v4 import full_racket_cfg
        cfg = full_racket_cfg(count, contract['physics_dt_s'])
    cfg.ground_size_m = tuple(contract['ground_size_m'])
    cfg.episode_length_s = contract['episode_s']
    if cfg.scene.env_spacing != contract['environment_spacing_m']:
        raise ValueError('Constructed layout differs from task contract')
    if abs(cfg.sim.dt*cfg.decimation-contract['control_dt_s']) > 1e-12:
        raise ValueError('Constructed control period differs from task contract')
    validate_sweep_contract(cfg, contract)
    return cfg


def require_resume_contract(checkpoint, contract):
    if checkpoint.get('metadata', {}).get('task_contract') != contract:
        raise ValueError('Resume task contract missing or changed; preserve old replay as diagnostic only')


def summarize_episode(row):
    """Only terminal evidence, never post-reset live buffers or invented zeros."""
    required = ('env_id', 'episode_index', 'episode_duration_s', 'episode_timeout',
                'qualified_airborne_string_contact', 'ball_side_complete_except_recovery',
                'recovery_success', 'complete5', 'body_ground_seen', 'rally_fault',
                'numerical_failures')
    missing = [k for k in required if k not in row]
    if missing:
        raise ValueError('Missing terminal evidence: ' + ', '.join(missing))
    result = {k: row[k] for k in required}
    result['head_contact'] = row.get('head_contact')
    # Historical body-contact criterion is NOT an independent all-body audit.
    result['independent_nonfoot_contact'] = row.get('independent_nonfoot_contact')
    result['event_source'] = row.get('event_source', 'unknown')
    # Preserve measured terminal returns; old records remain explicitly unknown.
    for name in ('native_episode_return', 'native_episode_length_steps', 'native_return_definition',
                 'native_episode_discounted_return', 'native_return_gamma'):
        result[name] = row.get(name)
    return result
