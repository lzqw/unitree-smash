"""Prepare explicit revised entries; retain original source and policy bytes."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def write_new(name, text):
    with (ROOT / name).open('x') as stream:
        stream.write(text)


def replace_once(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new, 1)


def main():
    original_path = ROOT / 'configs/g07_full_racket_eval_v4_r3_origin0.json'
    protocol = json.loads(original_path.read_text())
    protocol['version'] = 'full-racket-singles-rally-v5-contact-flight1-origin0'
    protocol['frozen_at'] = datetime.now(timezone.utc).isoformat()
    protocol['before_v5_policy_training'] = True
    protocol['recovery_association'] = dict(
        schema='contact-flight-recovery-v1',
        source_protocol_sha256=hashlib.sha256(original_path.read_bytes()).hexdigest(),
        preserves_physics_cases_target_and_thresholds=True,
        contact_flight_qualification_and_post_landing_continuous_recovery=True)
    write_new('configs/g07_full_racket_eval_v5_contact_recovery_origin0.json', json.dumps(protocol, indent=2) + '\n')
    config = json.loads((ROOT / 'configs/g07_full_racket_origin0_residual_leg15_v4.json').read_text())
    config['episode_trace']['first_episode_lanes'] += [69, 112]
    write_new('configs/g07_full_racket_contact_recovery_residual_v5.json', json.dumps(config, indent=2) + '\n')

    source = (ROOT / 'scripts/evaluate_full_racket_origin0_residual_train_v4.py').read_text()
    source = replace_once(source,
        "protocol_path = ROOT / 'configs/g07_full_racket_eval_v4_r3_origin0.json'",
        "protocol_path = ROOT / 'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json'")
    source = replace_once(source,
        'from quadsmash.tasks.stand.full_racket_origin0_v4 import full_racket_origin0_cfg',
        'from quadsmash.tasks.stand.contact_recovery_origin0_v5 import contact_recovery_origin0_cfg as full_racket_origin0_cfg\n'
        'from quadsmash.rl.contact_recovery_source_v5 import snapshot_execution_sources')
    source = replace_once(source, 'env = FrozenPanelEnv(cfg, cases=cases, reset_noise=noise)',
        "env = FrozenPanelEnv(cfg, cases=cases, reset_noise=noise,\n"
        "        contact_flight_recovery=protocol['recovery_association']['schema'])")
    source = replace_once(source, '    cfg = full_racket_origin0_cfg(count, protocol, args.precision_receipt)',
        '    execution_snapshot=snapshot_execution_sources(ROOT,args.output)\n'
        '    cfg = full_racket_origin0_cfg(count, protocol, args.precision_receipt)')
    source = replace_once(source,
        '        origin_precision_receipt_sha256=digest(args.precision_receipt),origin_spacing_m=0.,',
        "        source_checkpoint_protocol_sha256=checkpoint['protocol_sha256'],\n"
        '        execution_worktree=str(ROOT),\n'
        '        origin_precision_receipt_sha256=digest(args.precision_receipt),origin_spacing_m=0.,')
    source = replace_once(source,
        'from quadsmash.control.full_racket_origin0_residual_v4 import make_rally_controller, SCHEMA',
        'from quadsmash.control.full_racket_origin0_residual_v4 import make_rally_controller, SCHEMA\n'
        'from quadsmash.control.full_racket_gain_body_adaptation_v4 import SCHEMA as SOURCE_SCHEMA')
    source = replace_once(source,
        "    if checkpoint.get('schema') != SCHEMA or checkpoint.get('protocol_sha256') != digest(protocol_path):",
        "    required_protocol=(protocol['recovery_association']['source_protocol_sha256']\n"
        "        if checkpoint.get('schema')==SOURCE_SCHEMA else digest(protocol_path))\n"
        "    if checkpoint.get('schema') not in (SCHEMA,SOURCE_SCHEMA) or checkpoint.get('protocol_sha256') != required_protocol:")
    source = replace_once(source, '        limits = env.robot.data.joint_pos_limits[0, env.joint_ids]',
        '        limits = env.robot.data.joint_pos_limits[0, env.joint_ids]\n'
        "        atomic_json(args.output/'native_command_inputs.json',dict(\n"
        '            default_joint_position_rad=default.cpu().tolist(),joint_limits_rad=limits.cpu().tolist(),\n'
        '            protocol_sha256=digest(protocol_path),policy_sha256=digest(args.checkpoint)))')
    ast.parse(source)
    write_new('scripts/evaluate_full_racket_contact_recovery_origin0_train_v5.py', source)

    source = (ROOT / 'scripts/train_full_racket_origin0_residual_ppo_v4.py').read_text()
    source = replace_once(source,
        "protocol_path=ROOT/'configs/g07_full_racket_eval_v4_r3_origin0.json'",
        "protocol_path=ROOT/'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json'")
    source = replace_once(source, "default=ROOT/'configs/g07_full_racket_origin0_residual_leg15_v4.json'",
        "default=ROOT/'configs/g07_full_racket_contact_recovery_residual_v5.json'")
    source = replace_once(source,
        'from quadsmash.tasks.stand.full_racket_origin0_v4 import full_racket_origin0_cfg',
        'from quadsmash.tasks.stand.contact_recovery_origin0_v5 import contact_recovery_origin0_cfg as full_racket_origin0_cfg\n'
        'from quadsmash.rl.contact_recovery_source_v5 import snapshot_execution_sources')
    source = replace_once(source,
        'from quadsmash.rl.frozen_origin0_source_v4 import verify_frozen_origin0_source',
        'from quadsmash.rl.contact_recovery_source_v5 import verify_recovery_source as verify_frozen_origin0_source')
    source = replace_once(source,
        '    cfg=full_racket_origin0_cfg(args.num_envs,protocol,args.precision_receipt);cfg.seed=args.seed',
        '    execution_snapshot=snapshot_execution_sources(ROOT,args.output)\n'
        '    cfg=full_racket_origin0_cfg(args.num_envs,protocol,args.precision_receipt);cfg.seed=args.seed')
    source = replace_once(source,
        '        continue_after_miss=True,joint_placement_reward=joint_placement,live_alignment_reward=live_alignment)',
        '        continue_after_miss=True,joint_placement_reward=joint_placement,live_alignment_reward=live_alignment,\n'
        "        contact_flight_recovery=protocol['recovery_association']['schema'])")
    source = replace_once(source,
        "        atomic_json(args.output/'INPUTS.json',dict(contract=contract,training_indices=indices,",
        "        atomic_json(args.output/'INPUTS.json',dict(contract=contract,training_indices=indices,\n"
        '            execution_source=execution_snapshot,')
    source = replace_once(source,
        "    contract.update(fresh_exploration_std=configuration['fresh_exploration_std'],base_controller_schema=base['schema'],fixed_source_actor_count=2)",
        "    contract.update(fresh_exploration_std=configuration['fresh_exploration_std'],base_controller_schema=base['schema'],fixed_source_actor_count=2)\n"
        "    contract['recovery_association']=protocol['recovery_association']")
    source = replace_once(source,
        "        runner.save_restart(args.output/'checkpoint.pt',infos=dict(residual_contract=contract,\n"
        '            learner_initialization=initialization))',
        "        runner.save_restart(args.output/'checkpoint.pt',infos=dict(residual_contract=contract,\n"
        '            learner_initialization=initialization))\n'
        '        assert env.contact_recovery is not None\n'
        '        if any(bool(value.any()) for value in vars(env.contact_recovery).values() if torch.is_tensor(value)):\n'
        "            raise ValueError('Canonical ground restart retains contact-flight episode state')\n"
        "        atomic_json(args.output/'canonical_contact_flight_state.json',dict(\n"
        '            all_contact_flight_buffers_zero=True,all_episode_clocks_zero=not bool(env.episode_length_buf.any()),\n'
        "            checkpoint_sha256=digest(args.output/'checkpoint.pt'),arbitrary_mid_episode_restore_claimed=False))")
    source = replace_once(source, "        atomic_torch(args.output/'policy.pt',policy)",
        "        policy.update(protocol_sha256=contract['protocol_sha256'],\n"
        "            recovery_association=protocol['recovery_association'],\n"
        "            source_checkpoint_protocol_sha256=base['protocol_sha256'])\n"
        "        atomic_torch(args.output/'policy.pt',policy)")
    ast.parse(source)
    write_new('scripts/train_full_racket_contact_recovery_origin0_v5.py', source)
    print('Prepared explicit protocol and entries without modifying their original files.')


if __name__ == '__main__':
    main()
