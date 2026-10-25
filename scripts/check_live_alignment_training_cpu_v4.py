"""Check real canonical actor transfer and retrospective TRAIN event formulas.

No simulator is created. This is not a native rollout, historical payment
reconstruction, frozen-policy improvement or formal acceptance result.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import torch
from rsl_rl.modules import ActorCritic
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.initialize_residual_actor_v4 import select_actor_state, actor_key
from quadsmash.control.measured_string_alignment_reward_v4 import (
    REWARD_VERSION, alignment_settings, attack_aligned_target, MeasuredStringAlignmentRewardV4)
from quadsmash.evaluation_v4 import FAULTS, independent_episode


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); source_root = args.source_root.resolve()
    assert not args.output.exists(), 'Preserve previous CPU evidence'
    reports = source_root/'reports/G07_full_racket_v4'
    training = reports/'residual_ppo_joint_placement_actor32_n2048_probe8_01'
    fixed = reports/'residual_ppo_joint_placement_actor32_n2048_probe8_fixed_train128_01'
    for run in [training, fixed]:
        job = json.loads((run/'job.json').read_text())
        assert job['status'] == 'SUCCEEDED' and job['exit_code'] == 0
    assert sha(training/'checkpoint.pt') == '10bf6c026a39e3ea01bf08425029ba78a906e573051e9bc3bcf26ccd37373cb0'
    assert sha(training/'policy.pt') == '846656e9d02c74f4b72929253ed9196557af4f9affe39236fc55d7f4bc5d744c'
    config_path = ROOT/'configs/g07_full_racket_residual_ppo_live_alignment_v4.json'
    config = json.loads(config_path.read_text())
    settings = alignment_settings(config['live_alignment_reward'])
    assert settings['discount'] == config['runner']['algorithm']['gamma']
    protocol_path = ROOT/'configs/g07_full_racket_eval_v4_r3.json'
    protocol = json.loads(protocol_path.read_text()); protocol['_root'] = str(ROOT)
    for name, expected in {**protocol['physical_configs'], protocol['asset']['path']:protocol['asset']['sha256']}.items():
        assert sha(ROOT/name) == sha(source_root/name) == expected
    assert sha(protocol_path) == sha(source_root/'configs/g07_full_racket_eval_v4_r3.json')
    native = torch.load(training/'checkpoint.pt', map_location='cpu', weights_only=False)
    portable = torch.load(training/'policy.pt', map_location='cpu', weights_only=False)
    assert native['iter'] == 8 and native['native']['canonical_restart']
    current_interface = native['native']['interface']
    assert current_interface['observation_size'] == 794 and current_interface['asset_sha256'] == protocol['asset']['sha256']
    contract = copy.deepcopy(native['infos']['residual_contract'])
    contract.update(reward_version=REWARD_VERSION, live_alignment_reward=settings,
        target_event_requires_actual_attack=True, fixed_upper_uv_position_shaping=False)
    public = native['native']['environment']['tensors']['final_observation'][:200].clone()
    observations = TensorDict({'policy':public}, batch_size=[200])
    policy_config = copy.deepcopy(config['runner']['policy'])
    assert policy_config.pop('class_name') == 'ActorCritic'
    target = ActorCritic(observations, config['runner']['obs_groups'], 18, **policy_config).eval()
    before = {key:value.clone() for key,value in target.state_dict().items()}
    selected = select_actor_state(native, current_interface, contract, target.state_dict())
    merged = target.state_dict(); merged.update(selected); target.load_state_dict(merged, strict=True)
    assert all(torch.equal(value, native['model_state_dict'][name] if actor_key(name) else before[name])
        for name,value in target.state_dict().items())
    optimizer = torch.optim.Adam(target.parameters(), lr=config['runner']['algorithm']['learning_rate'])
    assert not optimizer.state
    exported = torch.jit.load(io.BytesIO(portable['residual_actor_jit']), map_location='cpu').eval()
    with torch.inference_mode():
        expected = exported(public); actual = target.act_inference(observations)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    summaries = []
    dependencies = [training/'checkpoint.pt', training/'policy.pt', config_path, protocol_path, Path(__file__).resolve()]
    for run in [training, fixed]:
        rows = json.loads((run/'episodes.json').read_text())
        outcomes = [independent_episode(row, protocol) for row in rows]
        legal_air = torch.tensor([r['legal_initial_contact'] and r['string_hit'] and r['airborne_string_hit'] for r in rows])
        faulted = torch.tensor([any(r[name] for name in FAULTS) for r in rows])
        released = torch.tensor([r['released'] for r in rows])
        velocity = torch.tensor([r['outgoing_velocity_mps'] for r in rows], dtype=torch.float32)
        attack = (legal_air & released & torch.isfinite(velocity).all(1)
            & (velocity.norm(dim=1) >= 8.) & (velocity[:, 0] > 0.) & (velocity[:, 2] <= -1.))
        landed_target = torch.tensor([r['landed_target'] for r in rows])
        old_target = legal_air & landed_target & (~faulted)
        new_target = attack_aligned_target(legal_air, attack, landed_target, faulted)
        count = len(rows); zeros = torch.zeros(count, dtype=torch.bool)
        term = MeasuredStringAlignmentRewardV4(count, 'cpu', settings,
            semiaxes_yz_m=[.0915,.1165], contact_plane_offset_m=.0256)
        data = dict(clock_s=torch.full((count,),8.), ball_relative_tool_m=torch.zeros(count,3),
            plane_distance_rate_mps=torch.zeros(count), ball_in_play=zeros,
            had_contact=torch.tensor([r['had_contact'] for r in rows]),
            ground_seen=torch.tensor([r['ground_seen'] for r in rows]), body_contact=zeros,
            faulted=faulted, legal_air=legal_air, released=released,
            frame_hit=torch.tensor([r['frame_hit'] for r in rows]), outgoing_velocity_mps=velocity,
            terminated=torch.ones(count,dtype=torch.bool))
        term.step(**data); components = term.last_components.clone()
        assert torch.isfinite(components).all() and components[:,0].eq(0).all()
        assert term.step(**data).eq(0).all()
        summaries.append(dict(source_run=str(run.relative_to(source_root)), episodes=count,
            original_independent_counts=dict(air=sum(r['airborne_contact'] for r in outcomes),
                conditional_recovery=sum(r['conditional_recovery'] for r in outcomes),
                complete=sum(r['complete'] for r in outcomes)),
            terminal_flag_old_target_eligible=int(old_target.sum()),
            terminal_flag_new_attack_target_eligible=int(new_target.sum()),
            terminal_flag_soft_targets_removed=int((old_target & (~attack)).sum()),
            retrospective_release_progress_positive=int((components[:,1]>0).sum()),
            retrospective_release_progress_sum=float(components[:,1].sum()),
            retrospective_frame_penalty_sum=float(components[:,2].sum()),
            actual_per_control_paid_rewards_reconstructed=False))
        dependencies += [run/'episodes.json', run/'metrics.json', run/'job.json']
    dependencies += [ROOT/'quadsmash/control/measured_string_alignment_reward_v4.py',
        ROOT/'quadsmash/tasks/stand/full_racket_residual_env_v4.py',
        ROOT/'scripts/initialize_residual_actor_v4.py', ROOT/'scripts/train_full_racket_residual_ppo_v4.py']
    for path in dependencies:
        if path.suffix == '.py':
            compile(path.read_text(), str(path), 'exec')
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(),
        scope='CPU source compatibility, actual canonical public-actor tensor transfer and retrospective terminal-event formula replay. No simulator, native proposal execution or policy-performance improvement.',
        configuration=str(config_path.relative_to(ROOT)), new_reward_version=REWARD_VERSION,
        actual_source_canonical_iteration=8, actual_source_ground_lanes=2048,
        actual_public_observations=200, actual_public_actor_max_error=float((actual-expected).abs().max()),
        transferred_actor_tensor_count=len(selected), only_public_actor_normalizer_and_exploration_selected=True,
        fresh_critic_and_critic_normalizer_unchanged=True, fresh_Adam_empty=True,
        no_physics_RNG_or_learning_counter_restore_performed=True,
        original_asset_physical_configs_protocol_and_thresholds_unchanged=True,
        terminal_event_replays=summaries,
        limitations=['Final flags may differ from flags at release/landing; terminal replay is not actual historical payment telemetry.',
            'The measured-position potential was not reconstructed as a historical per-control stream.',
            'CPU actor transfer does not prove a future native ground initialization or optimizer update.',
            'No v3 native training or frozen assessment has run. Existing main training, trials, ledger and all scores remain separate.'],
        source_sha256={str(path):sha(path) for path in dependencies},
        native_execution=False, formal_cases_read=False, formal_acceptance=False, goal_complete=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.output.with_suffix('.json.tmp');temporary.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');temporary.replace(args.output)
    print(json.dumps(dict(actual_public_actor_max_error=result['actual_public_actor_max_error'],
        transferred_tensors=len(selected), terminal_event_replays=summaries, native_execution=False)),flush=True)


if __name__ == '__main__':
    main()
