"""Check actor transfer on actual native weights, using a CPU metadata fixture.

The reset callback is recorded only. No PhysX reset or new rollout occurs.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from rsl_rl.modules import ActorCritic
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from scripts.initialize_residual_actor_v4 import initialize_actor, actor_key
from scripts.residual_ground_resize_v4 import validate_ground_resize
from quadsmash.control.measured_joint_placement_reward_v4 import REWARD_VERSION
from quadsmash.rl.native_checkpoint import interface
from quadsmash.rl.native_ppo import NativeRunner


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parent = Path('/home/autodrive/LZQW/unitree_smash/reports/G07_full_racket_v4/residual_ppo_n512_resume24_01')
    parser.add_argument('--source', type=Path, default=parent / 'checkpoint.pt')
    parser.add_argument('--exported-policy', type=Path, default=parent / 'policy.pt')
    args = parser.parse_args()
    assert not (HERE / 'REPORT.json').exists(), 'Never overwrite a preserved transfer check'
    state = torch.load(args.source, map_location='cpu', weights_only=False)
    portable = torch.load(args.exported_policy, map_location='cpu', weights_only=False)
    assert state['iter'] == portable['residual_training_metadata']['completed_updates'] == 32
    assert portable['native_execution'] and not portable['performance_inherited']
    public = state['native']['environment']['tensors']['final_observation'][:200].clone()
    assert public.shape == (200, 794) and torch.isfinite(public).all()
    obs = TensorDict({'policy': public}, batch_size=[200])
    cfg = copy.deepcopy(state['native']['training_config'])
    settings = json.loads((ROOT / 'configs/g07_full_racket_residual_ppo_joint_placement_v4.json').read_bytes())
    cfg['seed'] = 748503
    cfg['residual_contract'].update(reward_version=REWARD_VERSION,
        joint_placement_reward=settings['joint_placement_reward'], seed=748503)
    policy_cfg = copy.deepcopy(cfg['policy'])
    assert policy_cfg.pop('class_name') == 'ActorCritic'
    torch.manual_seed(cfg['seed'])
    policy = ActorCritic(obs, cfg['obs_groups'], 18, **policy_cfg)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg['algorithm']['learning_rate'])
    before = {key: value.clone() for key, value in policy.state_dict().items()}
    parameters_before = [id(p) for p in policy.parameters()]
    # These fields reproduce the real interface metadata, not robot physics.
    native_interface = state['native']['interface']
    metadata = SimpleNamespace(step_dt=native_interface['policy_dt'],
        robot=SimpleNamespace(joint_names=native_interface['joints']),
        joint_ids=list(range(18)), action_family=native_interface['action_family'], jump_clock=None,
        cfg=SimpleNamespace(robot=SimpleNamespace(spawn=SimpleNamespace(
            usd_path=str(ROOT / 'assets/quadsmash_full_racket_v4_r3.usd')))))
    current_interface = interface(metadata)
    assert current_interface == native_interface
    reset_calls = []
    wrapper = SimpleNamespace(num_envs=2048, unwrapped=metadata,
        reset=lambda: reset_calls.append('CPU callback only; real native ground reset not executed'))
    runner = SimpleNamespace(alg=SimpleNamespace(policy=policy, optimizer=optimizer),
        env=wrapper, original_training_config=cfg, current_learning_iteration=0,
        tot_timesteps=0, tot_time=0., device='cpu')
    rejections = {}
    try:
        NativeRunner.load(runner, args.source, load_optimizer=True, map_location='cpu')
    except ValueError as error:
        assert 'resume changes the PPO configuration' in str(error)
        rejections['strict_full_resume'] = str(error)
    else:
        raise AssertionError('Old full resume unexpectedly accepted the new reward contract')
    try:
        validate_ground_resize(state, current_interface, cfg, 2048)
    except ValueError as error:
        assert 'cannot change PPO' in str(error)
        rejections['strict_ground_resize'] = str(error)
    else:
        raise AssertionError('Old resize unexpectedly accepted the new reward contract')
    assert all(torch.equal(value, before[key]) for key, value in policy.state_dict().items())
    rng_before = torch.get_rng_state().clone()
    initialization = initialize_actor(runner, args.source)
    assert len(reset_calls) == 1 and torch.equal(torch.get_rng_state(), rng_before)
    assert [id(p) for p in policy.parameters()] == parameters_before
    assert not optimizer.state and runner.current_learning_iteration == runner.tot_timesteps == runner.tot_time == 0
    for key, value in policy.state_dict().items():
        expected = state['model_state_dict'][key] if actor_key(key) else before[key]
        assert torch.equal(value, expected), key
    actor = torch.jit.load(io.BytesIO(portable['residual_actor_jit']), map_location='cpu').eval()
    policy.eval()
    with torch.inference_mode():
        actual = policy.act_inference(obs)
        expected = actor(public)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    assert actual.shape == (200, 18) and torch.isfinite(actual).all()
    files = ['scripts/initialize_residual_actor_v4.py', 'scripts/train_full_racket_residual_ppo_v4.py',
        'scripts/residual_ground_resize_v4.py', 'quadsmash/rl/native_checkpoint.py',
        'quadsmash/rl/native_ppo.py', 'configs/g07_full_racket_residual_ppo_joint_placement_v4.json',
        'tests/test_initialize_residual_actor_v4.py']
    result = dict(created_at=datetime.now(timezone.utc).isoformat(),
        scope='Actual completed native iter32 weights transferred into a real fresh stock-RSL actor/critic on CPU. Environment interface/reset uses an explicitly labelled metadata fixture; no physical execution.',
        source_checkpoint=str(args.source), source_checkpoint_sha256=sha(args.source),
        source_portable_policy=str(args.exported_policy), source_portable_policy_sha256=sha(args.exported_policy),
        actual_source_iteration=state['iter'], actual_source_learner_steps=state['native']['total_timesteps'],
        original_strict_guards_rejected_changed_reward=rejections,
        initialization=initialization, actual_saved_TRAIN_public_observation_shape=[200, 794],
        output_shape=list(actual.shape), comparison_with_actual_portable_actor_max_error=float((actual - expected).abs().max()),
        comparison_atol=2e-5, comparison_rtol=2e-5,
        all_transferred_actor_normalizer_log_std_tensors_bitwise_equal=True,
        all_fresh_critic_and_critic_normalizer_tensors_bitwise_unchanged=True,
        parameter_objects_and_fresh_optimizer_bindings_preserved=True,
        source_optimizer_states=len(state['optimizer_state_dict']['state']), target_optimizer_states=len(optimizer.state),
        new_learner_counters_zero=True, CPU_rng_unchanged_with_noop_reset_callback=True,
        actual_native_ground_reset_executed=False, new_native_training_run=False, performance_inherited=False,
        source_sha256={path: sha(ROOT / path) for path in files},
        limitations=['This checks actual weight transfer, old-guard rejection and CPU inference. The reset callback is not a native environment.',
            'The common trainer must still verify actual ground root/q/qd and zero episode clocks on a future single GPU0 native run.',
            'The source is completed iter32, not the currently running learner. Choose and verify any later canonical source only after its job ends.',
            'Actor transfer starts a distinct learner/reward lineage at iteration0; it is not continuous old PPO/critic/Adam/RNG/physics resume.',
            'Existing sampling, evaluation rules and original A/B acceptance remain unchanged. No new policy-performance claim.'],
        formal_cases_read=False, formal_acceptance=False, goal_complete=False)
    tmp = HERE / 'REPORT.json.tmp'
    tmp.write_text(json.dumps(result, indent=2) + '\n')
    os.replace(tmp, HERE / 'REPORT.json')
    print(json.dumps(dict(source_iteration=32, actor_max_error=result['comparison_with_actual_portable_actor_max_error'],
        actor_tensors_bitwise_equal=True, critic_unchanged=True, optimizer_states=len(optimizer.state),
        strict_guard_rejections=list(rejections), new_native_training_run=False)))


if __name__ == '__main__':
    main()
