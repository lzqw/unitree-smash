"""Independently check fresh leg06 TRAIN8 weights and original frozen TRAIN128."""
import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys

TRAIN_NAME = 'residual_ppo_centered_leg06_after075_intermediates_n2048_probe8_01'
FROZEN_NAME = 'residual_ppo_centered_leg06_after075_intermediates_n2048_probe8_fixed_train128_01'
SOURCE_NAME = 'residual_ppo_centered_placement075_n2048_probe8_01'
BASE_NAME = 'residual_ppo_joint_placement_actor32_n2048_probe8_01'
CONFIG = 'configs/g07_full_racket_residual_ppo_centered_leg06_placement075_v4.json'


def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(root, require_bundled=False):
    root = root.resolve()
    sys.path[:0] = [str(root/'vendor/rsl_rl'), str(root)]
    import numpy as np
    import torch
    import rsl_rl
    from tensordict import TensorDict
    from rsl_rl.modules import ActorCritic
    from quadsmash.evaluation_v4 import independent_episode
    from quadsmash.control.full_racket_train_evaluation_v4 import training_cases, summarize_training
    from scripts.initialize_centered_policy_v4 import same_tree
    bundled = Path(rsl_rl.__file__).resolve().is_relative_to(root/'vendor/rsl_rl')
    assert bundled or not require_bundled
    reports = root/'reports/G07_full_racket_v4'
    train, frozen, source = [reports/name for name in [TRAIN_NAME, FROZEN_NAME, SOURCE_NAME]]
    for folder, steps in [(train, 2097152), (frozen, 102400)]:
        job = read(folder/'job.json')
        assert job['status'] == 'SUCCEEDED' and job['exit_code'] == 0 and job['receipt_present']
        assert job['charged_env_steps'] == read(folder/'completed.json')['env_steps'] == steps
    native = torch.load(train/'checkpoint.pt', map_location='cpu', weights_only=False)
    public = torch.load(train/'policy.pt', map_location='cpu', weights_only=False)
    source_public = torch.load(source/'policy.pt', map_location='cpu', weights_only=False)
    inputs, metrics = read(train/'INPUTS.json'), read(train/'metrics.json')
    init = inputs['initialization']
    assert init['mode'] == 'fresh_measured_centered_actor_with_explicit_action_unit_rescaling'
    assert not any(init[k] for k in ['critic_transferred','critic_normalizer_transferred',
        'optimizer_transferred','source_rng_restored','old_physical_state_restored','previous_learning_counters_restored'])
    assert init['source_policy_sha256'] == sha(source/'policy.pt')
    assert native['iter'] == 8 and native['native']['canonical_restart']
    assert native['native']['total_timesteps'] == 2097152
    assert not native['native']['environment']['tensors']['episode_length_buf'].any()
    model = native['model_state_dict']
    assert int(model['actor_obs_normalizer.count']) == 14680064
    assert int(model['critic_obs_normalizer.count']) == 2097152
    options = dict(read(root/CONFIG)['runner']['policy']); options.pop('class_name')
    obs = TensorDict({'policy': torch.zeros(2,794)}, batch_size=[2])
    with contextlib.redirect_stdout(io.StringIO()):
        actor = ActorCritic(obs, read(root/CONFIG)['runner']['obs_groups'], 18, **options)
    actor.load_state_dict(model, strict=True); actor.eval()
    names = [name for name,_ in actor.named_parameters()]
    states = native['optimizer_state_dict']['state']
    ids = native['optimizer_state_dict']['param_groups'][0]['params']
    assert len(states) == len(ids) == len(names) == 17
    adam = {name:float(states[index]['step']) for name,index in zip(names,ids)}
    assert all(value == (160 if name.startswith('critic.') else 120) for name,value in adam.items())
    assert all(torch.isfinite(v).all() for v in model.values() if torch.is_tensor(v))
    assert all(torch.isfinite(v).all() for s in states.values() for v in s.values() if torch.is_tensor(v))
    diagnostics = []
    for iteration in range(1,9):
        path = train/f'ppo/model_{iteration-1}.pt'
        state = torch.load(path, map_location='cpu', weights_only=False)
        assert state['iter'] == iteration and not state['native']['canonical_restart']
        m = state['model_state_dict']
        assert int(m['actor_obs_normalizer.count']) == 12582912+iteration*262144
        assert int(m['critic_obs_normalizer.count']) == iteration*262144
        assert len(state['optimizer_state_dict']['state']) == (8 if iteration <= 2 else 17)
        assert all(torch.isfinite(v).all() for v in m.values() if torch.is_tensor(v))
        diagnostics.append(dict(iteration=iteration, sha256=sha(path)))
    assert metrics['completed_updates'] == 8
    assert [row['actor_warmup'] for row in metrics['history']] == [True]*2+[False]*6
    protocol = read(root/'configs/g07_full_racket_eval_v4_r3.json'); protocol['_root'] = str(root)
    ground = read(train/'initial_ground_states.json')
    roots = np.asarray(ground['root_state_local'])
    assert roots.shape == (2048,13) and np.isfinite(roots).all()
    assert np.max(abs(roots[:,:3]-protocol['initialization']['root_position_m'])) < 1e-5
    assert np.max(abs(roots[:,3:7]-[1.,0.,0.,0.])) < 1e-5 and np.max(abs(roots[:,7:])) < 1e-5
    assert np.max(abs(np.asarray(ground['joint_velocity_rad_s']))) < 1e-5
    assert ground['all_episode_clocks_zero'] and ground['source_training_indices'] == list(range(128))*16
    rows = read(train/'episodes.json'); seen = [0]*2048
    for row in rows:
        lane = row['env_id']
        assert row['episode_index'] == seen[lane] and row['source_training_index'] == lane%128 and row['ground_init']
        seen[lane] += 1
    assert seen == native['native']['environment']['tensors']['episode_index'].tolist() and min(seen) >= 1
    outcomes = [independent_episode(row,protocol) for row in rows]
    counts = dict(completed_episodes=len(rows), air=sum(x['airborne_contact'] for x in outcomes),
        conditional_recovery_count=sum(x['conditional_recovery'] for x in outcomes),
        complete=sum(x['complete'] for x in outcomes), numerical_failures=sum(x['numerical_failures'] for x in rows))
    assert counts == metrics['counts'] and counts['numerical_failures'] == 0
    cases, indices = training_cases(root,protocol); assert indices == list(range(128))
    summary = summarize_training(read(frozen/'episodes.json'),cases,indices,protocol)
    fm, fi = read(frozen/'metrics.json'), read(frozen/'INPUTS.json')
    assert all(fm[k] == v for k,v in summary.items())
    assert fi['policy_sha256'] == fm['policy_sha256'] == sha(train/'policy.pt')
    assert fi['reset_noise_seed'] == fm['noise_seed'] == 724201 and fm['noise_level'] == 'none'
    assert all(v == 0 for v in fi['reset_noise_amplitudes'].values()) and summary['numerical_failures'] == 0
    with np.load(frozen/'all_first_episodes.npz') as arrays:
        valid = arrays['valid_first_episode'].astype(bool)
        assert arrays['link_poses'].shape == (400,128,28,7) and valid.shape == (400,128) and valid[0].all()
        assert np.isfinite(arrays['link_poses'][valid]).all() and np.isfinite(arrays['ball_position_m'][valid]).all()
    with np.load(frozen/'public_and_actions_first2.npz') as arrays:
        samples = torch.from_numpy(arrays['observations'][arrays['valid_first_episode'].astype(bool)].copy())
    restored_jit = torch.jit.load(io.BytesIO(public['residual_actor_jit']), map_location='cpu').eval()
    with torch.inference_mode():
        expected = actor.act_inference(TensorDict({'policy':samples},batch_size=[len(samples)]))
        error = float((restored_jit(samples)-expected).abs().max())
    assert error <= 2e-5 and not public['performance_inherited'] and not public['formal_acceptance']
    base = torch.load(reports/BASE_NAME/'policy.pt', map_location='cpu', weights_only=False)
    assert same_tree(public['base_controller'],base) and same_tree(source_public['base_controller'],base)
    return dict(checked_at=datetime.now(timezone.utc).isoformat(), root=str(root),
        independently_recomputed_changing_policy_counts=counts, independently_recomputed_original_TRAIN128=summary,
        canonical_sha256=sha(train/'checkpoint.pt'), public_sha256=sha(train/'policy.pt'),
        actual_valid_public_samples=len(samples), canonical_vs_public_actor_max_error=error,
        diagnostics=diagnostics, fresh_Adam_steps=adam, all2048_episode_chains_exact=True,
        bundled_RSL_used=bundled, new_native_execution=False, formal_cases_read=False,
        formal_acceptance=False, goal_complete=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require-bundled', action='store_true')
    args = parser.parse_args(); assert not args.output.exists()
    result = validate(args.root,args.require_bundled)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['canonical_vs_public_actor_max_error','actual_valid_public_samples','bundled_RSL_used']}))
