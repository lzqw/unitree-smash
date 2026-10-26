"""Independent CPU checks on completed native training and its frozen128 data."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(root, train_name, frozen_name, source_name, require_bundled=False):
    root = root.resolve()
    sys.path[:0] = [str(root), str(root/'vendor/rsl_rl')]
    import numpy as np
    import torch
    import rsl_rl
    from tensordict import TensorDict
    from quadsmash.evaluation_v4 import independent_episode
    from quadsmash.control.full_racket_train_evaluation_v4 import training_cases, summarize_training
    bundled = Path(rsl_rl.__file__).resolve().is_relative_to(root/'vendor/rsl_rl')
    assert bundled or not require_bundled
    reports = root/'reports/G07_full_racket_v4'
    train, frozen, source = [reports/name for name in [train_name, frozen_name, source_name]]
    for folder, steps in [(train, 8388608), (frozen, 102400), (source, 2097152)]:
        job = read(folder/'job.json')
        assert job['status'] == 'SUCCEEDED' and job['exit_code'] == 0 and job['receipt_present']
        assert job['charged_env_steps'] == read(folder/'completed.json')['env_steps'] == steps
    before = torch.load(source/'checkpoint.pt', map_location='cpu', weights_only=False)
    after = torch.load(train/'checkpoint.pt', map_location='cpu', weights_only=False)
    assert before['iter'] == 8 and after['iter'] == 40
    assert before['native']['canonical_restart'] and after['native']['canonical_restart']
    assert after['native']['training_signature'] == before['native']['training_signature']
    assert after['native']['total_timesteps'] == 10485760
    assert not after['native']['environment']['tensors']['episode_length_buf'].any()
    for name in ['actor_obs_normalizer.count', 'critic_obs_normalizer.count']:
        assert int(after['model_state_dict'][name])-int(before['model_state_dict'][name]) == 8388608
    old, new = before['optimizer_state_dict']['state'], after['optimizer_state_dict']['state']
    assert len(new) == 17 and set(old) == set(new)
    assert all(float(new[k]['step'])-float(old[k]['step']) == 640 for k in old)
    assert all(torch.isfinite(v).all() for v in after['model_state_dict'].values() if torch.is_tensor(v))
    assert all(torch.isfinite(v).all() for s in new.values() for v in s.values() if torch.is_tensor(v))
    protocol = read(root/'configs/g07_full_racket_eval_v4_r3.json')
    protocol['_root'] = str(root)
    rows = read(train/'episodes.json')
    outcomes = [independent_episode(row, protocol) for row in rows]
    counts = dict(completed_episodes=len(rows), air=sum(v['airborne_contact'] for v in outcomes),
        conditional_recovery_count=sum(v['conditional_recovery'] for v in outcomes),
        complete=sum(v['complete'] for v in outcomes), numerical_failures=sum(row['numerical_failures'] for row in rows))
    assert counts == read(train/'metrics.json')['counts'] and counts['numerical_failures'] == 0
    cases, indices = training_cases(root, protocol)
    summary = summarize_training(read(frozen/'episodes.json'), cases, indices, protocol)
    metrics, inputs = read(frozen/'metrics.json'), read(frozen/'INPUTS.json')
    assert all(metrics[k] == v for k, v in summary.items())
    assert inputs['policy_sha256'] == metrics['policy_sha256'] == sha(train/'policy.pt')
    assert inputs['reset_noise_seed'] == metrics['noise_seed'] == 724201
    assert metrics['noise_level'] == 'none' and indices == list(range(128)) and summary['numerical_failures'] == 0
    with np.load(frozen/'all_first_episodes.npz') as data:
        mask = data['valid_first_episode'].astype(bool)
        assert data['link_poses'].shape == (400, 128, 28, 7)
        assert mask.shape == (400, 128) and mask[0].all()
        assert np.isfinite(data['link_poses'][mask]).all() and np.isfinite(data['ball_position_m'][mask]).all()
    with np.load(frozen/'public_and_actions_first2.npz') as data:
        assert data['observations'].shape == (800, 2, 794) and data['actions'].shape == (800, 2, 18)
        samples = torch.from_numpy(data['observations'][data['valid_first_episode'].astype(bool)].copy())
    exporter = reports/'centered_placement075_intermediate_selection_queue01/export_diagnostic.py'
    spec = importlib.util.spec_from_file_location('completed40_export_math', exporter)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    policy = module.policy_from_native(after)
    public = torch.load(train/'policy.pt', map_location='cpu', weights_only=False)
    actor = torch.jit.load(io.BytesIO(public['residual_actor_jit']), map_location='cpu').eval()
    with torch.inference_mode():
        expected = policy.act_inference(TensorDict({'policy': samples}, batch_size=[len(samples)]))
        error = float((actor(samples)-expected).abs().max())
    assert error <= 2e-5 and not public['performance_inherited'] and not public['formal_acceptance']
    return dict(checked_at=datetime.now(timezone.utc).isoformat(), root=str(root), actual_training_run=train_name,
        actual_frozen_run=frozen_name, independently_recomputed_changing_policy_counts=counts,
        independently_recomputed_original_TRAIN128=summary, canonical40_sha256=sha(train/'checkpoint.pt'),
        public40_sha256=sha(train/'policy.pt'), actual_frozen_public_samples=len(samples),
        canonical_vs_public_actor_max_error=error, actor_RMS_count=int(after['model_state_dict']['actor_obs_normalizer.count']),
        critic_RMS_count=int(after['model_state_dict']['critic_obs_normalizer.count']),
        all17_Adam_states_preserved_and_advanced640=True, bundled_RSL_used=bundled,
        scope='CPU completed-data and weight consistency only; no new native execution or formal A/B score.',
        new_native_execution=False, formal_cases_read=False, formal_acceptance=False, goal_complete=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train', default='residual_ppo_centered_placement075_n2048_resume32_02')
    parser.add_argument('--frozen', default='residual_ppo_centered_placement075_n2048_iter40_fixed_train128_02')
    parser.add_argument('--source', default='residual_ppo_centered_placement075_n2048_probe8_01')
    parser.add_argument('--require-bundled', action='store_true')
    args = parser.parse_args()
    assert not args.output.exists()
    result = validate(args.root, args.train, args.frozen, args.source, args.require_bundled)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ['actual_training_run', 'actual_frozen_run',
        'canonical_vs_public_actor_max_error', 'actual_frozen_public_samples', 'bundled_RSL_used']}))
