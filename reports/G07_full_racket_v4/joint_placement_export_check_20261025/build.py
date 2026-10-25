"""Check the completed new learner's exported public actor on actual saved observations."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path

import torch
from tensordict import TensorDict
from rsl_rl.modules import ActorCritic

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RUN = ROOT / 'reports/G07_full_racket_v4/residual_ppo_joint_placement_actor32_n2048_probe8_01'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'REPORT.json')
    args = parser.parse_args()
    assert not args.output.exists(), 'Preserve existing evidence'
    job = json.loads((RUN / 'job.json').read_text())
    assert job['status'] == 'SUCCEEDED' and job['exit_code'] == 0
    native = torch.load(RUN / 'checkpoint.pt', map_location='cpu', weights_only=False)
    portable = torch.load(RUN / 'policy.pt', map_location='cpu', weights_only=False)
    assert native['iter'] == 8 and native['native']['canonical_restart']
    assert native['native']['total_timesteps'] == 2097152
    assert native['native']['environment']['num_envs'] == 2048
    assert native['infos']['learner_initialization']['mode'] == 'fresh_learner_with_transferred_public_actor'
    assert native['infos']['learner_initialization']['source_iteration'] == 32
    assert portable['schema'] == 'full-racket-residual-ppo-v4' and portable['native_execution']
    assert not portable['performance_inherited'] and not portable['formal_acceptance']
    meta = portable['residual_training_metadata']
    assert meta['completed_updates'] == 8 and meta['actual_control_env_steps'] == 2097152
    assert meta['num_envs'] == 2048
    public = native['native']['environment']['tensors']['final_observation'][:200].clone()
    assert public.shape == (200, 794) and torch.isfinite(public).all() and bool(public.abs().sum() > 0)
    observations = TensorDict({'policy': public}, batch_size=[200])
    config = copy.deepcopy(native['native']['training_config'])
    policy_config = copy.deepcopy(config['policy'])
    assert policy_config.pop('class_name') == 'ActorCritic'
    policy = ActorCritic(observations, config['obs_groups'], 18, **policy_config).eval()
    policy.load_state_dict(native['model_state_dict'], strict=True)
    actor = torch.jit.load(io.BytesIO(portable['residual_actor_jit']), map_location='cpu').eval()
    exported = actor.state_dict()
    expected = {}
    for name, value in native['model_state_dict'].items():
        if name.startswith('actor_obs_normalizer.'):
            expected['0.' + name[len('actor_obs_normalizer.'):]] = value
        elif name.startswith('actor.'):
            expected['1.' + name[len('actor.'):]] = value
    assert set(exported) == set(expected)
    assert all(torch.equal(value, exported[name]) for name, value in expected.items())
    with torch.inference_mode():
        reference = policy.act_inference(observations)
        actual = actor(public)
        split = torch.cat([actor(public[:128]), actor(public[128:])])
    torch.testing.assert_close(actual, reference, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(split, actual, atol=2e-5, rtol=2e-5)
    assert actual.shape == (200, 18) and torch.isfinite(actual).all()
    assert not any('critic' in name for name in actor.state_dict())
    blob_sha = hashlib.sha256(portable['residual_actor_jit']).hexdigest()
    assert blob_sha == meta['actor_jit_sha256']
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(),
        scope='Completed fresh joint-reward learner8 and its actual exported public actor; CPU fidelity check on200 native saved public terminal observations. Not a new physical rollout or policy-performance result.',
        canonical_checkpoint=str(RUN / 'checkpoint.pt'), checkpoint_sha256=sha(RUN / 'checkpoint.pt'),
        portable_policy=str(RUN / 'policy.pt'), policy_sha256=sha(RUN / 'policy.pt'),
        source_iteration=8, fresh_learner_control_env_steps=2097152,
        actual_public_observation_shape=list(public.shape), public_observation_source='Canonical native environment.final_observation; actual saved public terminal observations, not generated fixtures.',
        public_observation_sha256=hashlib.sha256(public.contiguous().numpy().tobytes()).hexdigest(),
        output_shape=list(actual.shape), all_actor_and_normalizer_tensors_exact=True,
        exported_actor_state_tensor_count=len(exported), no_critic_exported=True,
        CPU_actor_export_max_error=float((actual-reference).abs().max()),
        split_batch_max_error=float((split-actual).abs().max()),
        original_native_export_comparison_device=meta['export_comparison_device'],
        original_native_export_max_error=meta['export_actor_max_error'],
        actor_jit_sha256=blob_sha, source_program_sha256=sha(Path(__file__).resolve()),
        native_job_started_by_this_check=False, frozen_policy_performance=False,
        formal_cases_read=False, formal_acceptance=False, goal_complete=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(args.output)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
