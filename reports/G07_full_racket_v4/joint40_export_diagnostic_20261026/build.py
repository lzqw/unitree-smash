"""Check the completed new learner's exported public actor on actual saved observations."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
import sys
from tensordict import TensorDict
from rsl_rl.modules import ActorCritic

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RUN = ROOT / 'reports/G07_full_racket_v4/residual_ppo_joint_placement_actor32_n2048_resume32_01'
sys.path.insert(0, str(ROOT))


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
    assert native['iter'] == 40 and native['native']['canonical_restart']
    assert native['native']['total_timesteps'] == 10485760
    assert native['native']['environment']['num_envs'] == 2048
    assert native['infos']['learner_initialization']['mode'] == 'strict_canonical_ground_resume'
    assert native['infos']['learner_initialization']['initial_iteration'] == 8
    assert portable['schema'] == 'full-racket-residual-ppo-v4' and portable['native_execution']
    assert not portable['performance_inherited'] and not portable['formal_acceptance']
    meta = portable['residual_training_metadata']
    assert meta['completed_updates'] == 40 and meta['actual_control_env_steps'] == 8388608
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
    from quadsmash.control.full_racket_residual_v4 import FullRacketResidualControllerV4
    fixed = ROOT/'reports/G07_full_racket_v4/residual_ppo_joint_placement_actor32_n2048_iter40_fixed_train128_01'
    saved = np.load(fixed/'public_and_actions_first2.npz')
    stream = torch.tensor(saved['observations'].reshape(-1,794))
    assert stream.shape == (1600,794) and torch.isfinite(stream).all()
    with torch.inference_mode():
        stock = policy.act_inference(TensorDict({'policy':stream},batch_size=[len(stream)]))
        exported_stream = actor(stream)
    torch.testing.assert_close(exported_stream,stock,atol=2e-5,rtol=2e-5)
    source8 = ROOT/'reports/G07_full_racket_v4/residual_ppo_joint_placement_actor32_n2048_probe8_01/policy.pt'
    previous = torch.load(source8,map_location='cpu',weights_only=False)
    old_actor = torch.jit.load(io.BytesIO(previous['residual_actor_jit']),map_location='cpu').eval()
    with torch.inference_mode():old_stream=old_actor(stream)
    contract_path = ROOT/'reports/G07_full_racket_v4/CPU_REPLAY_AUTHORED_JOINTS_0724.json'
    contract = json.loads(contract_path.read_text())
    names = native['native']['interface']['joints']
    defaults = dict(contract['defaults_before_neutral_arm_override'])
    reset_path = ROOT/'configs/full_racket_reset_v4.json'
    arm_defaults = json.loads(reset_path.read_text())['arm_joint_positions_rad']
    defaults.update({f'joint{i+1}':v for i,v in enumerate(arm_defaults)})
    default = torch.tensor([defaults[name]for name in names],dtype=torch.float32)
    limits = torch.tensor([contract['limits'][name]for name in names],dtype=torch.float32)
    controller = FullRacketResidualControllerV4(portable,2,'cpu');replayed=[]
    last_clock = None;reset_count = 0
    with torch.inference_mode():
        for row in saved['observations']:
            obs=torch.tensor(row)
            if last_clock is not None:
                ids=torch.where(obs[:,775]<last_clock-1e-6)[0]
                if len(ids):controller.reset(ids);reset_count+=len(ids)
            replayed.append(controller.action({'policy':obs},default,limits).numpy().copy())
            last_clock=obs[:,775].clone()
    replayed=np.stack(replayed);valid=saved['valid_first_episode']
    controller_error=float(np.abs(replayed-saved['actions'])[valid].max())
    assert np.isfinite(replayed).all() and controller_error<=2e-5
    extra_checks=dict(actual_frozen40_public_rows=len(stream),
        actual_frozen40_actor_export_max_error=float((stock-exported_stream).abs().max()),
        full_controller_replayed_control_ticks=len(replayed),
        full_controller_CPU_vs_actual_native_max_error=controller_error,
        actual_observed_reset_events_replayed=reset_count,
        measured_actor_outputs_on_identical_frozen40_public_inputs=dict(
            new40_max_abs=float(exported_stream.abs().max()),new8_max_abs=float(old_stream.abs().max()),
            new40_mean_abs_per_joint=exported_stream.abs().mean(0).tolist(),
            new8_mean_abs_per_joint=old_stream.abs().mean(0).tolist(),
            mean_abs_difference_per_joint=(exported_stream-old_stream).abs().mean(0).tolist()),
        inference='The exact exported neural actor and full controller reproduce the recorded native actions. This excludes a tensor/serialization/action-replay mismatch on these actual rows; it does not establish the cause of the frozen performance collapse.',
        source_sha256={str(p.relative_to(ROOT)):sha(p)for p in [source8,fixed/'public_and_actions_first2.npz',contract_path,reset_path]})
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(),
        scope='Actual learner40 portable actor and full-controller replay after frozen TRAIN128 returned no contacts. CPU diagnostics only; actual native evaluation remains0/128.',
        canonical_checkpoint=str(RUN / 'checkpoint.pt'), checkpoint_sha256=sha(RUN / 'checkpoint.pt'),
        portable_policy=str(RUN / 'policy.pt'), policy_sha256=sha(RUN / 'policy.pt'),
        source_iteration=40, current_job_control_env_steps=8388608, cumulative_new_learner_control_env_steps=10485760, additional_actual_frozen_stream_checks=extra_checks,
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
