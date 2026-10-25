"""Recompute frozen iter32 TRAIN outcomes and verify the actual2048-env fork."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
REPORTS = ROOT / 'reports/G07_full_racket_v4'
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not (HERE / 'REPORT.json').exists(), 'Never overwrite an assessment report'
    spec = importlib.util.spec_from_file_location('assessment_checks', REPORTS / 'residual_resume24_queue01/run.py')
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    from quadsmash.evaluation_v4 import independent_episode
    protocol_path = ROOT / 'configs/g07_full_racket_eval_v4_r3.json'
    protocol = core.read(protocol_path)
    protocol['_root'] = str(ROOT)
    run = REPORTS / 'residual_ppo_fixed_train128_iter32_n512_01'
    policy_path = REPORTS / 'residual_ppo_n512_resume24_01/policy.pt'
    current = core.verify_evaluation(run, policy_path)
    previous = core.verify_evaluation(core.PRIOR_EVAL, core.PILOT / 'policy.pt')
    baseline_run = REPORTS / 'train128_lateral_c12_1025'
    baseline = core.verify_evaluation(baseline_run, core.BASE)
    rows = core.read(run / 'episodes.json')
    checked = [independent_episode(row, protocol) for row in rows]
    attack = lambda row: bool(row['released'] and np.isfinite(row['outgoing_velocity_mps']).all()
        and np.linalg.norm(row['outgoing_velocity_mps']) >= 8 and row['outgoing_velocity_mps'][0] > 0
        and row['outgoing_velocity_mps'][2] <= -1)
    air_rows = [row for row, result in zip(rows, checked) if result['airborne_contact']]
    chain = dict(air=len(air_rows), air_and_attack=sum(attack(row) for row in air_rows),
        air_and_target=sum(row['landed_target'] for row in air_rows),
        air_attack_and_target=sum(attack(row) and row['landed_target'] for row in air_rows),
        air_attack_and_recovery=sum(result['conditional_recovery'] and attack(row) for row, result in zip(rows, checked)),
        complete=sum(result['complete'] for result in checked))
    paired = {}
    for name, source in [('iter8', core.PRIOR_EVAL), ('lateral_baseline', baseline_run)]:
        old = [independent_episode(row, protocol) for row in core.read(source / 'episodes.json')]
        assert len(old) == len(checked) == 128
        paired[name] = {key: dict(retained=sum(a[key] and b[key] for a, b in zip(old, checked)),
            gained=sum(not a[key] and b[key] for a, b in zip(old, checked)),
            lost=sum(a[key] and not b[key] for a, b in zip(old, checked)))
            for key in ['airborne_contact', 'conditional_recovery', 'complete']}
    native = REPORTS / 'residual_ppo_n2048_resume64_01'
    inputs = core.read(native / 'INPUTS.json')
    ground_path = native / 'initial_ground_states.json'
    ground = core.read(ground_path)
    initial = inputs['initialization']
    assert initial['mode'] == 'canonical_learner_fork_on_fresh_ground'
    assert initial['previous_num_envs'] == 512 and initial['num_envs'] == 2048
    assert initial['initial_iteration'] == 32 and initial['previous_learner_control_env_steps'] == 1703936
    assert initial['policy_optimizer_normalizers_rng_preserved'] and not initial['old_physical_state_restored']
    assert inputs['resume_checkpoint_sha256'] == sha(REPORTS / 'residual_ppo_n512_resume24_01/checkpoint.pt')
    assert ground['all_episode_clocks_zero'] and ground['source_training_indices'] == list(range(128)) * 16
    roots = np.asarray(ground['root_state_local'])
    assert roots.shape == (2048, 13) and np.isfinite(roots).all()
    assert np.max(abs(roots[:, :3] - np.asarray(protocol['initialization']['root_position_m']))) <= 1e-5
    assert np.max(abs(roots[:, 3:7] - np.array([1., 0., 0., 0.]))) <= 1e-5
    assert np.max(abs(roots[:, 7:])) <= 1e-5
    assert np.max(abs(np.asarray(ground['joint_velocity_rad_s']))) <= 1e-5
    reference_ground = core.read(REPORTS / 'residual_ppo_n512_resume24_01/initial_ground_states.json')
    assert np.max(abs(np.asarray(ground['joint_position_rad']) - np.asarray(reference_ground['joint_position_rad'])[0])) <= 1e-5
    progress = core.read(native / 'progress.json')
    diagnostic_path = native / f'ppo/model_{progress["update"] - 1}.pt'
    model = torch.load(diagnostic_path, map_location='cpu', weights_only=False)
    assert model['iter'] == progress['update'] >= 33
    assert model['native']['environment']['num_envs'] == 2048
    assert model['native']['total_timesteps'] == 1703936 + progress['actual_control_env_steps']
    assert all(torch.isfinite(v).all() for v in model['model_state_dict'].values() if torch.is_tensor(v))
    assert all(torch.isfinite(v).all() for state in model['optimizer_state_dict']['state'].values()
               for v in state.values() if torch.is_tensor(v))
    assert progress['numerical_failures'] == 0
    confirmation = core.read(REPORTS / 'residual_resize2048_queue01/PREVIOUS_TRAINING_COMPLETE_CONTACT_CONFIRMATION.json')
    assert confirmation['exact_contact_record_matches']
    job = core.read(run / 'job.json')
    examples = []
    for index in [19, 23, 107, 0, 8]:
        row = rows[index]
        examples.append(dict(TRAIN_index=index, independent_outcome=checked[index],
            frame_hit=row['frame_hit'], outgoing_speed_mps=float(np.linalg.norm(row['outgoing_velocity_mps'])),
            outgoing_velocity_mps=row['outgoing_velocity_mps'], first_landing_m=row['first_landing_m'],
            recovery_s=row['jump']['recovery_s'], selection='posthoc labelled demonstration' if index == 8 else 'selected before fixed assessment'))
    result = dict(created_at=datetime.now(timezone.utc).isoformat(),
        scope='One frozen residual PPO iter32 TRAIN128 assessment; original gates, no formal A/B claim.',
        policy_sha256=sha(policy_path), independently_recomputed_fixed_TRAIN128=current,
        original_native_metrics_match=True, actual_control_env_steps=102400,
        actual_assessment_gpu_hours=job['charged_gpu_hours'], assessment_source_commit=job['git_sha'],
        earlier_fixed_iter8=previous, earlier_fixed_lateral_baseline=baseline,
        descriptive_paired_comparison=paired, measured_airborne_chain_counts=chain,
        bottleneck='Three actual airborne attacks and three target landings have zero overlap. Complete success remains0; contact/recovery gains do not solve the full chain.',
        selected_video_cases=examples,
        actual_n2048_fork=dict(initialization=initial, all2048_native_ground_states_verified=True,
            distinct_TRAIN_cases=128, lanes_per_case=16, actual_input_sha256=sha(native / 'INPUTS.json'),
            actual_ground_states_sha256=sha(ground_path),
            verified_diagnostic_update=progress['update'], current_job_steps=progress['actual_control_env_steps'],
            cumulative_learner_steps=model['native']['total_timesteps'],
            model_optimizer_tensors_finite=True, numerical_failures_at_snapshot=progress['numerical_failures'],
            diagnostic_checkpoint_sha256=sha(diagnostic_path), diagnostic_checkpoint_canonical_resume=False,
            frozen_policy_performance_pending=True),
        sources={str((run / name).relative_to(ROOT)): sha(run / name) for name in
            ['episodes.json', 'contacts.json', 'metrics.json', 'INPUTS.json', 'initial_states.json', 'job.json',
             'completed.json', 'source_snapshot.zip', 'source_hashes.json', 'all_first_episodes.npz']},
        limitations=['Separate native runs and different policies support descriptive paired counts, not a causal/statistical improvement claim.',
            'The four cases selected before assessment all failed. TRAIN8 is an explicitly posthoc air-attack-and-recovery example, also failing the target.',
            'The single earlier changing-policy training success is not inherited by fixed actor32.',
            'The new2048 batch repeats the same128 TRAIN cases16 times and is not2048 distinct cases.',
            'A native batch fork/finite update is engineering evidence; formal A/B each200 and full goal requirements remain unproven.'],
        formal_cases_read=False, formal_acceptance=False, goal_complete=False)
    core.atomic(HERE / 'REPORT.json', result)
    print(json.dumps(dict(fixed_TRAIN128={k: current[k] for k in ['episodes', 'airborne_contacts',
        'conditional_recovery_count', 'complete', 'numerical_failures']}, chain=chain,
        native2048_verified_update=progress['update'], native2048_steps=progress['actual_control_env_steps'])), flush=True)


if __name__ == '__main__':
    main()
