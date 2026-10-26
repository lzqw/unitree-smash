"""After actual gain/body cold128, learn a fresh origin0 whole-body residual.

No current runtime files are touched. New code is installed only after the
prior owner has actually exited successfully. All proposals are regenerated
from its selected, independently measured cold source before any GPU launch.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
REPORTS = ROOT/'reports/G07_full_racket_v4'
PREVIOUS = REPORTS/'gain_body_after_prepose_queue02'
PRECISION_RECEIPT = REPORTS/'origin_precision_after_aim_queue01/PRECISION_ADMISSION.json'
PROTOCOL = ROOT/'configs/g07_full_racket_eval_v4_r3_origin0.json'
PREVIOUS_PID = 1895642
ISOLATED = ROOT.parent/'unitree_smash_landing_targets'
PYTHON = '/home/autodrive/Isaac/env_isaaclab/bin/python'
TRAIN = REPORTS/'origin0_residual_fresh_n128_updates15_01'
FROZEN = REPORTS/'origin0_residual_fresh15_fixed_train128_01'
CONFIG = ROOT/'configs/g07_full_racket_origin0_residual_leg15_v4.json'
UPDATES, STEPS = 15, 245760
PLAN, STATE = HERE/'PLAN.json', HERE/'state.json'
INSTALL = ['quadsmash/control/full_racket_origin0_residual_v4.py',
           'quadsmash/rl/full_racket_origin0_residual_ppo_v4.py',
           'quadsmash/rl/frozen_origin0_source_v4.py',
           'scripts/train_full_racket_origin0_residual_ppo_v4.py',
           'scripts/evaluate_full_racket_origin0_residual_train_v4.py',
           'scripts/origin0_launcher_parser_v4.py',
           'scripts/check_full_racket_origin0_residual_selected_cpu_v4.py',
           'configs/g07_full_racket_origin0_residual_leg15_v4.json']
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('prepose_shared_guard_checks', REPORTS/'residual_resume24_queue01/run.py')
core = importlib.util.module_from_spec(spec); spec.loader.exec_module(core)
stopping, child = False, None


def save(path, value):
    assert not path.exists(), 'Preserve previous evidence: '+str(path)
    core.atomic(path, value)


def write(state, **extra):
    core.atomic(STATE, dict(state=state, queue_pid=os.getpid(), observed_at=datetime.now(timezone.utc).isoformat(),
        formal_cases_read=False, formal_acceptance=False, goal_complete=False, **extra))


def check(expected):
    for name, digest in expected.items():
        assert core.sha(Path(name)) == digest, name


def ready(summary):
    return (summary['complete_success_rate'] >= .5 and summary['airborne_contact_rate'] >= .7
            and summary['conditional_recovery_rate'] >= .85 and summary['numerical_failures'] == 0)


def run_cpu(script, arguments, label):
    assert not stopping
    with (HERE/(label+'.log')).open('x') as log:
        result = subprocess.run([PYTHON, str(ROOT/script), *arguments], cwd=ROOT,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2'), stdout=log, stderr=subprocess.STDOUT)
    assert result.returncode == 0, 'Preserve CPU failure; do not launch an unreviewed native comparison'


def run_job(output, script, arguments, cap, seconds, label, expected):
    global child
    assert not output.exists() and not stopping
    check(expected)
    assert not any(row['status'] == 'RUNNING' for row in core.read(ROOT/'status/GPU_LEDGER.json')['runs'])
    command = [PYTHON, str(ROOT/'scripts/run_gpu_job.py'), '--goal', 'G07', '--budget-window',
        'g07-full-racket-unlimited-20261024', '--max-seconds', str(seconds), '--env-steps-cap', str(cap),
        '--output', str(output), '--require-artifact', 'completed.json', '--', PYTHON, str(ROOT/script),
        '--output', str(output), *arguments, '--headless']
    with (HERE/(label+'.guard.log')).open('x') as log:
        child = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2'),
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write(label, guard_pid=child.pid, output=str(output))
        while child.poll() is None:
            assert not stopping
            if output == TRAIN:
                for iteration in [1, 7, 15]:
                    proof = HERE/f'ACTUAL_UPDATE_{iteration:02d}.json'
                    if (TRAIN/f'ppo/model_{iteration-1}.pt').exists() and not proof.exists():
                        save(proof, verify_update(iteration, Path(arguments[arguments.index('--base-checkpoint')+1])))
            time.sleep(5)
    assert child.returncode == 0 and not stopping, 'Preserve actual failure; no automatic retry'
    check(expected)
    return verify_job(output, cap)


def verify_job(output, cap):
    job, receipt = core.read(output/'job.json'), core.read(output/'completed.json')
    assert job['status'] == 'SUCCEEDED' and job['charged_env_steps'] == cap
    assert 0 < receipt['env_steps'] <= cap and not (output/'error.txt').exists()
    assert not core.live(job['pid'], output.name) and not core.live(job['child_pid'], output.name)
    # The existing guard conservatively charges its declared step cap. Native
    # receipt/env-step counts stay separate, including early ended candidates.
    return job


def verify_current_evaluation(output, policy):
    import numpy as np
    from quadsmash.control.full_racket_train_evaluation_v4 import training_cases, summarize_training
    protocol = core.read(PROTOCOL); protocol['_root'] = str(ROOT)
    cases, indices = training_cases(ROOT, protocol)
    inputs, metrics = core.read(output/'INPUTS.json'), core.read(output/'metrics.json')
    assert inputs['training_indices'] == indices == list(range(128))
    assert inputs['policy_sha256'] == metrics['policy_sha256'] == core.sha(policy)
    assert inputs['protocol_sha256'] == metrics['protocol_sha256'] == core.sha(PROTOCOL)
    assert inputs['origin_spacing_m'] == 0. and inputs['origin_precision_receipt_sha256'] == core.sha(PRECISION_RECEIPT)
    assert all(value == 0 for value in inputs['reset_noise_amplitudes'].values()) and metrics['noise_level'] == 'none'
    origins = np.asarray(core.read(output/'origin_layout.json')['actual_environment_origins_m'])
    assert origins.shape == (128, 3) and not origins.any()
    summary = summarize_training(core.read(output/'episodes.json'), cases, indices, protocol)
    assert all(metrics[key] == value for key, value in summary.items()) and summary['numerical_failures'] == 0
    assert metrics['actual_control_env_steps'] == core.read(output/'completed.json')['env_steps']
    verify_job(output, 102400)
    return summary


def prepare():
    assert ROOT.name == 'unitree_smash' and not any(path.exists() for path in [PLAN, STATE, TRAIN, FROZEN])
    assert core.live(PREVIOUS_PID, 'gain_body_after_prepose_queue02/run.py')
    expected = dict(core.read(PREVIOUS/'PLAN.json')['source_sha256'])
    check(expected)
    proof_dir = ISOLATED/'reports/G07_full_racket_v4/origin0_selected_residual_binding_20261027'
    proof = core.read(proof_dir/'CPU_CHECK.json')
    assert proof['new_zero_residual_vs_unchanged_source_max_action_error'] == proof['export_max_actor_error'] == 0
    assert proof['CPU_vs_recorded_origin0_GPU_max_action_error'] <= 5e-4 and not proof['new_native_execution']
    check(proof['source_sha256'])
    for name in INSTALL:
        assert (ISOLATED/name).is_file() and not (ROOT/name).exists()
        expected[str(ISOLATED/name)] = core.sha(ISOLATED/name)
    for path in [Path(__file__), PROTOCOL, PRECISION_RECEIPT, PREVIOUS/'PLAN.json',
                 proof_dir/'CPU_CHECK.json', REPORTS/'residual_resume24_queue01/run.py']:
        expected[str(path)] = core.sha(path)
    save(PLAN, dict(created_at=datetime.now(timezone.utc).isoformat(), source_sha256=expected,
        wait_actual_parent_pid=PREVIOUS_PID, required_parent_terminal_state='COMPLETED_GAIN_BODY_ORIGIN0_SEARCH_AND_COLD128',
        source_choice='Bind actual completed gain/body winner and its full cold128 after real owner exit. Verify every original case, weights, physics and actual origin0 source anew.',
        hypothesis='Fresh whole-body residual adapts strike and recovery under corrected numerical dynamics; old source actors and their normalization remain intact.',
        initialization='Fresh zero18D mean, new actor/critic RMS and empty Adam. 0.15rad residual scales; physical request sigma leg0.03/arm0.015rad. Seven critic-only updates followed by eight actor updates.',
        train_num_envs=128, train_updates=UPDATES, training_env_steps_cap=STEPS, training_max_seconds=7200,
        training_seed=751001, cold_env_steps_cap=102400, cold_max_seconds=2700, cold_seed=724201,
        install_only_new_files_after_actual_parent_exit=INSTALL,
        original_target_hardware_physics_cases_and_ground_reset_retained=True,
        automatic_retry=False, formal_cases_read=False, formal_acceptance=False, goal_complete=False))
    print(json.dumps(dict(prepared=str(PLAN), bound_inputs=len(expected), native_job_started=False)))


def optimizer_steps(native):
    model, optimizer = native['model_state_dict'], native['optimizer_state_dict']
    ids = [i for group in optimizer['param_groups'] for i in group['params']]
    names = [name for name in model if name == 'log_std' or name.startswith(('actor.', 'critic.'))]
    assert len(ids) == len(names) == 17
    return {name:int(optimizer['state'][i]['step']) for i, name in zip(ids, names) if i in optimizer['state']}


def verify_update(iteration, base_path):
    import numpy as np
    import torch
    initial = torch.load(TRAIN/'fresh_initial_model_state.pt', map_location='cpu', weights_only=False)
    start = initial['model_state_dict']
    assert initial['base_policy_sha256'] == core.sha(base_path) and initial['actual_control_env_steps'] == 0
    assert initial['optimizer_entries'] == 0 and initial['fixed_source_residual_actor_count'] == 2
    assert int(start['actor_obs_normalizer.count']) == int(start['critic_obs_normalizer.count']) == 0
    assert torch.allclose(start['log_std'].exp(), torch.tensor([.2]*12+[.1]*6), atol=1e-7, rtol=0)
    actor_keys = [name for name in start if name.startswith('actor.')]
    assert not start[actor_keys[-1]].any() and not start[actor_keys[-2]].any()
    path = TRAIN/f'ppo/model_{iteration-1}.pt'
    native = torch.load(path, map_location='cpu', weights_only=False)
    model, adam = native['model_state_dict'], optimizer_steps(native)
    steps = iteration*16384
    assert native['iter'] == iteration and not native['native']['canonical_restart']
    assert native['native']['total_timesteps'] == steps
    assert int(model['actor_obs_normalizer.count']) == int(model['critic_obs_normalizer.count']) == steps
    equal = all(torch.equal(model[name], start[name]) for name in [*actor_keys, 'log_std'])
    assert equal == (iteration <= 7)
    assert len(adam) == (8 if iteration <= 7 else 17)
    assert all(value == (iteration*20 if name.startswith('critic.') else (iteration-7)*20) for name, value in adam.items())
    if iteration == 7:
        assert int(native['native']['environment']['tensors']['episode_index'].min()) >= 1
    inputs = core.read(TRAIN/'INPUTS.json')
    assert inputs['configuration_sha256'] == core.sha(CONFIG) and inputs['num_envs'] == 128 and inputs['requested_updates'] == UPDATES
    assert inputs['contract']['base_policy_sha256'] == core.sha(base_path) and inputs['contract']['seed'] == 751001
    assert inputs['contract']['protocol_sha256'] == core.sha(PROTOCOL)
    if iteration == 1:
        ground = core.read(TRAIN/'initial_ground_states.json'); protocol = core.read(PROTOCOL)
        roots = np.asarray(ground['root_state_local'])
        assert roots.shape == (128, 13) and np.isfinite(roots).all()
        assert abs(roots[:, :3]-protocol['initialization']['root_position_m']).max() < 1e-5
        assert abs(roots[:, 3:7]-[1., 0., 0., 0.]).max() < 1e-5 and abs(roots[:, 7:]).max() < 1e-5
        assert abs(np.asarray(ground['joint_velocity_rad_s'])).max() < 1e-5
        assert ground['all_episode_clocks_zero'] and ground['source_training_indices'] == list(range(128))
    return dict(checked_at=datetime.now(timezone.utc).isoformat(), actual_iteration=iteration,
        diagnostic_sha256=core.sha(path), fresh_initial_actor_and_optimizer_checked=True,
        actual_actor_unchanged=equal, actual_Adam_steps=adam, actual_normalizer_count=steps,
        all128_first_episodes_completed=iteration == 7, diagnostic_is_canonical_restart=False,
        formal_acceptance=False)


def verify_training(base_path):
    import torch
    from quadsmash.evaluation_v4 import independent_episode
    from scripts.initialize_centered_policy_v4 import same_tree
    job = verify_job(TRAIN, STEPS)
    native = torch.load(TRAIN/'checkpoint.pt', map_location='cpu', weights_only=False)
    assert native['iter'] == UPDATES and native['native']['canonical_restart'] and native['native']['total_timesteps'] == STEPS
    assert not native['native']['environment']['tensors']['episode_length_buf'].any()
    state, adam = native['model_state_dict'], optimizer_steps(native)
    assert int(state['actor_obs_normalizer.count']) == int(state['critic_obs_normalizer.count']) == STEPS
    assert len(adam) == 17 and all(value == (300 if name.startswith('critic.') else 160) for name, value in adam.items())
    assert all(torch.isfinite(value).all() for value in state.values() if torch.is_tensor(value))
    assert all(torch.isfinite(value).all() for item in native['optimizer_state_dict']['state'].values()
               for value in item.values() if torch.is_tensor(value))
    metrics = core.read(TRAIN/'metrics.json')
    assert metrics['completed_updates'] == UPDATES and len(metrics['history']) == UPDATES
    assert [row['actor_warmup'] for row in metrics['history']] == [True]*7+[False]*8
    public = torch.load(TRAIN/'policy.pt', map_location='cpu', weights_only=False)
    assert public['schema'] == 'full-racket-origin0-residual-ppo-v4' and public['native_execution'] and not public['performance_inherited']
    assert same_tree(public['base_controller'], torch.load(base_path, map_location='cpu', weights_only=False))
    assert core.sha(TRAIN/'policy.pt') == metrics['policy_sha256'] and public['protocol_sha256'] == core.sha(PROTOCOL)
    assert public['residual_training_metadata']['export_actor_max_error'] <= 2e-5
    protocol = core.read(PROTOCOL); protocol['_root'] = str(ROOT)
    rows, seen, outcomes = core.read(TRAIN/'episodes.json'), [0]*128, []
    for row in rows:
        lane = row['env_id']
        assert row['ground_init'] and row['source_training_index'] == lane and row['episode_index'] == seen[lane]
        seen[lane] += 1; outcomes.append(independent_episode(row, protocol))
    assert seen == native['native']['environment']['tensors']['episode_index'].tolist()
    counts = dict(completed_episodes=len(rows), air=sum(row['airborne_contact'] for row in outcomes),
        conditional_recovery_count=sum(row['conditional_recovery'] for row in outcomes),
        complete=sum(row['complete'] for row in outcomes), numerical_failures=sum(row['numerical_failures'] for row in rows))
    assert counts == metrics['counts'] and counts['numerical_failures'] == 0
    return dict(checked_at=datetime.now(timezone.utc).isoformat(), actual_control_env_steps=STEPS,
        actual_GPU_hours=job['charged_gpu_hours'], checkpoint_sha256=core.sha(TRAIN/'checkpoint.pt'),
        policy_sha256=core.sha(TRAIN/'policy.pt'), independent_changing_policy_counts=counts,
        source_teacher_exact=True, all128_episode_chains_checked=True, actor_Adam_steps=160, critic_Adam_steps=300,
        frozen_policy_performance=False, formal_acceptance=False)


def verify_cold_commands(policy_path, metadata_path):
    import numpy as np
    import torch
    from quadsmash.control.full_racket_origin0_residual_v4 import make_rally_controller
    metadata = core.read(metadata_path)
    default, limits = [torch.tensor(metadata[key], dtype=torch.float32) for key in
                       ['default_joint_position_rad', 'joint_limits_rad']]
    policy = torch.load(policy_path, map_location='cpu', weights_only=False)
    controller = make_rally_controller(policy, 2, 'cpu')
    with np.load(FROZEN/'public_and_actions_first2.npz') as arrays:
        public, actions, valid = [arrays[key].copy() for key in ['observations', 'actions', 'valid_first_episode']]
    maximum = 0.
    # The actual fixed evaluator does not reset its controller after done.
    # Preserve that behavior rather than renumbering/resynthesizing episodes.
    with torch.inference_mode():
        for observation, actual, keep in zip(public, actions, valid):
            command = controller.action({'policy':torch.from_numpy(observation)}, default, limits).numpy()
            if keep.any():
                maximum = max(maximum, float(abs(command[keep]-actual[keep]).max()))
    assert maximum <= 5e-4
    return dict(checked_at=datetime.now(timezone.utc).isoformat(), actual_valid_public_rows=int(valid.sum()),
        CPU_vs_actual_GPU_max_action_error=maximum, original_action_tolerance=5e-4,
        policy_sha256=core.sha(policy_path), raw_public_sha256=core.sha(FROZEN/'public_and_actions_first2.npz'),
        formal_cases_read=False, formal_acceptance=False)


def main():
    assert PLAN.exists() and not STATE.exists()
    plan = core.read(PLAN); expected = dict(plan['source_sha256']); expected[str(PLAN)] = core.sha(PLAN)
    check(expected)
    write('WAITING_FOR_ACTUAL_GAIN_BODY_OWNER_EXIT', wait_parent_pid=PREVIOUS_PID, native_job_started=False)
    while core.live(PREVIOUS_PID, 'gain_body_after_prepose_queue02/run.py'):
        assert not stopping
        time.sleep(5)
    state = core.read(PREVIOUS/'state.json')['state']
    if state != plan['required_parent_terminal_state']:
        write('UPSTREAM_GAIN_BODY_NOT_COMPLETE', actual_upstream_terminal_state=state, new_native_job_started=False)
        return
    check(expected)
    admission = core.read(PREVIOUS/'ADMISSION_DECISION.json'); check(admission['source_sha256'])
    expected.update(admission['source_sha256'])
    selected = core.read(PREVIOUS/'BEST_MEASURED_SOURCE.json')['selected']
    base_path, evaluation = Path(selected['policy']), Path(selected['evaluation'])
    assert verify_current_evaluation(evaluation, base_path) == selected['summary']
    for name in INSTALL:
        path = ROOT/name
        assert not path.exists(), 'Never overwrite an existing source or user file'
        with path.open('xb') as stream:
            stream.write((ISOLATED/name).read_bytes())
        expected[str(path)] = core.sha(path)
    for path in sorted(ROOT.glob('quadsmash/**/*.py'))+sorted(ROOT.glob('scripts/*.py')):
        expected[str(path)] = core.sha(path)
    metadata = REPORTS/'gain_body8_sequential_n128_01/INPUTS.json'
    for path in [base_path, metadata, *[evaluation/name for name in
                 ['INPUTS.json', 'metrics.json', 'episodes.json', 'source_hashes.json', 'origin_layout.json',
                  'public_and_actions_first2.npz', 'job.json', 'completed.json']]]:
        expected[str(path)] = core.sha(path)
    write('VERIFYING_ACTUAL_SELECTED_ORIGIN0_SOURCE', measured_source=selected)
    run_cpu('scripts/check_full_racket_origin0_residual_selected_cpu_v4.py',
        ['--source', str(base_path), '--public-data', str(evaluation/'public_and_actions_first2.npz'),
         '--command-metadata', str(metadata), '--cold-evaluation', str(evaluation),
         '--output-dir', str(HERE/'ACTUAL_SOURCE_CPU')], 'actual_selected_source_CPU')
    proof = core.read(HERE/'ACTUAL_SOURCE_CPU/CPU_CHECK.json')
    assert proof['actual_cold_source_admission'] is not None and not proof['positive_completed_origin0_cold_source_verification_pending']
    assert proof['new_zero_residual_vs_unchanged_source_max_action_error'] == 0.
    assert proof['CPU_vs_recorded_origin0_GPU_max_action_error'] <= 5e-4
    expected[str(HERE/'ACTUAL_SOURCE_CPU/CPU_CHECK.json')] = core.sha(HERE/'ACTUAL_SOURCE_CPU/CPU_CHECK.json')
    save(HERE/'ADMISSION_DECISION.json', dict(checked_at=datetime.now(timezone.utc).isoformat(),
        measured_source=selected, source_sha256=expected, actual_selected_source_CPU=proof,
        previous_owner_actually_exited=True, original_target_retained=True, formal_acceptance=False))
    run_job(TRAIN, 'scripts/train_full_racket_origin0_residual_ppo_v4.py',
        ['--base-checkpoint', str(base_path), '--body-source-evaluation', str(evaluation),
         '--precision-receipt', str(PRECISION_RECEIPT), '--config', str(CONFIG),
         '--num-envs', '128', '--updates', '15', '--seed', '751001'],
        STEPS, 7200, 'RUNNING_FRESH_ORIGIN0_RESIDUAL15', expected)
    for iteration in [1, 7, 15]:
        if not (HERE/f'ACTUAL_UPDATE_{iteration:02d}.json').exists():
            save(HERE/f'ACTUAL_UPDATE_{iteration:02d}.json', verify_update(iteration, base_path))
    save(HERE/'TRAINING_RECEIPT.json', verify_training(base_path))
    policy = TRAIN/'policy.pt'; expected[str(policy)] = core.sha(policy)
    run_job(FROZEN, 'scripts/evaluate_full_racket_origin0_residual_train_v4.py',
        ['--checkpoint', str(policy), '--noise', 'none', '--seed', '724201', '--precision-receipt', str(PRECISION_RECEIPT)],
        102400, 2700, 'RUNNING_FROZEN_ORIGIN0_RESIDUAL15_TRAIN128', expected)
    measured = verify_current_evaluation(FROZEN, policy)
    save(HERE/'COLD_TRAIN128_RESULT.json', measured)
    save(HERE/'COLD_CPU_COMMAND_CHECK.json', verify_cold_commands(policy, metadata))
    run_cpu('scripts/export_full_racket_origin0_train_csv_v4.py',
        ['--run', str(FROZEN), '--output-prefix', str(HERE/'COLD128_EPISODES')], 'csv_export')
    best = dict(policy=str(policy), evaluation=str(FROZEN), policy_sha256=core.sha(policy), summary=measured)
    save(HERE/'MEASURED_RESULT.json', best)
    check(expected)
    write('COMPLETED_ORIGIN0_RESIDUAL15_AND_COLD128', measured=best,
        ready_for_formal_source_review=ready(measured), source_custody_released_on_actual_parent_exit=True)


def stop(*_):
    global stopping
    stopping = True
    if child is not None and child.poll() is None:
        child.send_signal(signal.SIGTERM)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true'); args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        with (HERE/'observer.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
            try:
                main()
            except BaseException:
                if child is not None and child.poll() is None:
                    child.send_signal(signal.SIGTERM); child.wait(timeout=60)
                write('STOPPED' if stopping else 'FAILED', error=traceback.format_exc(), automatic_retry=False)
                raise


