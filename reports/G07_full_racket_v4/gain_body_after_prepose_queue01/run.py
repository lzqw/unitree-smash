"""After the measured prepose comparison, adapt body timing and recovery with two residual controls.

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
PREVIOUS = REPORTS/'prepose_origin0_after_precision_queue02'
PRECISION_RECEIPT = REPORTS/'origin_precision_after_aim_queue01/PRECISION_ADMISSION.json'
PROTOCOL = ROOT/'configs/g07_full_racket_eval_v4_r3_origin0.json'
PREVIOUS_PID = 1884350
ISOLATED = ROOT.parent/'unitree_smash_landing_targets'
PYTHON = '/home/autodrive/Isaac/env_isaaclab/bin/python'
SEARCH = REPORTS/'gain_body8_sequential_n128_01'
FROZEN = REPORTS/'gain_body8_best_fixed_train128_01'
PLAN, STATE = HERE/'PLAN.json', HERE/'state.json'
INSTALL = ['quadsmash/control/full_racket_residual_gain_v4.py',
           'quadsmash/control/full_racket_gain_body_adaptation_v4.py',
           'scripts/prepare_full_racket_gain_body_v4.py',
           'scripts/check_full_racket_gain_body_cpu_v4.py',
           'scripts/search_full_racket_gain_body_origin0_v4.py',
           'scripts/evaluate_full_racket_gain_body_origin0_train_v4.py']
PROVENANCE = REPORTS/'body_residual_leg15_fresh_n2048_updates48_01/trajectory_traces/manifest.json'
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


def prepare():
    assert ROOT.name == 'unitree_smash' and not any(path.exists() for path in [PLAN, STATE, SEARCH, FROZEN])
    assert core.live(PREVIOUS_PID, 'prepose_origin0_after_precision_queue02/run.py') or core.read(PREVIOUS/'state.json')['state'] == 'COMPLETED_PREPOSE_ORIGIN0_SEARCH_AND_COLD128'
    expected = dict(core.read(PREVIOUS/'ADMISSION_DECISION.json')['source_sha256'])
    check(expected)
    proof_dir = ISOLATED/'reports/G07_full_racket_v4/origin0_gain_body_candidate_20261027'
    proof = core.read(proof_dir/'CPU_CHECK.json')
    assert proof['exact_reference_max_action_error'] == proof['zero_layers_vs_analytic_max_action_error'] == 0
    assert proof['CPU_vs_recorded_source_GPU_max_action_error'] <= 5e-4 and not proof['new_native_execution']
    check(proof['source_sha256'])
    bounds = core.read(proof_dir/'BODY_BOUNDS_AND_RESET_CHECK.json')
    assert bounds['candidates_after_causal_planning_checked'] == 8 and bounds['partial_reset_preserves_other_lane']
    for name in INSTALL:
        assert (ISOLATED/name).is_file() and not (ROOT/name).exists()
        expected[str(ISOLATED/name)] = core.sha(ISOLATED/name)
    for path in [Path(__file__), PRECISION_RECEIPT, PROTOCOL, PROVENANCE,
                 proof_dir/'CPU_CHECK.json', proof_dir/'PROPOSALS.json', proof_dir/'BODY_BOUNDS_AND_RESET_CHECK.json',
                 proof_dir/'check_bounds_and_reset.py', PREVIOUS/'PLAN.json',
                 REPORTS/'residual_resume24_queue01/run.py']:
        expected[str(path)] = core.sha(path)
    save(PLAN, dict(created_at=datetime.now(timezone.utc).isoformat(), source_sha256=expected,
        wait_actual_parent_pid=PREVIOUS_PID, required_parent_terminal_state='COMPLETED_PREPOSE_ORIGIN0_SEARCH_AND_COLD128',
        source_choice='Bind the actual prepose winner only after its original all128 cold assessment and real owner exit; no historical score inherited.',
        hypothesis='Actual preparation has root tilt19.14deg before launch, while old residual leg requests are small. Compare two exact source/residual controls plus six walking, jump phase and recovery-handover adaptations.',
        proposals='Eight fixed shared candidates: source, all residuals off, jump plus/minus40ms with handover preserved, recovery phase plus/minus0.15, walk plus0.10m with both jump directions. No case-ID input.',
        native_comparison='8 sequential128-lane normal ground resets, all original128 cases each, admitted origin0.',
        search_env_steps_cap=819200, search_max_seconds=14400, search_seed=750901,
        cold_env_steps_cap=102400, cold_max_seconds=2700, cold_seed=724201,
        install_only_new_files_after_actual_parent_exit=INSTALL,
        original_target_hardware_physics_and_ground_reset_retained=True,
        automatic_retry=False, formal_cases_read=False, formal_acceptance=False, goal_complete=False))
    print(json.dumps(dict(prepared=str(PLAN), bound_inputs=len(expected), native_job_started=False)))


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


def verify_search(source_path, proposal):
    import numpy as np
    import torch
    from quadsmash.control.full_racket_gain_body_adaptation_v4 import GainBodyAdaptationControllerV4, SCHEMA
    from quadsmash.control.full_racket_body_search_v4 import summarize_candidate
    from quadsmash.control.full_racket_train_evaluation_v4 import training_cases
    from scripts.initialize_centered_policy_v4 import same_tree
    source = torch.load(source_path, map_location='cpu', weights_only=False)
    protocol = core.read(PROTOCOL); protocol['_root'] = str(ROOT)
    cases, indices = training_cases(ROOT, protocol)
    inputs, metrics = core.read(SEARCH/'INPUTS.json'), core.read(SEARCH/'metrics.json')
    assert inputs['source_policy_sha256'] == core.sha(source_path) and inputs['training_indices'] == indices
    assert inputs['candidate_gain_body_definitions'] == proposal['candidates'] and inputs['num_envs'] == 128
    assert inputs['protocol_sha256'] == core.sha(PROTOCOL) and inputs['origin_spacing_m'] == 0.
    assert inputs['origin_precision_receipt_sha256'] == core.sha(PRECISION_RECEIPT)
    assert inputs['proposal_sha256'] == core.sha(HERE/'ACTUAL_PROPOSALS.json') and len(metrics['history']) == 8
    default, limits = [torch.tensor(inputs[key], dtype=torch.float32) for key in
                       ('default_joint_position_rad', 'joint_limits_rad')]
    receipts = []; best = None; steps = 0
    for group, (candidate, entry) in enumerate(zip(proposal['candidates'], metrics['history'])):
        rows = core.read(SEARCH/f'episodes_{group:03d}.json')
        initial = core.read(SEARCH/f'initial_ground_states_{group:03d}.json')
        assert len(rows) == 128 and initial['all_clocks_zero']
        origins = np.asarray(initial['actual_environment_origins_m'])
        roots = np.asarray(initial['root_state_local'])
        assert origins.shape == (128, 3) and not origins.any()
        assert abs(roots[:, :3]-protocol['initialization']['root_position_m']).max() < 1e-5
        assert abs(roots[:, 3:7]-[1., 0., 0., 0.]).max() < 1e-5 and abs(roots[:, 7:]).max() < 1e-5
        for lane, row in enumerate(rows):
            assert row['env_id'] == row['jump']['env_id'] == lane
            assert row['candidate_id'] == group and row['source_training_index'] == lane and row['ground_init']
            assert row['case_id'] == cases[lane]['case_id'] and row['episode_index'] == initial['episode_ids'][lane]
        summary = summarize_candidate(rows, protocol)
        assert summary == entry['summary'] and summary['numerical_failures'] == 0
        assert entry['candidate_id'] == group and entry['candidate'] == candidate
        assert entry['env_steps'] == entry['control_ticks']*128 and 0 < entry['control_ticks'] <= 800
        steps += entry['env_steps']; rank = tuple(summary['rank'])
        if best is None or rank > best[0]:
            best = rank, group
        with np.load(SEARCH/f'public_and_actions_{group:03d}.npz') as arrays:
            public, actions, valid = [arrays[key].copy() for key in ('observations', 'actions', 'valid_first_episode')]
        controller = GainBodyAdaptationControllerV4(source, candidate['gains'], 2, 'cpu', candidate['body_deltas'])
        maximum = 0.; previous = None
        with torch.inference_mode():
            for obs, actual, keep in zip(public, actions, valid):
                observation = torch.from_numpy(obs)
                if previous is not None:
                    reset = torch.where(observation[:, 775] < previous-1e-5)[0]
                    if len(reset):
                        controller.reset(reset)
                command = controller.action({'policy': observation}, default, limits).numpy()
                if keep.any():
                    maximum = max(maximum, float(abs(command[keep]-actual[keep]).max()))
                previous = observation[:, 775].clone()
        assert maximum <= 5e-4, 'Preserve actual command mismatch'
        receipts.append(dict(candidate_id=group, actual_valid_public_rows=int(valid.sum()),
            CPU_vs_actual_GPU_max_action_error=maximum, summary=summary))
    policy = torch.load(SEARCH/'best_candidate.pt', map_location='cpu', weights_only=False)
    assert policy['schema'] == SCHEMA and policy['native_execution'] and not policy['performance_inherited']
    assert same_tree(policy['source_public_policy'], source) and torch.equal(policy['parameters'], source['parameters'])
    assert policy['residual_gains_outer_to_inner'] == proposal['candidates'][best[1]]['gains']
    assert policy['runtime_body_deltas'] == proposal['candidates'][best[1]]['body_deltas']
    assert policy['actual_native_search']['candidate_id'] == best[1]
    assert policy['numerical_execution'] == protocol['numerical_execution'] and policy['protocol_sha256'] == core.sha(PROTOCOL)
    assert metrics['actual_control_env_steps'] == core.read(SEARCH/'completed.json')['env_steps'] == steps
    return dict(checked_at=datetime.now(timezone.utc).isoformat(), all1024_native_rows_independently_checked=True,
        actual_control_env_steps=steps, guard_conservatively_charged_env_steps_cap=819200,
        original_source_checkpoint_tree_unchanged=True, actual_public_replays=receipts,
        best_candidate_id=best[1], source_performance_inherited=False, formal_cases_read=False, formal_acceptance=False)

def main():
    assert PLAN.exists() and not STATE.exists()
    plan = core.read(PLAN); expected = dict(plan['source_sha256']); expected[str(PLAN)] = core.sha(PLAN)
    check(expected)
    write('WAITING_FOR_ACTUAL_PREPOSE_OWNER_EXIT', wait_parent_pid=PREVIOUS_PID, native_job_started=False)
    while core.live(PREVIOUS_PID, 'prepose_origin0_after_precision_queue02/run.py'):
        assert not stopping
        time.sleep(5)
    state = core.read(PREVIOUS/'state.json')['state']
    if state != plan['required_parent_terminal_state']:
        write('UPSTREAM_PREPOSE_NOT_COMPLETE', actual_upstream_terminal_state=state, new_native_job_started=False)
        return
    check(expected)
    from quadsmash.tasks.stand.full_racket_origin0_v4 import validate_admission, validate_protocol
    assert validate_admission(PRECISION_RECEIPT)['admitted']
    validate_protocol(core.read(PROTOCOL))
    selected = core.read(PREVIOUS/'BEST_MEASURED_SOURCE.json')['selected']
    source_path, evaluation = Path(selected['policy']), Path(selected['evaluation'])
    prior_summary = verify_current_evaluation(evaluation, source_path)
    assert prior_summary == selected['summary']
    for name in INSTALL:
        path = ROOT/name
        assert not path.exists(), 'Never overwrite an existing source or user file'
        with path.open('xb') as stream:
            stream.write((ISOLATED/name).read_bytes())
        expected[str(path)] = core.sha(path)
    for path in sorted(ROOT.glob('quadsmash/**/*.py'))+sorted(ROOT.glob('scripts/*.py')):
        expected[str(path)] = core.sha(path)
    for path in [source_path, *[evaluation/name for name in
        ['INPUTS.json','metrics.json','episodes.json','contacts.json','job.json','completed.json','public_and_actions_first2.npz','origin_layout.json']]]:
        expected[str(path)] = core.sha(path)
    write('PREPARING_ACTUAL_SOURCE_GAIN_BODY_ADAPTATION', measured_source=selected, source_performance_inherited=False)
    run_cpu('quadsmash/control/full_racket_gain_body_adaptation_v4.py',
           'scripts/prepare_full_racket_gain_body_v4.py',
        ['--source', str(source_path), '--source-evaluation', str(evaluation),
         '--output', str(HERE/'ACTUAL_PROPOSALS.json')], 'actual_proposals')
    proposal = core.read(HERE/'ACTUAL_PROPOSALS.json'); assert len(proposal['candidates']) == 8
    run_cpu('scripts/check_full_racket_gain_body_cpu_v4.py',
        ['--proposals', str(HERE/'ACTUAL_PROPOSALS.json'), '--provenance', str(PROVENANCE),
         '--output', str(HERE/'ACTUAL_SELECTED_SOURCE_COMMAND_CHECK.json')], 'actual_source_commands')
    for path in [HERE/'ACTUAL_PROPOSALS.json', HERE/'ACTUAL_SELECTED_SOURCE_COMMAND_CHECK.json']:
        expected[str(path)] = core.sha(path)
    save(HERE/'ADMISSION_DECISION.json', dict(checked_at=datetime.now(timezone.utc).isoformat(), measured_source=selected,
        source_sha256=expected, previous_owner_actually_exited=True, original_target_retained=True,
        origin_precision_admission_sha256=core.sha(PRECISION_RECEIPT), source_performance_inherited=False, formal_acceptance=False))
    run_job(SEARCH, 'scripts/search_full_racket_gain_body_origin0_v4.py',
        ['--source', str(source_path), '--source-evaluation', str(evaluation), '--proposals', str(HERE/'ACTUAL_PROPOSALS.json'),
         '--precision-receipt', str(PRECISION_RECEIPT), '--seed', '750901'], 819200, 14400,
        'RUNNING_NATIVE_GAIN_BODY_ORIGIN0_8', expected)
    write('VERIFYING_NATIVE_GAIN_BODY_ORIGIN0_8')
    save(HERE/'SEARCH_RECEIPT.json', verify_search(source_path, proposal))
    policy = SEARCH/'best_candidate.pt'; expected[str(policy)] = core.sha(policy)
    run_job(FROZEN, 'scripts/evaluate_full_racket_gain_body_origin0_train_v4.py',
        ['--checkpoint', str(policy), '--noise', 'none', '--seed', '724201', '--precision-receipt', str(PRECISION_RECEIPT)],
        102400, 2700, 'RUNNING_FIXED_GAIN_BODY_ORIGIN0_TRAIN128', expected)
    measured = verify_current_evaluation(FROZEN, policy)
    save(HERE/'COLD_TRAIN128_RESULT.json', measured)
    run_cpu('scripts/export_full_racket_origin0_train_csv_v4.py',
        ['--run', str(FROZEN), '--output-prefix', str(HERE/'COLD128_EPISODES')], 'csv_export')
    best = dict(candidate_id='gain_body_origin0_8', policy=str(policy), evaluation=str(FROZEN),
                policy_sha256=core.sha(policy), summary=measured, protocol_sha256=core.sha(PROTOCOL))
    save(HERE/'BEST_MEASURED_SOURCE.json', dict(selected=best, measured_upstream_source=selected,
        source_performance_not_inherited=True, formal_cases_read=False, formal_acceptance=False))
    check(expected)
    write('COMPLETED_GAIN_BODY_ORIGIN0_SEARCH_AND_COLD128', selected=best,
        ready_for_formal_source_review=ready(best['summary']), source_custody_released_on_actual_parent_exit=True)

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
