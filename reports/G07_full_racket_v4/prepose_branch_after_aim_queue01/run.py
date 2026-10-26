"""Wait for actual Aim7 owner exit, then execute preparation-branch comparison.

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
PREVIOUS = REPORTS/'contact_aim_after_body48_queue01'
PREVIOUS_PID = 1827656
ISOLATED = ROOT.parent/'unitree_smash_landing_targets'
PYTHON = '/home/autodrive/Isaac/env_isaaclab/bin/python'
SEARCH = REPORTS/'prepose_branch8_n1024_01'
FROZEN = REPORTS/'prepose_branch8_best_fixed_train128_01'
PLAN, STATE = HERE/'PLAN.json', HERE/'state.json'
INSTALL = ['quadsmash/control/full_racket_prepose_branch_v4.py',
           'scripts/prepare_full_racket_prepose_branch_v4.py',
           'scripts/check_full_racket_prepose_branch_cpu_v4.py',
           'scripts/search_full_racket_prepose_branch_v4.py',
           'scripts/evaluate_full_racket_prepose_branch_train_v4.py']
PROVENANCE = REPORTS/'body_residual_leg15_fresh_n2048_updates48_01/trajectory_traces/manifest.json'
STATIC_BANK = REPORTS/'near_net_full_pose_bank_20261027/RESULT.json'
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
    assert core.live(PREVIOUS_PID, 'contact_aim_after_body48_queue01/run.py')
    expected = dict(core.read(PREVIOUS/'ADMISSION_DECISION.json')['source_sha256'])
    check(expected)
    proof_dir = ISOLATED/'reports/G07_full_racket_v4/prepose_branch_cpu_20261027'
    proof = core.read(proof_dir/'CPU_CHECK.json')
    assert proof['exact_reference_max_action_error'] == proof['grouped_vs_separate_max_action_error'] == 0
    assert proof['maximum_slew_excess_rad'] <= 2e-7 and not proof['new_native_execution']
    check(proof['source_sha256'])
    for name in INSTALL:
        assert (ISOLATED/name).is_file() and not (ROOT/name).exists()
        expected[str(ISOLATED/name)] = core.sha(ISOLATED/name)
    for path in [Path(__file__), proof_dir/'CPU_CHECK.json', proof_dir/'OLD_SOURCE_PROPOSALS.json',
                 PROVENANCE, STATIC_BANK, REPORTS/'residual_resume24_queue01/run.py']:
        expected[str(path)] = core.sha(path)
    save(PLAN, dict(created_at=datetime.now(timezone.utc).isoformat(), source_sha256=expected,
        wait_actual_parent_pid=PREVIOUS_PID, required_parent_terminal_state='COMPLETED_CONTACT_AIM_SEARCH_AND_COLD128',
        source_choice='Prior owner selected cold128 policy; skip further search if actual50/70/85 and NF0 readiness passes.',
        hypothesis='A measured high contact pose and an equivalent wrist branch shorten preparation; compare speed and walking targets through unchanged native dynamics.',
        proposals='Regenerate8 shared constants from completed TRAIN contacts; no runtime case lookup or formal cases.',
        native_comparison='8 candidates x original128 ground cases in1024 environments; original scoring and no reset noise.',
        search_env_steps_cap=819200, search_max_seconds=5400, search_seed=750501,
        cold_env_steps_cap=102400, cold_max_seconds=2700, cold_seed=724201,
        install_only_new_files_after_actual_parent_exit=INSTALL,
        prior419_inputs_unchanged=True, original_target_hardware_physics_and_ground_reset_retained=True,
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
    return core.verify_job(output, cap)


def verify_search(source_path, proposal):
    import numpy as np
    import torch
    from quadsmash.control.full_racket_prepose_branch_v4 import PreposeBranchControllerV4, SCHEMA
    from quadsmash.control.full_racket_body_search_v4 import summarize_candidate
    from quadsmash.control.full_racket_train_evaluation_v4 import training_cases
    from scripts.initialize_centered_policy_v4 import same_tree
    source = torch.load(source_path, map_location='cpu', weights_only=False)
    protocol = core.read(ROOT/'configs/g07_full_racket_eval_v4_r3.json'); protocol['_root'] = str(ROOT)
    cases, indices = training_cases(ROOT, protocol)
    inputs, metrics = core.read(SEARCH/'INPUTS.json'), core.read(SEARCH/'metrics.json')
    assert inputs['source_policy_sha256'] == core.sha(source_path) and inputs['training_indices'] == indices
    assert inputs['candidates'] == proposal['candidates'] and inputs['num_envs'] == 1024
    assert inputs['proposal_sha256'] == core.sha(HERE/'ACTUAL_PROPOSALS.json')
    rows, initial = core.read(SEARCH/'episodes.json'), core.read(SEARCH/'initial_ground_states.json')
    assert len(rows) == 1024 and initial['all_clocks_zero']
    roots = np.asarray(initial['root_state_local'])
    assert abs(roots[:, :3]-protocol['initialization']['root_position_m']).max() < 1e-5
    assert abs(roots[:, 3:7]-[1., 0., 0., 0.]).max() < 1e-5 and abs(roots[:, 7:]).max() < 1e-5
    for lane, row in enumerate(rows):
        assert row['env_id'] == row['jump']['env_id'] == lane
        assert row['candidate_id'] == lane//128 and row['source_training_index'] == lane % 128
        assert row['ground_init'] and row['case_id'] == cases[lane % 128]['case_id']
        assert row['episode_index'] == initial['episode_ids'][lane]
    summaries = [summarize_candidate(rows[group*128:(group+1)*128], protocol) for group in range(8)]
    assert summaries == metrics['summaries'] and all(summary['numerical_failures'] == 0 for summary in summaries)
    ranked = sorted(range(8), key=lambda group: tuple(summaries[group]['rank']), reverse=True)
    assert ranked == metrics['ranked_candidate_indices'] and ranked[0] == metrics['best_candidate_id']
    default, limits = [torch.tensor(inputs[key], dtype=torch.float32) for key in
                       ('default_joint_position_rad', 'joint_limits_rad')]
    with np.load(SEARCH/'public_and_actions.npz') as arrays:
        public, actions, valid, stream_ids = [arrays[key].copy() for key in
            ('observations', 'actions', 'valid_first_episode', 'native_env_ids')]
    assert stream_ids.tolist() == [128*group+i for group in range(8) for i in range(2)]
    receipts = []
    for group, candidate in enumerate(proposal['candidates']):
        controller = PreposeBranchControllerV4(source, candidate['overrides'], 2, 'cpu')
        maximum = 0.; previous = None
        with torch.inference_mode():
            for obs, actual, keep in zip(public[:, 2*group:2*group+2], actions[:, 2*group:2*group+2], valid[:, 2*group:2*group+2]):
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
        receipts.append(dict(candidate_id=group, actual_valid_public_rows=int(valid[:, 2*group:2*group+2].sum()),
            CPU_vs_actual_GPU_max_action_error=maximum, summary=summaries[group]))
    policy = torch.load(SEARCH/'best_candidate.pt', map_location='cpu', weights_only=False)
    assert policy['schema'] == SCHEMA and policy['native_execution'] and not policy['performance_inherited']
    assert same_tree(policy['source_public_policy'], source) and torch.equal(policy['parameters'], source['parameters'])
    assert policy['runtime_overrides'] == proposal['candidates'][ranked[0]]['overrides']
    assert policy['actual_native_search']['candidate_id'] == ranked[0]
    assert metrics['actual_control_env_steps'] == core.read(SEARCH/'completed.json')['env_steps'] == 819200
    return dict(checked_at=datetime.now(timezone.utc).isoformat(), all1024_native_rows_independently_checked=True,
        actual_control_env_steps=819200, original_source_checkpoint_tree_unchanged=True,
        actual_public_replays=receipts, best_candidate_id=ranked[0], formal_cases_read=False, formal_acceptance=False)


def main():
    assert PLAN.exists() and not STATE.exists()
    plan = core.read(PLAN); expected = dict(plan['source_sha256']); expected[str(PLAN)] = core.sha(PLAN)
    check(expected)
    write('WAITING_FOR_ACTUAL_AIM_OWNER_EXIT', wait_parent_pid=PREVIOUS_PID, native_job_started=False)
    while core.live(PREVIOUS_PID, 'contact_aim_after_body48_queue01/run.py'):
        assert not stopping
        time.sleep(5)
    assert core.read(PREVIOUS/'state.json')['state'] == plan['required_parent_terminal_state']
    check(expected)
    selected = core.read(PREVIOUS/'BEST_MEASURED_SOURCE.json')['selected']
    source_path, evaluation = Path(selected['policy']), Path(selected['evaluation'])
    summary = core.verify_evaluation(evaluation, source_path); assert summary == selected['summary']
    if ready(summary):
        write('READY_FOR_FORMAL_SOURCE_REVIEW', selected=selected, further_search_skipped=True)
        return
    for name in INSTALL:
        path = ROOT/name
        assert not path.exists(), 'Never overwrite an existing runtime or user file'
        with path.open('xb') as stream:
            stream.write((ISOLATED/name).read_bytes())
        expected[str(path)] = core.sha(path)
    for path in sorted(ROOT.glob('quadsmash/**/*.py'))+sorted(ROOT.glob('scripts/*.py')):
        expected[str(path)] = core.sha(path)
    for path in [source_path, *[evaluation/name for name in
        ['INPUTS.json','metrics.json','episodes.json','contacts.json','job.json','completed.json','public_and_actions_first2.npz']]]:
        expected[str(path)] = core.sha(path)
    write('PREPARING_ACTUAL_SELECTED_SOURCE_PROPOSALS', selected=selected)
    run_cpu('scripts/prepare_full_racket_prepose_branch_v4.py',
        ['--source', str(source_path), '--source-evaluation', str(evaluation), '--static-bank', str(STATIC_BANK),
         '--output', str(HERE/'ACTUAL_PROPOSALS.json')], 'actual_proposals')
    proposal = core.read(HERE/'ACTUAL_PROPOSALS.json'); assert len(proposal['candidates']) == 8
    run_cpu('scripts/check_full_racket_prepose_branch_cpu_v4.py',
        ['--proposals', str(HERE/'ACTUAL_PROPOSALS.json'), '--provenance', str(PROVENANCE),
         '--output', str(HERE/'ACTUAL_SELECTED_SOURCE_COMMAND_CHECK.json')], 'actual_source_commands')
    for path in [HERE/'ACTUAL_PROPOSALS.json', HERE/'ACTUAL_SELECTED_SOURCE_COMMAND_CHECK.json']:
        expected[str(path)] = core.sha(path)
    save(HERE/'ADMISSION_DECISION.json', dict(checked_at=datetime.now(timezone.utc).isoformat(), selected=selected,
        source_sha256=expected, previous_owner_actually_exited=True, original_target_retained=True, formal_acceptance=False))
    run_job(SEARCH, 'scripts/search_full_racket_prepose_branch_v4.py',
        ['--proposals', str(HERE/'ACTUAL_PROPOSALS.json'), '--seed', '750501'], 819200, 5400,
        'RUNNING_NATIVE_PREPOSE_BRANCH8', expected)
    write('VERIFYING_NATIVE_PREPOSE_BRANCH8')
    save(HERE/'SEARCH_RECEIPT.json', verify_search(source_path, proposal))
    policy = SEARCH/'best_candidate.pt'; expected[str(policy)] = core.sha(policy)
    run_job(FROZEN, 'scripts/evaluate_full_racket_prepose_branch_train_v4.py',
        ['--checkpoint', str(policy), '--noise', 'none', '--seed', '724201'], 102400, 2700,
        'RUNNING_FIXED_PREPOSE_BRANCH_TRAIN128', expected)
    measured = core.verify_evaluation(FROZEN, policy)
    save(HERE/'COLD_TRAIN128_RESULT.json', measured)
    run_cpu('scripts/export_full_racket_train_csv_v4.py',
        ['--run', str(FROZEN), '--output-prefix', str(HERE/'COLD128_EPISODES')], 'csv_export')
    key = lambda result: (result['complete'], result['conditional_recovery_count'], result['airborne_contacts'], result['attack_outgoing'])
    best = selected
    if key(measured) > key(summary):
        best = dict(candidate_id='prepose_branch8', policy=str(policy), evaluation=str(FROZEN),
                    policy_sha256=core.sha(policy), summary=measured)
    save(HERE/'BEST_MEASURED_SOURCE.json', dict(selected=best, prior=selected,
        new_measured=dict(policy=str(policy), evaluation=str(FROZEN), summary=measured),
        formal_cases_read=False, formal_acceptance=False))
    check(expected)
    write('COMPLETED_PREPOSE_BRANCH_SEARCH_AND_COLD128', selected=best,
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
