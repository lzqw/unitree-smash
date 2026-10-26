"""Wait for the actual48-course owner, then measure interior string aims.

Only new files are installed, after the preceding owner's actual exit. One
shared main GPU0 guard runs the search and subsequent fixed TRAIN128. The
original target contract remains in force unless the user explicitly changes it.
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
PREVIOUS = REPORTS/'body_residual48_after_grid16_queue01'
PREVIOUS_PID = 1814619
ISOLATED = ROOT.parent/'unitree_smash_landing_targets'
PYTHON = '/home/autodrive/Isaac/env_isaaclab/bin/python'
SEARCH = REPORTS/'contact_aim7_sequential_n128_01'
FROZEN = REPORTS/'contact_aim7_best_fixed_train128_01'
PLAN, STATE = HERE/'PLAN.json', HERE/'state.json'
INSTALL = ['quadsmash/control/full_racket_contact_aim_v4.py',
           'scripts/search_full_racket_contact_aim_v4.py',
           'scripts/evaluate_full_racket_contact_aim_train_v4.py']
sys.path.insert(0,str(ROOT))
spec = importlib.util.spec_from_file_location('contact_aim_shared_guard_checks',
    REPORTS/'residual_resume24_queue01/run.py')
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
stopping, child = False, None


def save(path, value):
    assert not path.exists(), 'Preserve previous evidence: '+str(path)
    core.atomic(path,value)


def write(state, **extra):
    core.atomic(STATE,dict(state=state,queue_pid=os.getpid(),
        observed_at=datetime.now(timezone.utc).isoformat(),
        formal_cases_read=False,formal_acceptance=False,goal_complete=False,**extra))


def check(expected):
    for name,digest in expected.items():
        assert core.sha(Path(name)) == digest, name


def ready(summary):
    return (summary['complete_success_rate'] >= .5 and summary['airborne_contact_rate'] >= .7
        and summary['conditional_recovery_rate'] >= .85 and summary['numerical_failures'] == 0)


def prepare():
    assert ROOT.name == 'unitree_smash' and not any(p.exists() for p in [PLAN,STATE,SEARCH,FROZEN])
    assert core.live(PREVIOUS_PID,'body_residual48_after_grid16_queue01/run.py')
    admission = core.read(PREVIOUS/'ADMISSION_DECISION.json')
    expected = {str(ROOT/name):digest for name,digest in admission['source_hashes'].items()}
    check(expected)
    proof_dir = ISOLATED/'reports/G07_full_racket_v4/contact_aim_candidate_20261026'
    proof = core.read(proof_dir/'RESULT.json')
    assert proof['matched_reference_max_normalized_action_error'] == 0
    assert proof['physical_support_kinematics_max_error'] == 0 and not proof['new_native_execution']
    check(core.read(proof_dir/'PLAN.json')['source_sha256'])
    for name in INSTALL:
        assert (ISOLATED/name).is_file() and not (ROOT/name).exists()
        expected[str(ISOLATED/name)] = core.sha(ISOLATED/name)
    for path in [Path(__file__),proof_dir/'RESULT.json',proof_dir/'PLAN.json',
                 REPORTS/'residual_resume24_queue01/run.py']:
        expected[str(path)] = core.sha(path)
    check(expected)
    save(PLAN,dict(created_at=datetime.now(timezone.utc).isoformat(),source_sha256=expected,
        wait_actual_parent_pid=PREVIOUS_PID,
        required_parent_terminal_state='COMPLETED_FRESH_BODY_RESIDUAL48_AND_FIXED_COMPARISON',
        hypothesis='Reduce measured edge/frame contacts by changing the analytic string aim; preserve the selected measured source, all actors, body timing and physical geometry.',
        source_choice='Best actually measured source from all previous courses and the completed48 course; skip extra search if original readiness gates pass.',
        candidate_aims_uv_m=[[0,.1],[0,.08],[0,.06],[0,.04],[0,0],[-.015,.06],[.015,.06]],
        native_comparison='Seven sequential128-lane normal ground resets, original all128 cases each, preserve native episode IDs.',
        seed=750401,search_env_steps_cap=7*128*800,search_max_seconds=10800,
        cold_env_steps_cap=102400,cold_max_seconds=2700,cold_seed=724201,
        install_only_new_files_after_actual_parent_exit=INSTALL,
        original_target_and_all_physics_hardware_and_success_gates_retained=True,
        original_active407_inputs_not_modified=True,
        scope_change='User target-range preference remains pending. This experiment retains the original contract; no broadened score is substituted.',
        automatic_retry=False,formal_cases_read=False,formal_acceptance=False,goal_complete=False))
    print(json.dumps(dict(prepared=str(PLAN),bound_inputs=len(expected),native_job_started=False)))


def verify_source_commands(source_path, evaluation):
    import numpy as np
    import torch
    from quadsmash.control.full_racket_contact_aim_v4 import ContactAimControllerV4, aim_candidates
    from quadsmash.control.full_racket_body_residual_v4 import make_rally_controller
    source = torch.load(source_path,map_location='cpu',weights_only=False)
    with np.load(evaluation/'public_and_actions_first2.npz') as arrays:
        observations,actions,valid = [arrays[k].copy() for k in ('observations','actions','valid_first_episode')]
    provenance = core.read(REPORTS/'body_residual_leg15_fresh_n2048_updates48_01/trajectory_traces/manifest.json')['provenance']
    default,limits = [torch.tensor(provenance[k],dtype=torch.float32) for k in
        ('default_joint_position_rad','joint_limits_rad')]
    original = make_rally_controller(source,2,'cpu')
    candidates = [ContactAimControllerV4(source,uv,2,'cpu') for uv in aim_candidates()]
    max_reference = max_native = 0.
    with torch.inference_mode():
        for obs,actual,keep in zip(observations,actions,valid):
            public = torch.from_numpy(obs)
            reference = original.action({'policy':public},default,limits)
            if keep.any():
                max_native = max(max_native,float(abs(reference.numpy()[keep]-actual[keep]).max()))
            for i,candidate in enumerate(candidates):
                command = candidate.action({'policy':public},default,limits)
                assert command.shape == (2,18) and torch.isfinite(command).all() and (command.abs() <= 1).all()
                if i == 0 and keep.any():
                    max_reference = max(max_reference,float(abs(command.numpy()[keep]-reference.numpy()[keep]).max()))
    assert max_reference == 0 and max_native <= 5e-4
    return dict(checked_at=datetime.now(timezone.utc).isoformat(),actual_valid_public_rows=int(valid.sum()),
        source_policy_sha256=core.sha(source_path),exact_reference_action_error=max_reference,
        CPU_vs_actual_source_native_action_error=max_native,
        all7_new_commands_finite_and_bounded=True,new_native_execution=False,formal_acceptance=False)


def run_job(output, script, arguments, cap, seconds, label, expected):
    global child
    assert not output.exists() and not stopping
    check(expected)
    assert not any(r['status'] == 'RUNNING' for r in core.read(ROOT/'status/GPU_LEDGER.json')['runs'])
    command = [PYTHON,str(ROOT/'scripts/run_gpu_job.py'),'--goal','G07','--budget-window',
        'g07-full-racket-unlimited-20261024','--max-seconds',str(seconds),'--env-steps-cap',str(cap),
        '--output',str(output),'--require-artifact','completed.json','--',PYTHON,str(ROOT/script),
        '--output',str(output),*arguments,'--headless']
    with (HERE/(label+'.guard.log')).open('x') as log:
        child = subprocess.Popen(command,cwd=ROOT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2'),
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(label,guard_pid=child.pid,output=str(output))
        while child.poll() is None:
            assert not stopping
            time.sleep(5)
    assert child.returncode == 0 and not stopping,'Preserve actual failure; no automatic retry'
    check(expected)
    job,receipt = core.read(output/'job.json'),core.read(output/'completed.json')
    assert job['status'] == 'SUCCEEDED' and job['charged_env_steps'] == cap
    assert 0 < receipt['env_steps'] <= cap and not (output/'error.txt').exists()
    assert not core.live(job['pid'],output.name) and not core.live(job['child_pid'],output.name)
    return job


def verify_search(source_path):
    import numpy as np
    import torch
    from quadsmash.control.full_racket_contact_aim_v4 import ContactAimControllerV4, SCHEMA, aim_candidates
    from quadsmash.control.full_racket_body_search_v4 import summarize_candidate
    from quadsmash.control.full_racket_train_evaluation_v4 import training_cases
    from scripts.initialize_centered_policy_v4 import same_tree
    source = torch.load(source_path,map_location='cpu',weights_only=False)
    protocol = core.read(ROOT/'configs/g07_full_racket_eval_v4_r3.json')
    protocol['_root'] = str(ROOT)
    cases,indices = training_cases(ROOT,protocol)
    inputs,metrics = core.read(SEARCH/'INPUTS.json'),core.read(SEARCH/'metrics.json')
    assert inputs['source_policy_sha256'] == core.sha(source_path) and inputs['training_indices'] == indices
    assert inputs['candidate_aims_uv_m'] == [list(uv) for uv in aim_candidates()]
    assert len(metrics['history']) == 7
    default,limits = [torch.tensor(inputs[k],dtype=torch.float32) for k in
        ('default_joint_position_rad','joint_limits_rad')]
    best,receipts,steps = None,[],0
    for i,(entry,uv) in enumerate(zip(metrics['history'],aim_candidates())):
        rows = core.read(SEARCH/f'episodes_{i:03d}.json')
        initial = core.read(SEARCH/f'initial_ground_states_{i:03d}.json')
        assert len(rows) == 128 and initial['all_clocks_zero']
        for lane,row in enumerate(rows):
            assert row['env_id'] == row['jump']['env_id'] == lane
            assert row['candidate_id'] == i and row['source_training_index'] == lane and row['ground_init']
            assert row['case_id'] == cases[lane]['case_id'] and row['episode_index'] == initial['episode_ids'][lane]
        summary = summarize_candidate(rows,protocol)
        assert summary == entry['summary'] and summary['numerical_failures'] == 0
        assert entry['env_steps'] == entry['control_ticks']*128 and entry['candidate_id'] == i
        rank = tuple(summary['rank'])
        if best is None or rank > best[0]:
            best = rank,i,uv
        steps += entry['env_steps']
        with np.load(SEARCH/f'public_and_actions_{i:03d}.npz') as arrays:
            observations,actions,valid = [arrays[k].copy() for k in ('observations','actions','valid_first_episode')]
        controller = ContactAimControllerV4(source,uv,2,'cpu')
        previous = None
        maximum = 0.
        with torch.inference_mode():
            for obs,actual,keep in zip(observations,actions,valid):
                public = torch.from_numpy(obs)
                if previous is not None:
                    reset = torch.where(public[:,775] < previous-1e-5)[0]
                    if len(reset):
                        controller.reset(reset)
                command = controller.action({'policy':public},default,limits).numpy()
                if keep.any():
                    maximum = max(maximum,float(abs(command[keep]-actual[keep]).max()))
                previous = public[:,775].clone()
        assert maximum <= 5e-4, 'Preserve actual action mismatch'
        receipts.append(dict(candidate_id=i,actual_valid_public_rows=int(valid.sum()),
            CPU_vs_actual_GPU_max_action_error=maximum,summary=summary))
    policy = torch.load(SEARCH/'best_candidate.pt',map_location='cpu',weights_only=False)
    assert policy['schema'] == SCHEMA and policy['native_execution'] and not policy['performance_inherited']
    assert same_tree(policy['source_public_policy'],source) and torch.equal(policy['parameters'],source['parameters'])
    assert list(policy['aim_uv_m']) == list(best[2]) and policy['actual_native_search']['candidate_id'] == best[1]
    assert metrics['actual_control_env_steps'] == core.read(SEARCH/'completed.json')['env_steps'] == steps
    return dict(checked_at=datetime.now(timezone.utc).isoformat(),all896_selected_native_episodes_checked=True,
        actual_control_env_steps=steps,source_policy_unchanged=True,actual_public_replays=receipts,
        best_candidate_id=best[1],best_aim_uv_m=best[2],formal_cases_read=False,formal_acceptance=False)


def main():
    assert PLAN.exists() and not STATE.exists()
    plan = core.read(PLAN)
    expected = dict(plan['source_sha256'])
    expected[str(PLAN)] = core.sha(PLAN)
    check(expected)
    write('WAITING_FOR_ACTUAL_BODY48_PARENT_EXIT',wait_parent_pid=PREVIOUS_PID,new_native_job_started=False)
    while core.live(PREVIOUS_PID,'body_residual48_after_grid16_queue01/run.py'):
        assert not stopping
        time.sleep(5)
    assert core.read(PREVIOUS/'state.json')['state'] == plan['required_parent_terminal_state']
    check(expected)
    selected = core.read(PREVIOUS/'BEST_MEASURED_SOURCE.json')['selected']
    source_path,evaluation = Path(selected['policy']),Path(selected['evaluation'])
    summary = core.verify_evaluation(evaluation,source_path)
    assert summary == selected['summary']
    if ready(summary):
        write('READY_FOR_FORMAL_SOURCE_REVIEW',selected=selected,extra_search_skipped=True)
        return
    for name in INSTALL:
        path = ROOT/name
        assert not path.exists(), 'Never overwrite a user or active runtime file'
        with path.open('xb') as stream:
            stream.write((ISOLATED/name).read_bytes())
        expected[str(path)] = core.sha(path)
    for path in sorted(ROOT.glob('quadsmash/**/*.py'))+sorted(ROOT.glob('scripts/*.py')):
        expected[str(path)] = core.sha(path)
    for path in [source_path,*[evaluation/name for name in
        ('INPUTS.json','metrics.json','episodes.json','job.json','completed.json','public_and_actions_first2.npz')]]:
        expected[str(path)] = core.sha(path)
    source_check = verify_source_commands(source_path,evaluation)
    save(HERE/'ACTUAL_SELECTED_SOURCE_COMMAND_CHECK.json',source_check)
    expected[str(HERE/'ACTUAL_SELECTED_SOURCE_COMMAND_CHECK.json')] = core.sha(HERE/'ACTUAL_SELECTED_SOURCE_COMMAND_CHECK.json')
    save(HERE/'ADMISSION_DECISION.json',dict(checked_at=datetime.now(timezone.utc).isoformat(),source=selected,
        source_sha256=expected,actual_prior_parent_exited=True,original_target_retained=True,formal_acceptance=False))
    run_job(SEARCH,'scripts/search_full_racket_contact_aim_v4.py',
        ['--source',str(source_path),'--source-evaluation',str(evaluation),'--seed','750401'],
        plan['search_env_steps_cap'],plan['search_max_seconds'],'RUNNING_NATIVE_CONTACT_AIM7',expected)
    write('VERIFYING_ACTUAL_CONTACT_AIM_SEARCH')
    save(HERE/'SEARCH_RECEIPT.json',verify_search(source_path))
    policy = SEARCH/'best_candidate.pt'
    expected[str(policy)] = core.sha(policy)
    run_job(FROZEN,'scripts/evaluate_full_racket_contact_aim_train_v4.py',
        ['--checkpoint',str(policy),'--noise','none','--seed','724201'],102400,2700,
        'RUNNING_FIXED_CONTACT_AIM_TRAIN128',expected)
    measured = core.verify_evaluation(FROZEN,policy)
    save(HERE/'COLD_TRAIN128_RESULT.json',measured)
    with (HERE/'csv_export.log').open('x') as log:
        result = subprocess.run([PYTHON,str(ROOT/'scripts/export_full_racket_train_csv_v4.py'),
            '--run',str(FROZEN),'--output-prefix',str(HERE/'COLD128_EPISODES')],cwd=ROOT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2'),stdout=log,stderr=subprocess.STDOUT)
    assert result.returncode == 0
    key = lambda s:(s['complete'],s['conditional_recovery_count'],s['airborne_contacts'],s['attack_outgoing'])
    best = selected
    if key(measured) > key(summary):
        best = dict(candidate_id='contact_aim7',policy=str(policy),evaluation=str(FROZEN),
            policy_sha256=core.sha(policy),summary=measured)
    save(HERE/'BEST_MEASURED_SOURCE.json',dict(selected=best,prior=selected,
        new_measured=dict(policy=str(policy),evaluation=str(FROZEN),summary=measured),
        formal_cases_read=False,formal_acceptance=False))
    check(expected)
    write('COMPLETED_CONTACT_AIM_SEARCH_AND_COLD128',selected=best,
        ready_for_formal_source_review=ready(best['summary']),source_custody_released_on_actual_parent_exit=True)


def stop(*_):
    global stopping
    stopping = True
    if child is not None and child.poll() is None:
        child.send_signal(signal.SIGTERM)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        with (HERE/'observer.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            signal.signal(signal.SIGTERM,stop)
            signal.signal(signal.SIGINT,stop)
            try:
                main()
            except BaseException:
                if child is not None and child.poll() is None:
                    child.send_signal(signal.SIGTERM)
                    child.wait(timeout=60)
                write('STOPPED' if stopping else 'FAILED',error=traceback.format_exc(),automatic_retry=False)
                raise
