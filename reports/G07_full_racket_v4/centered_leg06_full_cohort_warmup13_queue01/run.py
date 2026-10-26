"""Compare full-first-cohort critic warmup with the completed leg06 pilot.

One shared GPU0 guard, unchanged native trainer, same source/scales/seed and six
actor updates. Seven critic-only updates precede those actor updates. No resume
of partial jobs, no automatic retry, and no automatic longer training.
"""
import argparse
import contextlib
from datetime import datetime, timezone
import fcntl
import importlib.util
import io
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
PRIOR = REPORTS/'centered_leg06_after075_intermediates_pilot8_queue01'
CONFIG = ROOT/'configs/g07_full_racket_residual_ppo_centered_leg06_full_cohort_warmup_v4.json'
TRAIN = REPORTS/'residual_ppo_centered_leg06_full_cohort_warmup7_n2048_probe13_01'
FROZEN = REPORTS/'residual_ppo_centered_leg06_full_cohort_warmup7_n2048_probe13_fixed_train128_01'
SOURCE = REPORTS/'residual_ppo_centered_placement075_n2048_probe8_01'
SOURCE_EVAL = REPORTS/'residual_ppo_centered_placement075_n2048_probe8_fixed_train128_01'
BASE = REPORTS/'residual_ppo_joint_placement_actor32_n2048_probe8_01/policy.pt'
VIDEO_ADMISSION = Path('/home/autodrive/LZQW/unitree_smash_video_framing/reports/G07_full_racket_v4/orthographic_leg06_pilot8_after_native_queue01/ACTUAL_FROZEN8_INPUT_CHECK.json')
PYTHON = '/home/autodrive/Isaac/env_isaaclab/bin/python'
PLAN, STATE = HERE/'PLAN.json', HERE/'state.json'
STEPS = 13*2048*128
sys.path.insert(0,str(ROOT))
spec = importlib.util.spec_from_file_location('actual_leg06_reference',PRIOR/'run.py')
reference = importlib.util.module_from_spec(spec); spec.loader.exec_module(reference)
core = reference.core
stopping, child = False, None


def save(path,value):
    assert not path.exists(), 'Preserve prior evidence: '+str(path)
    core.atomic(path,value)


def write(state,**extra):
    core.atomic(STATE,dict(state=state,queue_pid=os.getpid(),updated_at=datetime.now(timezone.utc).isoformat(),
        CPU_only=False,formal_cases_read=False,formal_acceptance=False,goal_complete=False,**extra))


def cpu_check():
    import numpy as np
    import torch
    from tensordict import TensorDict
    from rsl_rl.modules import ActorCritic
    from scripts.initialize_centered_action_units_v4 import apply_action_unit_mapped_actor
    from scripts.initialize_centered_policy_v4 import verify_measured_source
    old = core.read(reference.CONFIG); configuration = core.read(CONFIG)
    expected = dict(old,actor_warmup_updates=7)
    assert configuration == expected and old['actor_warmup_updates'] == 2
    source = torch.load(SOURCE/'checkpoint.pt',map_location='cpu',weights_only=False)
    contract = dict(source['infos']['residual_contract'],joint_scale_rad=configuration['joint_scale_rad'],
        seed=749301,actor_warmup_updates=7)
    provenance = verify_measured_source(ROOT,SOURCE/'policy.pt',SOURCE_EVAL,contract)
    options = dict(configuration['runner']['policy']); options.pop('class_name')
    torch.manual_seed(749301)
    with contextlib.redirect_stdout(io.StringIO()):
        policy = ActorCritic(TensorDict({'policy':torch.zeros(2,794)},batch_size=[2]),
            configuration['runner']['obs_groups'],18,**options)
    optimizer = torch.optim.Adam(policy.parameters(),lr=1e-4)
    with np.load(SOURCE_EVAL/'public_and_actions_first2.npz') as arrays:
        witness = torch.from_numpy(arrays['observations'][arrays['valid_first_episode'].astype(bool)].copy())
    public = torch.load(SOURCE/'policy.pt',map_location='cpu',weights_only=False)
    base = torch.load(BASE,map_location='cpu',weights_only=False)
    transferred = apply_action_unit_mapped_actor(policy,public,source['native']['interface'],contract,base,witness)
    assert not optimizer.state and int(policy.critic_obs_normalizer.count) == 0
    return dict(checked_at=datetime.now(timezone.utc).isoformat(),configuration_only_change='actor_warmup_updates:2->7',
        fresh_actor_updates_in_each_trial=6,total_new_updates=13,actual_source_witness_rows=len(witness),
        source_provenance=provenance,transfer=transferred,fresh_Adam_entries=0,
        source_seed_and_all_physical_contracts_unchanged=True,new_native_execution=False,formal_acceptance=False)


def prepare():
    assert not any(p.exists() for p in [PLAN,STATE,TRAIN,FROZEN])
    assert not core.live(1768388,'centered_leg06_after075_intermediates_pilot8_queue01/run.py')
    assert core.read(PRIOR/'state.json')['state'] == 'READY_FOR_VISUAL_REVIEW'
    assert VIDEO_ADMISSION.is_file(), 'Wait for actual completed-data video admission before new runtime custody'
    assert not any(row['status'] == 'RUNNING' for row in core.read(ROOT/'status/GPU_LEDGER.json')['runs'])
    prior_training = reference.verify_training(core.verify_job(reference.TRAIN,2097152))
    prior_summary = core.verify_evaluation(reference.EVALUATION,reference.TRAIN/'policy.pt')
    assert not reference.ready(prior_summary), 'Review a ready frozen source before additional training'
    original = core.read(PRIOR/'PREDECLARED_PLAN.json')['source_hashes']
    assert all(core.sha(ROOT/name) == digest for name,digest in original.items())
    checked = cpu_check(); save(HERE/'CPU_SOURCE_TRANSFER_CHECK.json',checked)
    sources = dict(original)
    for path in [Path(__file__),CONFIG,PRIOR/'PREDECLARED_PLAN.json',
        PRIOR/'TRAINING_RECEIPT.json',PRIOR/'FIXED_TRAIN128_RESULT.json',HERE/'CPU_SOURCE_TRANSFER_CHECK.json']:
        if path.is_file(): sources[str(path.relative_to(ROOT))] = core.sha(path)
    training = [PYTHON,str(ROOT/'scripts/train_full_racket_residual_ppo_v4.py'),'--output',str(TRAIN),
        '--base-checkpoint',str(BASE),'--config',str(CONFIG),'--num-envs','2048','--updates','13',
        '--initialize-centered-action-units',str(SOURCE/'policy.pt'),'--centered-source-evaluation',str(SOURCE_EVAL),
        '--seed','749301','--headless']
    evaluation = [PYTHON,str(ROOT/'scripts/evaluate_full_racket_train_v4.py'),'--output',str(FROZEN),
        '--checkpoint',str(TRAIN/'policy.pt'),'--noise','none','--seed','724201','--headless']
    save(PLAN,dict(created_at=datetime.now(timezone.utc).isoformat(),source_hashes=sources,
        training_command=training,evaluation_command=evaluation,prior_actual_training=prior_training,
        prior_original_TRAIN128=prior_summary,video_admission_sha256=core.sha(VIDEO_ADMISSION),
        hypothesis='Fresh critic should observe every initial full8s ground episode before actor weight learning. Existing actual first cohort closed by update7; earlier failure-only subsets are censored.',
        comparison='Same0758 source, same leg06 mapping/base/scales/rewards/seed749301,6 actor updates; critic warmup2->7 and total updates8->13. More critic samples/cost are explicit.',
        actor_normalization_continues_during_warmup=True,policy_function_not_frozen_during_warmup=True,
        actual_steps_cap=STEPS,per_job_wall_cap_s=10800,automatic_retry=False,automatic_longer_training=False,
        source_custody='Hold bound inputs and this PLAN until actual training,fixed128,CSV and trace auditing parent exit.',
        formal_cases_read=False,formal_acceptance=False,goal_complete=False))
    print(json.dumps(dict(prepared=str(PLAN),bound_inputs=len(sources),new_native_execution=False)),flush=True)


def verify_update(iteration):
    import numpy as np
    import torch
    path = TRAIN/f'ppo/model_{iteration-1}.pt'
    native = torch.load(path,map_location='cpu',weights_only=False)
    assert native['iter'] == iteration and not native['native']['canonical_restart']
    assert native['native']['total_timesteps'] == iteration*262144
    model = native['model_state_dict']; steps = reference.optimizer_steps(native)
    assert int(model['actor_obs_normalizer.count']) == 12582912+iteration*262144
    assert int(model['critic_obs_normalizer.count']) == iteration*262144
    initial = reference.mapped_initial_actor(core.read(PRIOR/'PREDECLARED_PLAN.json')['selected'])
    equal = all(torch.equal(model[name],value) for name,value in initial.items())
    if iteration <= 7:
        assert equal and len(steps) == 8 and all(v == iteration*20 for v in steps.values())
        assert torch.allclose(model['log_std'].exp(),torch.tensor([.05]*12+[.1]*6),atol=1e-7,rtol=0)
    else:
        assert not equal and len(steps) == 17
        assert all(v == (iteration*20 if name.startswith('critic.') else (iteration-7)*20) for name,v in steps.items())
    if iteration == 7:
        assert int(native['native']['environment']['tensors']['episode_index'].min()) >= 1
    ground_checked = False
    if iteration == 1:
        inputs = core.read(TRAIN/'INPUTS.json'); init = inputs['initialization']
        assert inputs['configuration_sha256'] == core.sha(CONFIG) and inputs['num_envs'] == 2048
        assert inputs['requested_updates'] == 13 and inputs['contract']['actor_warmup_updates'] == 7
        assert inputs['contract']['seed'] == 749301 and inputs['contract']['base_policy_sha256'] == core.sha(BASE)
        assert init['mode'] == 'fresh_measured_centered_actor_with_explicit_action_unit_rescaling'
        assert init['source_policy_sha256'] == core.sha(SOURCE/'policy.pt') and init['actual_witness_physical_mean_max_error_rad'] <= 2e-6
        assert all(init[name] is False for name in ['critic_transferred','critic_normalizer_transferred',
            'optimizer_transferred','source_rng_restored','old_physical_state_restored','previous_learning_counters_restored'])
        ground = core.read(TRAIN/'initial_ground_states.json')
        protocol = core.read(ROOT/'configs/g07_full_racket_eval_v4_r3.json')
        roots = np.asarray(ground['root_state_local'])
        assert roots.shape == (2048,13) and np.isfinite(roots).all()
        assert np.max(abs(roots[:,:3]-protocol['initialization']['root_position_m'])) < 1e-5
        assert np.max(abs(roots[:,3:7]-[1.,0.,0.,0.])) < 1e-5 and np.max(abs(roots[:,7:])) < 1e-5
        assert np.max(abs(np.asarray(ground['joint_velocity_rad_s']))) < 1e-5
        assert ground['all_episode_clocks_zero'] and ground['source_training_indices'] == list(range(128))*16
        ground_checked = True
    return dict(checked_at=datetime.now(timezone.utc).isoformat(),actual_iteration=iteration,
        diagnostic_sha256=core.sha(path),mapped_actor_weights_unchanged=equal,actual_Adam_steps=steps,
        all2048_first_ground_episodes_completed_by_update7=(iteration == 7),
        actor_RMS_count=int(model['actor_obs_normalizer.count']),critic_RMS_count=int(model['critic_obs_normalizer.count']),
        actual_fresh_ground2048_checked=ground_checked,
        native_restart_checkpoint=False,formal_acceptance=False,goal_complete=False)


def run_job(output,command,seconds,steps,label,expected,training=False):
    global child
    assert not output.exists() and not stopping
    assert all(core.sha(ROOT/name) == digest for name,digest in expected.items())
    assert not any(row['status'] == 'RUNNING' for row in core.read(ROOT/'status/GPU_LEDGER.json')['runs'])
    wrapper = [PYTHON,str(ROOT/'scripts/run_gpu_job.py'),'--goal','G07','--budget-window',
        'g07-full-racket-unlimited-20261024','--max-seconds',str(seconds),'--env-steps-cap',str(steps),
        '--output',str(output),'--require-artifact','completed.json','--',*command]
    with (HERE/(label+'.guard.log')).open('x') as log:
        child = subprocess.Popen(wrapper,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2'),
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(label,guard_pid=child.pid,output=str(output))
        checked = set()
        try:
            while child.poll() is None:
                assert not stopping
                if training:
                    for iteration in [1,7,8]:
                        if iteration not in checked and (TRAIN/f'ppo/model_{iteration-1}.pt').exists():
                            result = verify_update(iteration)
                            save(HERE/f'ACTUAL_UPDATE{iteration}_CHECK.json',result); checked.add(iteration)
                time.sleep(5)
        except BaseException:
            # A semantic validation failure interrupts only this workflow's
            # guard, allowing its existing checkpoint/receipt shutdown path.
            # A slow observation never enters this branch: polling has no
            # observation timeout, and it never restarts a native job.
            if child.poll() is None:
                if not stopping: child.send_signal(signal.SIGTERM)
                child.wait(timeout=30)
            raise
        assert child.returncode == 0 and not stopping, 'Preserve actual failure; no automatic retry'
        if training:
            for iteration in [1,7,8]:
                if iteration not in checked: save(HERE/f'ACTUAL_UPDATE{iteration}_CHECK.json',verify_update(iteration))
    assert all(core.sha(ROOT/name) == digest for name,digest in expected.items())
    return core.verify_job(output,steps)


def verify_training():
    import torch
    from quadsmash.evaluation_v4 import independent_episode
    from scripts.initialize_centered_policy_v4 import same_tree
    job = core.verify_job(TRAIN,STEPS)
    native = torch.load(TRAIN/'checkpoint.pt',map_location='cpu',weights_only=False)
    metrics,inputs = core.read(TRAIN/'metrics.json'),core.read(TRAIN/'INPUTS.json')
    assert native['iter'] == 13 and native['native']['canonical_restart'] and native['native']['total_timesteps'] == STEPS
    assert not native['native']['environment']['tensors']['episode_length_buf'].any()
    assert inputs['configuration_sha256'] == core.sha(CONFIG) and inputs['requested_updates'] == 13
    assert inputs['initialization']['mode'] == 'fresh_measured_centered_actor_with_explicit_action_unit_rescaling'
    assert inputs['initialization']['source_policy_sha256'] == core.sha(SOURCE/'policy.pt')
    assert inputs['contract']['actor_warmup_updates'] == 7 and inputs['contract']['seed'] == 749301
    assert metrics['completed_updates'] == 13 and [r['actor_warmup'] for r in metrics['history']] == [True]*7+[False]*6
    state = native['model_state_dict']; adam = reference.optimizer_steps(native)
    assert int(state['actor_obs_normalizer.count']) == 12582912+STEPS and int(state['critic_obs_normalizer.count']) == STEPS
    assert len(adam) == 17 and all(v == (260 if name.startswith('critic.') else 120) for name,v in adam.items())
    assert all(torch.isfinite(v).all() for v in state.values() if torch.is_tensor(v))
    assert all(torch.isfinite(v).all() for s in native['optimizer_state_dict']['state'].values() for v in s.values() if torch.is_tensor(v))
    public = torch.load(TRAIN/'policy.pt',map_location='cpu',weights_only=False)
    assert same_tree(public['base_controller'],torch.load(BASE,map_location='cpu',weights_only=False))
    assert not public['performance_inherited'] and public['residual_training_metadata']['export_actor_max_error'] <= 2e-5
    assert core.sha(TRAIN/'policy.pt') == metrics['policy_sha256']
    rows = core.read(TRAIN/'episodes.json'); seen = [0]*2048
    for row in rows:
        lane = row['env_id']; assert row['ground_init'] and row['source_training_index'] == lane%128 and row['episode_index'] == seen[lane]
        seen[lane] += 1
    assert seen == native['native']['environment']['tensors']['episode_index'].tolist() and min(seen) >= 1
    protocol = core.read(ROOT/'configs/g07_full_racket_eval_v4_r3.json'); protocol['_root'] = str(ROOT)
    outcomes = [independent_episode(row,protocol) for row in rows]
    counts = dict(completed_episodes=len(rows),air=sum(r['airborne_contact'] for r in outcomes),
        conditional_recovery_count=sum(r['conditional_recovery'] for r in outcomes),complete=sum(r['complete'] for r in outcomes),
        numerical_failures=sum(r['numerical_failures'] for r in rows))
    assert counts == metrics['counts'] and counts['numerical_failures'] == 0
    return dict(checked_at=datetime.now(timezone.utc).isoformat(),actual_control_env_steps=STEPS,
        actual_GPU_hours=job['charged_gpu_hours'],canonical_sha256=core.sha(TRAIN/'checkpoint.pt'),
        public_sha256=core.sha(TRAIN/'policy.pt'),independent_changing_policy_counts=counts,
        all2048_episode_chains_exact=True,actor_Adam_steps=120,critic_Adam_steps=260,
        frozen_policy_performance=False,formal_acceptance=False,goal_complete=False)


def main():
    assert PLAN.is_file() and not STATE.exists()
    plan = core.read(PLAN); expected = dict(plan['source_hashes'],**{str(PLAN.relative_to(ROOT)):core.sha(PLAN)})
    run_job(TRAIN,plan['training_command'],10800,STEPS,'RUNNING_FULL_COHORT_WARMUP13',expected,True)
    save(HERE/'TRAINING_RECEIPT.json',verify_training())
    write('AUDITING_ACTUAL_TRAINING_TRACES')
    with (HERE/'trace_audit.log').open('x') as log:
        result = subprocess.run([PYTHON,str(ROOT/'scripts/check_full_racket_training_traces_v4.py'),
            '--run',str(TRAIN),'--output',str(HERE/'TRAINING_TRACE_AUDIT.json')],cwd=ROOT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2'),stdout=log,stderr=subprocess.STDOUT)
    assert result.returncode == 0
    run_job(FROZEN,plan['evaluation_command'],2700,102400,'RUNNING_FIXED_WARMUP13_TRAIN128',expected)
    summary = core.verify_evaluation(FROZEN,TRAIN/'policy.pt')
    save(HERE/'FIXED_TRAIN128_RESULT.json',summary)
    with (HERE/'csv_export.log').open('x') as log:
        result = subprocess.run([PYTHON,str(ROOT/'scripts/export_full_racket_train_csv_v4.py'),'--run',str(FROZEN),
            '--output-prefix',str(HERE/'fixed_train128')],
            cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2'),stdout=log,stderr=subprocess.STDOUT)
    assert result.returncode == 0
    assert all(core.sha(ROOT/name) == digest for name,digest in expected.items())
    write('COMPLETED_FULL_COHORT_WARMUP13_AND_ORIGINAL_TRAIN128',summary=summary,source_custody_released_on_actual_parent_exit=True)


def stop(*_):
    global stopping
    stopping = True
    if child is not None and child.poll() is None: child.send_signal(signal.SIGTERM)
    raise RuntimeError('Full cohort workflow interrupted; preserve actual job and partial outputs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-check-only',action='store_true')
    parser.add_argument('--prepare-only',action='store_true')
    args = parser.parse_args()
    with (HERE/'observer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.cpu_check_only: save(HERE/'ACTUAL_CPU_CANDIDATE_CHECK.json',cpu_check())
        elif args.prepare_only: prepare()
        else:
            signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
            try: main()
            except BaseException as error:
                (HERE/'error.txt').write_text(traceback.format_exc())
                write('FAILED_NO_AUTOMATIC_RETRY',error=repr(error)); raise
