"""Wait for actual grid exit, choose measured source, learn and freeze four actors.

This is a new learner, not a resume of the old075 actor/critic/Adam. All native
work uses the existing main GPU0 guard. A failed dependency never auto-retries.
"""
import argparse
from datetime import datetime,timezone
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

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
REPORTS=ROOT/'reports/G07_full_racket_v4'
PRIOR=REPORTS/'body_strike_grid16_queue01'
PRIOR_PID=1810743
TRAIN=REPORTS/'body_residual_leg15_fresh_n2048_updates48_01'
CONFIG=ROOT/'configs/g07_full_racket_body_residual_leg15_v4.json'
PLAN,STATE=HERE/'PLAN.json',HERE/'state.json'
PYTHON='/home/autodrive/Isaac/env_isaaclab/bin/python'
UPDATES,STEPS=48,48*2048*128
ITERATIONS=[13,24,36,48]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location('original_shared_job_checks',REPORTS/'residual_resume24_queue01/run.py')
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
import export_intermediate
child=None
stopping=False

def save(path,value):
    assert not path.exists(),str(path)
    core.atomic(path,value)

def write(state,**extra):
    core.atomic(STATE,dict(state=state,queue_pid=os.getpid(),observed_at=datetime.now(timezone.utc).isoformat(),
        formal_cases_read=False,formal_acceptance=False,goal_complete=False,**extra))

def check(expected):
    for name,digest in expected.items():assert core.sha(ROOT/name)==digest,name

def ready(s):
    return s['complete_success_rate']>=.5 and s['airborne_contact_rate']>=.7 and s['conditional_recovery_rate']>=.85 and s['numerical_failures']==0

def rank(c):
    s=c['summary']
    return s['complete'],s['conditional_recovery_count'],s['airborne_contacts'],s['attack_outgoing'],-c['ordinal']

def frozen(iteration):return REPORTS/f'body_residual_fresh_iter{iteration:02d}_fixed_train128_01'

def cpu(command,label):
    with (HERE/(label+'.log')).open('x') as log:
        result=subprocess.run([PYTHON,*command],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2'),
            stdout=log,stderr=subprocess.STDOUT)
    assert result.returncode==0,'Preserve actual CPU failure: '+label

def prepare():
    assert not any(p.exists() for p in [PLAN,STATE,TRAIN])
    assert core.live(PRIOR_PID,'body_strike_grid16_queue01/run.py')
    active=[r for r in core.read(ROOT/'status/GPU_LEDGER.json')['runs'] if r['status']=='RUNNING']
    assert len(active)==1 and Path(active[0]['output'])==REPORTS/'body_strike_grid16_n2048_01'
    assert core.live(active[0]['pid'],'body_strike_grid16_n2048_01') and core.live(active[0]['child_pid'],'body_strike_grid16_n2048_01')
    prior=core.read(PRIOR/'PLAN.json');check(prior['source_hashes'])
    proof=core.read(HERE/'CPU_PREPARATION_CHECK.json')
    assert proof['zero_residual_source_action_max_error']==0 and proof['both_existing_actors_preserved_and_frozen']
    for name,digest in proof['source_sha256'].items():assert core.sha(Path(name))==digest,name
    nonzero=core.read(HERE/'CPU_NONZERO_EXPORT_CHECK.json')
    assert nonzero['max_actor_export_error']<=2e-5 and nonzero['max_nonzero_normalized_residual']>0
    assert not nonzero['new_policy_native_execution']
    for name,digest in nonzero['source_sha256'].items():assert core.sha(Path(name))==digest,name
    expected=dict(prior['source_hashes'])
    paths=[Path(__file__),HERE/'check_source_zero.py',HERE/'export_intermediate.py',HERE/'CPU_PREPARATION_CHECK.json',
        HERE/'CPU_NONZERO_EXPORT_CHECK.json',PRIOR/'PLAN.json']
    # Bind the actual imported owner/check helpers too, not just native code.
    paths += [Path(m.__file__) for m in list(sys.modules.values()) if getattr(m,'__file__',None)
        and str(m.__file__).startswith(str(REPORTS)) and str(m.__file__).endswith('.py')]
    expected.update({str(p.relative_to(ROOT)):core.sha(p) for p in paths});check(expected)
    save(PLAN,dict(created_at=datetime.now(timezone.utc).isoformat(),source_hashes=expected,
        wait_actual_parent_pid=PRIOR_PID,wait_terminal_state='COMPLETED_BODY_STRIKE_GRID16_AND_COLD128',
        hypothesis='Start a fresh public-observation residual from the best independently measured body teacher, preserving both existing actors; explicitly increase requested physical leg exploration from.003 to.03rad and retain learned intermediates.',
        source_selection='Recompute all14 existing fixed128 policies plus completed grid16 fixed128, rank complete/recovery/air/attack/earlier ordinal. Never use the16-way mixed search rate as a fixed-policy score.',
        skip_learning_if_any_measured_source_meets_original50_70_85=True,
        initialization='Fresh zero18D output, actor/critic RMS0, empty Adam, sigma legs.2 arms.1, scales all.15rad. Fixed teacher and its two residual actors/normalizers remain exact.',
        seed=750301,num_envs=2048,steps_per_update=262144,updates=UPDATES,critic_only_warmup_updates=7,
        train_env_steps_cap=STEPS,train_max_seconds=21600,
        frozen_iterations=ITERATIONS,evaluation_env_steps_cap=102400,evaluation_max_seconds=2700,
        all_original128_TRAIN_cases_each=True,evaluation_noise='none',evaluation_seed=724201,
        intermediate_export='Actual model_i actor weights and public normalization for inference only; require later actual collection. Never use diagnostic files as canonical restart.',
        checkpoints='Native canonical checkpoint retains optimizer, normalization, RNG and counters at a new ground boundary; diagnostic model_i retained for every update.',
        automatic_retry=False,source_custody='At admission bind selected future source/data SHA; keep all bound runtime/config/asset files and PLAN until actual workflow exit.',
        original_physics_hardware_5rad_s_arm_target_rate_and_success_gates_unchanged=True,
        previous_goal_turn_classification='progress: actual grid launch and2048 ground verification, delivered verified91.68s video, measured body4/body6 weights and both fixed128 records.',
        formal_cases_read=False,formal_acceptance=False,goal_complete=False))
    print(json.dumps(dict(prepared=str(PLAN),bound_inputs=len(expected),new_native_job_started=False)),flush=True)

def select_source():
    import torch
    from quadsmash.control.full_racket_body_residual_v4 import validate_frozen_baseline
    candidates=core.read(PRIOR/'PLAN.json')['actual_fourteen_source_selection']
    candidates += [dict(candidate_id='grid16',policy=str(REPORTS/'body_strike_grid16_n2048_01/best_candidate.pt'),
        evaluation=str(REPORTS/'body_strike_grid16_fixed_train128_01'))]
    measured=[]
    for ordinal,c in enumerate(candidates):
        policy,evaluation=Path(c['policy']),Path(c['evaluation'])
        summary=core.verify_evaluation(evaluation,policy)
        measured.append(dict(candidate_id=c['candidate_id'],ordinal=ordinal,policy=str(policy),evaluation=str(evaluation),
            policy_sha256=core.sha(policy),summary=summary))
    qualifying=[c for c in measured if ready(c['summary'])]
    chosen=max(qualifying or measured,key=rank)
    if not qualifying:
        validate_frozen_baseline(torch.load(chosen['policy'],map_location='cpu',weights_only=False))
    selection=dict(checked_at=datetime.now(timezone.utc).isoformat(),all_candidates=measured,selected=chosen,
        actual_grid_parent_exited=True,formal_cases_read=False,formal_acceptance=False)
    save(HERE/'SOURCE_SELECTION.json',selection)
    return chosen,bool(qualifying)

def optimizer_steps(native):
    model,optimizer=native['model_state_dict'],native['optimizer_state_dict']
    ids=[i for group in optimizer['param_groups'] for i in group['params']]
    names=[name for name in model if name=='log_std' or name.startswith(('actor.','critic.'))]
    assert len(ids)==len(names)==17
    return {name:int(optimizer['state'][i]['step']) for i,name in zip(ids,names) if i in optimizer['state']}

def verify_update(iteration,base_path):
    import numpy as np
    import torch
    initial=torch.load(TRAIN/'fresh_initial_model_state.pt',map_location='cpu',weights_only=False)
    start=initial['model_state_dict']
    assert initial['base_policy_sha256']==core.sha(base_path) and initial['actual_control_env_steps']==0
    assert initial['optimizer_entries']==0 and initial['fixed_source_residual_actor_count']==2
    assert int(start['actor_obs_normalizer.count'])==int(start['critic_obs_normalizer.count'])==0
    assert torch.allclose(start['log_std'].exp(),torch.tensor([.2]*12+[.1]*6),atol=1e-7,rtol=0)
    actor_keys=[name for name in start if name.startswith('actor.')]
    assert not start[actor_keys[-1]].any() and not start[actor_keys[-2]].any()
    path=TRAIN/f'ppo/model_{iteration-1}.pt';native=torch.load(path,map_location='cpu',weights_only=False)
    model=native['model_state_dict'];adam=optimizer_steps(native)
    assert native['iter']==iteration and not native['native']['canonical_restart']
    assert native['native']['total_timesteps']==iteration*262144
    assert int(model['actor_obs_normalizer.count'])==int(model['critic_obs_normalizer.count'])==iteration*262144
    equal=all(torch.equal(model[name],start[name]) for name in [*actor_keys,'log_std'])
    assert equal==(iteration<=7)
    assert len(adam)==(8 if iteration<=7 else 17)
    assert all(value==(iteration*20 if name.startswith('critic.') else (iteration-7)*20) for name,value in adam.items())
    if iteration==7:assert int(native['native']['environment']['tensors']['episode_index'].min())>=1
    inputs=core.read(TRAIN/'INPUTS.json');init=inputs['initialization']
    assert inputs['configuration_sha256']==core.sha(CONFIG) and inputs['num_envs']==2048 and inputs['requested_updates']==UPDATES
    assert inputs['contract']['base_policy_sha256']==core.sha(base_path) and inputs['contract']['actor_warmup_updates']==7
    assert inputs['contract']['fixed_source_actor_count']==2 and inputs['contract']['seed']==750301
    assert init['mode']=='fresh_zero_mean_residual_over_measured_two_actor_body_teacher'
    assert all(init[name] is False for name in ['actor_transferred','critic_transferred','critic_normalizer_transferred',
        'optimizer_transferred','source_rng_restored','old_physical_state_restored','previous_learning_counters_restored'])
    if iteration==1:
        ground=core.read(TRAIN/'initial_ground_states.json');protocol=core.read(ROOT/'configs/g07_full_racket_eval_v4_r3.json')
        roots=np.asarray(ground['root_state_local']);assert roots.shape==(2048,13) and np.isfinite(roots).all()
        assert np.max(abs(roots[:,:3]-protocol['initialization']['root_position_m']))<1e-5
        assert np.max(abs(roots[:,3:7]-[1,0,0,0]))<1e-5 and np.max(abs(roots[:,7:]))<1e-5
        assert np.max(abs(np.asarray(ground['joint_velocity_rad_s'])))<1e-5
        assert ground['all_episode_clocks_zero'] and ground['source_training_indices']==list(range(128))*16
    return dict(checked_at=datetime.now(timezone.utc).isoformat(),actual_iteration=iteration,
        diagnostic_sha256=core.sha(path),fresh_zero_actor_and_empty_Adam_initial_state_verified=True,
        actor_weights_and_std_unchanged=equal,actual_Adam_steps=adam,
        actor_RMS_count=int(model['actor_obs_normalizer.count']),critic_RMS_count=int(model['critic_obs_normalizer.count']),
        all2048_first_episodes_completed=(iteration==7),actual_initial_ground2048_checked=(iteration==1),
        diagnostic_is_canonical_restart=False,formal_acceptance=False)

def run_job(output,command,seconds,steps,label,expected,base_path,training=False):
    global child
    assert not output.exists() and not stopping;check(expected)
    assert not any(r['status']=='RUNNING' for r in core.read(ROOT/'status/GPU_LEDGER.json')['runs'])
    wrapper=[PYTHON,str(ROOT/'scripts/run_gpu_job.py'),'--goal','G07','--budget-window',
        'g07-full-racket-unlimited-20261024','--max-seconds',str(seconds),'--env-steps-cap',str(steps),
        '--output',str(output),'--require-artifact','completed.json','--',*command]
    checked=set()
    with (HERE/(label+'.guard.log')).open('x') as log:
        child=subprocess.Popen(wrapper,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2'),
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(label,guard_pid=child.pid,output=str(output))
        try:
            while child.poll() is None:
                assert not stopping
                if training:
                    for iteration in [1,7,8]:
                        if iteration not in checked and (TRAIN/f'ppo/model_{iteration-1}.pt').exists():
                            save(HERE/f'ACTUAL_UPDATE{iteration}_CHECK.json',verify_update(iteration,base_path));checked.add(iteration)
                time.sleep(5)
        except BaseException:
            if child.poll() is None:child.send_signal(signal.SIGTERM);child.wait(timeout=60)
            raise
    assert child.returncode==0 and not stopping,'Preserve actual failure; no automatic retry'
    if training:
        for iteration in [1,7,8]:
            if iteration not in checked:save(HERE/f'ACTUAL_UPDATE{iteration}_CHECK.json',verify_update(iteration,base_path))
    check(expected);return core.verify_job(output,steps)

def verify_training(base_path):
    import torch
    from quadsmash.evaluation_v4 import independent_episode
    from scripts.initialize_centered_policy_v4 import same_tree
    job=core.verify_job(TRAIN,STEPS);native=torch.load(TRAIN/'checkpoint.pt',map_location='cpu',weights_only=False)
    assert native['iter']==UPDATES and native['native']['canonical_restart'] and native['native']['total_timesteps']==STEPS
    assert not native['native']['environment']['tensors']['episode_length_buf'].any()
    state=native['model_state_dict'];adam=optimizer_steps(native)
    assert int(state['actor_obs_normalizer.count'])==int(state['critic_obs_normalizer.count'])==STEPS
    assert len(adam)==17 and all(value==(960 if name.startswith('critic.') else 820) for name,value in adam.items())
    assert all(torch.isfinite(v).all() for v in state.values() if torch.is_tensor(v))
    assert all(torch.isfinite(v).all() for s in native['optimizer_state_dict']['state'].values() for v in s.values() if torch.is_tensor(v))
    inputs,metrics=core.read(TRAIN/'INPUTS.json'),core.read(TRAIN/'metrics.json')
    assert metrics['completed_updates']==UPDATES and len(metrics['history'])==UPDATES
    assert [r['actor_warmup'] for r in metrics['history']]==[True]*7+[False]*41
    public=torch.load(TRAIN/'policy.pt',map_location='cpu',weights_only=False)
    assert public['schema']=='full-racket-body-residual-ppo-v4' and public['native_execution'] and not public['performance_inherited']
    assert same_tree(public['base_controller'],torch.load(base_path,map_location='cpu',weights_only=False))
    assert core.sha(TRAIN/'policy.pt')==metrics['policy_sha256']
    assert public['residual_training_metadata']['export_actor_max_error']<=2e-5
    rows=core.read(TRAIN/'episodes.json');seen=[0]*2048
    protocol=core.read(ROOT/'configs/g07_full_racket_eval_v4_r3.json');protocol['_root']=str(ROOT)
    outcomes=[]
    for row in rows:
        lane=row['env_id'];assert row['ground_init'] and row['source_training_index']==lane%128 and row['episode_index']==seen[lane]
        seen[lane]+=1;outcomes.append(independent_episode(row,protocol))
    assert seen==native['native']['environment']['tensors']['episode_index'].tolist()
    counts=dict(completed_episodes=len(rows),air=sum(r['airborne_contact'] for r in outcomes),
        conditional_recovery_count=sum(r['conditional_recovery'] for r in outcomes),complete=sum(r['complete'] for r in outcomes),
        numerical_failures=sum(r['numerical_failures'] for r in rows))
    assert counts==metrics['counts'] and counts['numerical_failures']==0
    return dict(checked_at=datetime.now(timezone.utc).isoformat(),actual_control_env_steps=STEPS,
        actual_GPU_hours=job['charged_gpu_hours'],checkpoint_sha256=core.sha(TRAIN/'checkpoint.pt'),
        policy_sha256=core.sha(TRAIN/'policy.pt'),independent_changing_policy_counts=counts,
        source_teacher_exact=True,all2048_episode_chains_exact=True,actor_Adam_steps=820,critic_Adam_steps=960,
        frozen_policy_performance=False,formal_acceptance=False)

def main():
    assert PLAN.exists() and not STATE.exists();plan=core.read(PLAN)
    expected=dict(plan['source_hashes']);expected[str(PLAN.relative_to(ROOT))]=core.sha(PLAN);check(expected)
    write('WAITING_FOR_ACTUAL_GRID16_PARENT_EXIT',wait_parent_pid=PRIOR_PID,new_native_job_started=False)
    while core.live(PRIOR_PID,'body_strike_grid16_queue01/run.py'):
        assert not stopping;time.sleep(5)
    assert core.read(PRIOR/'state.json')['state']==plan['wait_terminal_state']
    check(expected);chosen,already_ready=select_source()
    if already_ready:
        write('READY_FOR_FORMAL_SOURCE_REVIEW',selected=chosen,reason='An actual original128 source met all original readiness thresholds; skip more TRAIN optimization.')
        return
    base_path,evaluation=Path(chosen['policy']),Path(chosen['evaluation'])
    cpu([str(HERE/'check_source_zero.py'),'--policy',str(base_path),'--evaluation',str(evaluation),
        '--output',str(HERE/'ACTUAL_SELECTED_SOURCE_ZERO_CHECK.json')],'selected_source_zero')
    for p in [base_path,evaluation/'episodes.json',evaluation/'metrics.json',evaluation/'INPUTS.json',
        evaluation/'source_hashes.json',evaluation/'job.json',evaluation/'completed.json',evaluation/'public_and_actions_first2.npz',
        HERE/'SOURCE_SELECTION.json',HERE/'ACTUAL_SELECTED_SOURCE_ZERO_CHECK.json']:
        expected[str(p.relative_to(ROOT))]=core.sha(p)
    save(HERE/'ADMISSION_DECISION.json',dict(checked_at=datetime.now(timezone.utc).isoformat(),source=chosen,
        source_hashes=expected,source_actor_count=2,new_trainable_actor_count=1,formal_acceptance=False))
    command=[PYTHON,str(ROOT/'scripts/train_full_racket_body_residual_ppo_v4.py'),'--output',str(TRAIN),
        '--base-checkpoint',str(base_path),'--body-source-evaluation',str(evaluation),'--config',str(CONFIG),
        '--num-envs','2048','--updates',str(UPDATES),'--seed','750301','--headless']
    run_job(TRAIN,command,21600,STEPS,'RUNNING_FRESH_BODY_RESIDUAL48',expected,base_path,True)
    write('VERIFYING_ACTUAL_COMPLETED_TRAINING');save(HERE/'TRAINING_RECEIPT.json',verify_training(base_path))
    cpu([str(ROOT/'scripts/check_full_racket_training_traces_v4.py'),'--run',str(TRAIN),
        '--output',str(HERE/'TRAINING_TRACE_AUDIT.json')],'trace_audit')
    policies={48:TRAIN/'policy.pt'}
    for iteration in [13,24,36]:
        policies[iteration],_=export_intermediate.export(iteration,TRAIN,base_path,evaluation)
    measured=list(core.read(HERE/'SOURCE_SELECTION.json')['all_candidates'])
    results=[]
    for iteration in ITERATIONS:
        policy=policies[iteration];expected[str(policy.relative_to(ROOT))]=core.sha(policy)
        output=frozen(iteration)
        command=[PYTHON,str(ROOT/'scripts/evaluate_full_racket_body_residual_train_v4.py'),'--output',str(output),
            '--checkpoint',str(policy),'--noise','none','--seed','724201','--headless']
        run_job(output,command,2700,102400,f'RUNNING_FIXED_ITER{iteration:02d}_TRAIN128',expected,base_path)
        summary=core.verify_evaluation(output,policy);save(HERE/f'FIXED_ITER{iteration:02d}_TRAIN128_RESULT.json',summary)
        cpu([str(ROOT/'scripts/export_full_racket_train_csv_v4.py'),'--run',str(output),
            '--output-prefix',str(HERE/f'fixed_iter{iteration:02d}_TRAIN128')],f'csv_iter{iteration:02d}')
        entry=dict(candidate_id=f'body_residual{iteration}',ordinal=len(measured),iteration=iteration,policy=str(policy),
            evaluation=str(output),policy_sha256=core.sha(policy),summary=summary)
        measured.append(entry);results.append(entry)
        if ready(summary):break
    check(expected);qualifying=[c for c in measured if ready(c['summary'])]
    best=max(qualifying or measured,key=rank)
    save(HERE/'BEST_MEASURED_SOURCE.json',dict(selected=best,all_candidates=measured,new_evaluations=results,
        preserves_better_prior_teacher=True,formal_acceptance=False))
    write('COMPLETED_FRESH_BODY_RESIDUAL48_AND_FIXED_COMPARISON',selected=best,
        new_frozen_policies=len(results),ready_for_formal_source_review=ready(best['summary']),
        source_custody_released_on_actual_parent_exit=True)

def stop(*_):
    global stopping
    stopping=True
    if child is not None and child.poll() is None:child.send_signal(signal.SIGTERM)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    if args.prepare:prepare()
    else:
        with (HERE/'observer.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
            try:main()
            except BaseException:
                if child is not None and child.poll() is None:child.send_signal(signal.SIGTERM);child.wait(timeout=60)
                write('STOPPED' if stopping else 'FAILED',error=traceback.format_exc(),automatic_retry=False)
                raise
