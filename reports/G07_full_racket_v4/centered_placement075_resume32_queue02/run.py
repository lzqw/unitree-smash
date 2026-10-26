"""Conditionally resume the centered learner after its frozen TRAIN128 check.

Use only the original main GPU0 guard/ledger. A changing-policy success or
successful job receipt never stands in for frozen performance or full G07.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback
import time

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
REPORTS=ROOT/'reports/G07_full_racket_v4';sys.path.insert(0,str(ROOT))
PYTHON='/home/autodrive/Isaac/env_isaaclab/bin/python'
spec=importlib.util.spec_from_file_location('preserved_native_checks',REPORTS/'residual_resume24_queue01/run.py')
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
PARENT=REPORTS/'centered_placement075_pilot8_queue01'
SOURCE=REPORTS/'residual_ppo_centered_placement075_n2048_probe8_01'
PARENT_EVAL=REPORTS/'residual_ppo_centered_placement075_n2048_probe8_fixed_train128_01'
TRAIN=REPORTS/'residual_ppo_centered_placement075_n2048_resume32_02'
EVALUATION=REPORTS/'residual_ppo_centered_placement075_n2048_iter40_fixed_train128_02'
BASE=REPORTS/'residual_ppo_joint_placement_actor32_n2048_probe8_01/policy.pt'
CONFIG=ROOT/'configs/g07_full_racket_residual_ppo_centered_placement075_v4.json'
PLAN=HERE/'PREDECLARED_PLAN.json';STATE=HERE/'state.json'
VIDEO=REPORTS/'video/Current_effect_centered_placement075_iter40_retry02_20261026.mp4'
SNAPSHOT=REPORTS/'video_snapshot_centered_placement075_iter40_retry02_20261026'
VIDEO_SPEC=REPORTS/'video_spec_centered_placement075_iter40_retry02_20261026.json'
PRESELECTED=[19,23,107,0,8,58]
STEPS=32*2048*128
CUMULATIVE_STEPS=40*2048*128
stopping=False;child=None


def save_once(path,value):
    assert not path.exists(),'Preserve existing evidence: '+str(path)
    core.atomic(path,value)


def write(state,**extra):
    value=dict(state=state,queue_pid=os.getpid(),updated_at=datetime.now(timezone.utc).isoformat(),
        training_output=str(TRAIN),evaluation_output=str(EVALUATION),
        formal_cases_read=False,formal_acceptance=False,goal_complete=False,**extra)
    core.atomic(STATE,value);print(json.dumps(value),flush=True)


def verify_episode_chains(source,final,rows,count,allow_empty=False):
    import numpy as np
    initial=source['native']['environment']['tensors']['episode_index'].cpu().numpy()
    terminal=final['native']['environment']['tensors']['episode_index'].cpu().numpy()
    assert initial.shape==terminal.shape==(count,)
    seen=[[]for _ in range(count)]
    for row in rows:
        assert type(row['env_id'])is int and 0<=row['env_id']<count
        seen[row['env_id']].append(row['episode_index'])
    assert allow_empty or all(seen),'Each actual lane must have a closed episode'
    assert all(ids==list(range(int(initial[lane]),int(terminal[lane])))for lane,ids in enumerate(seen)), 'Missing, duplicated or reordered episodes after canonical resume'
    return dict(lanes=count,closed_episodes=len(rows),initial_episode_id_min=int(initial.min()),initial_episode_id_max=int(initial.max()),every_source_inclusive_to_final_exclusive_chain_exact=True)


def decision(summary,trace_ok):
    if summary['numerical_failures']!=0:return 'REVIEW_NUMERICAL_FAILURE'
    if summary['complete_success_rate']>=.5 and summary['airborne_contact_rate']>=.7 and summary['conditional_recovery_rate']>=.85:
        return 'READY_FOR_ORIGINAL_FORMAL_REVIEW'
    if not trace_ok:return 'REVIEW_MISSING_OR_FAILED_TRAINING_TRACE_AUDIT'
    if summary['airborne_contacts']<16 or summary['conditional_recovery_count']<8:
        return 'REVIEW_FROZEN_TRAIN_COLLAPSE'
    return 'CONTINUE_STRICT_SAME_CONTRACT_32_UPDATES'



def check_sources(expected):
    for path,digest in expected.items():assert core.sha(ROOT/path)==digest,'Pinned source changed: '+path


def command(output,seconds,steps,script,arguments):
    return [PYTHON,str(ROOT/'scripts/run_gpu_job.py'),'--goal','G07',
        '--budget-window','g07-full-racket-unlimited-20261024','--max-seconds',str(seconds),
        '--env-steps-cap',str(steps),'--output',str(output),'--require-artifact','completed.json','--',
        PYTHON,str(ROOT/script),'--output',str(output),*arguments,'--headless']


def commands():
    train=command(TRAIN,14400,STEPS,'scripts/train_full_racket_residual_ppo_v4.py',
        ['--base-checkpoint',str(BASE),'--config',str(CONFIG),'--num-envs','2048','--updates','32','--resume',str(SOURCE/'checkpoint.pt'),'--seed','748901'])
    evaluate=command(EVALUATION,2700,102400,'scripts/evaluate_full_racket_train_v4.py',
        ['--checkpoint',str(TRAIN/'policy.pt'),'--noise','none','--seed','724201'])
    return train,evaluate


def verify_prior_interruption():
    import torch
    from scripts.initialize_centered_policy_v4 import same_tree
    prior = REPORTS/'residual_ppo_centered_placement075_n2048_resume32_01'
    observer = REPORTS/'centered_placement075_resume32_queue01'
    job = core.read(prior/'job.json');state = core.read(observer/'state.json')
    assert job['status'] == 'FAILED' and not job['receipt_present'] and not job['timed_out']
    assert state['state'] == 'FAILED_NATIVE_OR_FOLLOWUP' and 'Observer cancelled' in state['error']
    assert not core.live(state['queue_pid'],str(observer/'run.py'))
    assert all(not core.live(job[k],str(prior)) for k in ['pid','child_pid'])
    assert not (prior/'completed.json').exists()
    assert not any(v['status'] == 'RUNNING' for v in core.read(ROOT/'status/GPU_LEDGER.json')['runs'])
    metrics = core.read(prior/'metrics.json');interrupted = core.read(prior/'interrupted.json')
    assert metrics['completed_updates'] == 0 and metrics['history'] == []
    assert interrupted['env_steps'] == 96256 and metrics['counts']['numerical_failures'] == 0
    source = torch.load(SOURCE/'checkpoint.pt',map_location='cpu',weights_only=False)
    stopped = torch.load(prior/'checkpoint.pt',map_location='cpu',weights_only=False)
    assert source['iter'] == stopped['iter'] == 8
    assert source['native']['canonical_restart'] and stopped['native']['canonical_restart']
    assert source['native']['total_timesteps'] == stopped['native']['total_timesteps'] == 2097152
    assert same_tree(source['optimizer_state_dict'],stopped['optimizer_state_dict'])
    before,after = source['model_state_dict'],stopped['model_state_dict']
    changed = [key for key in before if not same_tree(before[key],after[key])]
    assert set(changed) == {prefix+suffix for prefix in ['actor_obs_normalizer.','critic_obs_normalizer.'] for suffix in ['_mean','_var','_std','count']}
    for name in ['actor_obs_normalizer.count','critic_obs_normalizer.count']:
        assert int(after[name])-int(before[name]) == interrupted['env_steps']
    old_plan = core.read(observer/'PREDECLARED_PLAN.json');check_sources(old_plan['source_hashes'])
    evidence = dict(checked_at=datetime.now(timezone.utc).isoformat(),prior_observer_pid=state['queue_pid'],
        prior_guard_pid=job['pid'],prior_native_pid=job['child_pid'],all_prior_processes_verified_exited=True,
        prior_status=job['status'],prior_GPU_hours=job['charged_gpu_hours'],prior_charged_env_steps=job['charged_env_steps'],
        prior_actual_interrupted_env_steps=interrupted['env_steps'],prior_completed_updates=0,
        prior_actor_critic_weights_and_Adam_unchanged=True,prior_normalizers_changed_only_by_unfinished_rollout=True,
        previous_source_pins_verified=len(old_plan['source_hashes']),source_checkpoint_sha256=core.sha(SOURCE/'checkpoint.pt'),
        stopped_checkpoint_sha256=core.sha(prior/'checkpoint.pt'),
        new_attempt_source='Original completed canonical075 stage8, not the interrupted partial normalizers.',
        resource_accounting='Previous failed attempt remains charged in the original shared ledger; no reset.',
        observer_launch='Detached session with file-backed stdout/stderr; SIGTERM still requests a graceful checkpoint stop.',
        formal_acceptance=False,goal_complete=False)
    save_once(HERE/'PREVIOUS_INTERRUPTION_CHECK.json',evidence)
    return evidence


def prepare():
    import torch
    prior_interruption = verify_prior_interruption()
    assert ROOT == ROOT.parent/'unitree_smash', 'Use original main, never a copied ledger'
    assert not any(path.exists() for path in [PLAN,STATE,TRAIN,EVALUATION,VIDEO,SNAPSHOT,VIDEO_SPEC])
    parent = core.read(PARENT/'state.json')
    parent_live = core.live(parent['queue_pid'],'centered_placement075_pilot8_queue01/run.py')
    assert parent_live or parent['state'] == 'READY_FOR_VISUAL_REVIEW'
    assert not any(v['status'] == 'RUNNING' for v in core.read(ROOT/'status/GPU_LEDGER.json')['runs'])
    expected = dict(core.read(PARENT/'PREDECLARED_PLAN.json')['source_hashes']);check_sources(expected)
    parent_spec = importlib.util.spec_from_file_location('prepared075_parent',PARENT/'run.py')
    parent_module = importlib.util.module_from_spec(parent_spec);parent_spec.loader.exec_module(parent_module)
    parent_job = core.verify_job(SOURCE,8*2048*128)
    receipt = parent_module.verify_training(parent_job)
    summary = core.verify_evaluation(PARENT_EVAL,SOURCE/'policy.pt')
    assert summary == core.read(PARENT/'FIXED_TRAIN128_RESULT.json')
    trace_ok = core.read(PARENT/'TRACE_AUDIT_ATTEMPT.json')['passed']
    assert decision(summary,trace_ok) == 'CONTINUE_STRICT_SAME_CONTRACT_32_UPDATES'
    audited = core.read(PARENT/'TRAINING_TRACE_AUDIT.json')
    assert audited['all_selected_trajectories_checked'] and audited['selected_episodes_checked'] == 12
    assert all(core.sha(Path(name)) == digest for name,digest in audited['source_sha256'].items())
    source = torch.load(SOURCE/'checkpoint.pt',map_location='cpu',weights_only=False)
    actor_count = int(source['model_state_dict']['actor_obs_normalizer.count'])
    critic_count = int(source['model_state_dict']['critic_obs_normalizer.count'])
    assert source['iter'] == 8 and source['native']['canonical_restart']
    assert actor_count == 12582912 and critic_count == 2097152
    assert source['native']['total_timesteps'] == critic_count
    reference = core.read(REPORTS/'centered_residual_resume32_queue01/TRAINING_RECEIPT.json')
    assert reference['actual_episode_chains']['every_source_inclusive_to_final_exclusive_chain_exact']
    check = dict(checked_at=datetime.now(timezone.utc).isoformat(),source_iteration=8,
        source_actor_normalizer_count=actor_count,source_critic_normalizer_count=critic_count,
        source_actor_normalizer_offset=actor_count-critic_count,
        expected_final_actor_normalizer_count=actor_count+STEPS,expected_final_critic_normalizer_count=critic_count+STEPS,
        expected_final_new_stage_steps=CUMULATIVE_STEPS,expected_final_actor_Adam_steps=760,expected_final_critic_Adam_steps=800,
        actual_prior_resume_chain_check_reused=True,source_checkpoint_sha256=core.sha(SOURCE/'checkpoint.pt'),
        new_native_execution=False,formal_acceptance=False,goal_complete=False)
    save_once(HERE/'SOURCE_ASYMMETRIC_RESUME_CPU_CHECK.json',check)
    dependencies = [PARENT/'PREDECLARED_PLAN.json',PARENT/'FIXED_TRAIN128_RESULT.json',PARENT/'TRAINING_RECEIPT.json',
        PARENT/'TRAINING_TRACE_AUDIT.json',PARENT/'TRACE_AUDIT_ATTEMPT.json',SOURCE/'checkpoint.pt',SOURCE/'policy.pt',
        SOURCE/'INPUTS.json',SOURCE/'metrics.json',SOURCE/'job.json',SOURCE/'completed.json',
        HERE/'SOURCE_ASYMMETRIC_RESUME_CPU_CHECK.json',Path(__file__).resolve(),
        REPORTS/'centered_residual_resume32_queue01/TRAINING_RECEIPT.json']
    dependencies += [PARENT_EVAL/name for name in ['INPUTS.json','metrics.json','job.json','completed.json','episodes.json','source_hashes.json']]
    prior = REPORTS/'residual_ppo_centered_placement075_n2048_resume32_01'
    previous_observer = REPORTS/'centered_placement075_resume32_queue01'
    dependencies += [prior/name for name in ['job.json','metrics.json','interrupted.json','checkpoint.pt']]
    dependencies += [previous_observer/name for name in ['run.py','PREDECLARED_PLAN.json','state.json','error.txt']]
    dependencies.append(HERE/'PREVIOUS_INTERRUPTION_CHECK.json')
    for name in audited['source_sha256']:dependencies.append(Path(name))
    for path in dependencies:expected[str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)] = core.sha(path)
    check_sources(expected);train,evaluate = commands()
    save_once(PLAN,dict(created_at=datetime.now(timezone.utc).isoformat(),source_hashes=expected,
        retry_after_verified_terminal_interruption=prior_interruption,
        verified_live_parent_pid=parent['queue_pid'],parent_actual_live_at_registration=parent_live,
        source_checkpoint=str(SOURCE/'checkpoint.pt'),source_checkpoint_sha256=core.sha(SOURCE/'checkpoint.pt'),
        source_policy_sha256=core.sha(SOURCE/'policy.pt'),measured_source_original_TRAIN128=summary,
        training_command=train,evaluation_command=evaluate,source_iteration=8,additional_updates=32,target_iteration=40,
        num_envs=2048,distinct_TRAIN_cases=128,copies_per_case=16,additional_control_env_steps=STEPS,
        expected_cumulative_new_learner_steps=CUMULATIVE_STEPS,source_actor_normalizer_offset=actor_count-critic_count,
        source_actor_normalizer_count=actor_count,source_critic_normalizer_count=critic_count,
        same_config_sha256=core.sha(CONFIG),fixed_base_policy_sha256=core.sha(BASE),
        decision_timing='After actual075 fixed128 result; development continuation, not a blind decision.',
        rationale='Measured075 stage8 gives26air,19conditional recoveries,5attacks,0complete/NF0. Preserve this improved measured source and its critic/Adam/RNG for longer learning; avoid another fresh-critic short pilot now. Larger-leg candidate remains optional and is not integrated.',
        internal_continuation_gate=dict(minimum_airborne_contacts=16,minimum_conditional_recoveries=8,numerical_failures=0,
            selected_native_training_trace_audit_required=True,scope='Development collapse guard only; original formal acceptance unchanged.'),
        original_formal_readiness_gate=dict(complete_success_rate=.5,airborne_contact_rate=.7,conditional_recovery_rate=.85,numerical_failures=0),
        restart_semantics='Strict same-config canonical8 resume: actor/critic/Adam/normalizers/RNG/counters retained, actual fresh ground. Actor RMS has prior10485760 offset, critic/new-stage counters do not.',
        episode_identifier_verification='Actual source-inclusive/final-exclusive per-lane chains; never require episode0 on resume.',
        first_resumed_native_update_check=True,physical_model_rules_cases_hardware_and_original_acceptance_unchanged=True,
        video_case_preselection=PRESELECTED,video_upload_requires_root_visual_review=True,
        source_custody='Keep all bound parent and continuation runtime/inputs unchanged until actual continuation/fixed/video process exit.',
        resource_authorization='Original main GPU0 shared guard/ledger only; unlimited cumulative authorization,14400s and8388608 steps finite job caps; no cloud.',
        formal_cases_read=False,formal_acceptance=False,goal_complete=False))
    print(json.dumps(dict(prepared=str(PLAN),sources=len(expected),actual_parent_live=parent_live,
        actual_source_original_TRAIN128=summary,source_actor_normalizer_count=actor_count,source_critic_normalizer_count=critic_count,
        new_native_job_started=False)),flush=True)


def verify_first_resumed_update():
    import numpy as np
    import torch
    source = torch.load(SOURCE/'checkpoint.pt',map_location='cpu',weights_only=False)
    native = torch.load(TRAIN/'ppo/model_8.pt',map_location='cpu',weights_only=False)
    inputs = core.read(TRAIN/'INPUTS.json')
    assert inputs['resume_checkpoint_sha256'] == core.sha(SOURCE/'checkpoint.pt')
    assert inputs['contract'] == core.read(SOURCE/'INPUTS.json')['contract'] and inputs['configuration_sha256'] == core.sha(CONFIG)
    assert inputs['initialization'] == dict(mode='strict_canonical_ground_resume',num_envs=2048,initial_iteration=8)
    assert native['iter'] == 9 and not native['native']['canonical_restart']
    assert native['native']['total_timesteps'] == source['native']['total_timesteps']+262144
    model = native['model_state_dict'];previous = source['model_state_dict']
    assert all(torch.isfinite(v).all() for v in model.values() if torch.is_tensor(v))
    for name in ['actor_obs_normalizer.count','critic_obs_normalizer.count']:
        assert int(model[name])-int(previous[name]) == 262144
    before,after = source['optimizer_state_dict']['state'],native['optimizer_state_dict']['state']
    assert set(before) == set(after) and len(after) == 17
    assert all(float(after[k]['step'])-float(before[k]['step']) == 20 for k in before)
    assert all(torch.isfinite(v).all() for state in after.values() for v in state.values() if torch.is_tensor(v))
    final_ids = native['native']['environment']['tensors']['episode_index'].tolist()
    rows = core.read(TRAIN/'episodes.partial.json')
    rows = [row for row in rows if row['episode_index'] < final_ids[row['env_id']]]
    partial_chains = verify_episode_chains(source,native,rows,2048,allow_empty=True)
    ground = core.read(TRAIN/'initial_ground_states.json');roots = np.asarray(ground['root_state_local'])
    protocol = core.read(ROOT/'configs/g07_full_racket_eval_v4_r3.json')
    assert roots.shape == (2048,13) and np.isfinite(roots).all() and ground['all_episode_clocks_zero']
    assert ground['source_training_indices'] == list(range(128))*16
    assert np.max(abs(roots[:,:3]-protocol['initialization']['root_position_m'])) <= 1e-5
    assert np.max(abs(roots[:,3:7]-[1,0,0,0])) <= 1e-5 and np.max(abs(roots[:,7:])) <= 1e-5
    assert np.max(abs(np.asarray(ground['joint_velocity_rad_s']))) <= 1e-5
    active = [v for v in core.read(ROOT/'status/GPU_LEDGER.json')['runs'] if v['status'] == 'RUNNING']
    assert len(active) == 1 and Path(active[0]['output']) == TRAIN
    assert core.live(active[0]['pid'],str(TRAIN)) and core.live(active[0]['child_pid'],str(TRAIN))
    progress = core.read(TRAIN/'progress.json');assert progress['update'] >= 9 and progress['numerical_failures'] == 0
    assert progress['trajectory_recording']['enabled'] and progress['trajectory_recording']['verified_full_public_rows'] >= 262144
    save_once(HERE/'FIRST_ACTUAL_RESUMED_UPDATE_CHECK.json',dict(checked_at=datetime.now(timezone.utc).isoformat(),
        actual_native_pid=active[0]['child_pid'],actual_stage_update=9,actual_current_job_control_env_steps=262144,
        actual_stage_total_env_steps=native['native']['total_timesteps'],all17_Adam_states_preserved_and_advanced20=True,
        actor_normalizer_count=int(model['actor_obs_normalizer.count']),critic_normalizer_count=int(model['critic_obs_normalizer.count']),
        prior_actor_normalizer_offset_preserved=True,actual_first_update_episode_ID_chains=partial_chains,
        fresh_2048_ground_checked=True,recording_enabled=True,numerical_failures=0,
        source_checkpoint_sha256=core.sha(SOURCE/'checkpoint.pt'),diagnostic9_sha256=core.sha(TRAIN/'ppo/model_8.pt'),
        formal_acceptance=False,goal_complete=False))



def guarded(command,output,steps,label,expected):
    global child
    assert not stopping and not output.exists();check_sources(expected)
    first_checked = output != TRAIN
    with (HERE/(label+'.guard.log')).open('x') as log:
        child = subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2'),
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(label,guard_pid=child.pid)
        try:
            while child.poll() is None:
                assert not stopping,'Observer cancelled'
                progress = output/'progress.json'
                if not first_checked and progress.exists() and (output/'episodes.partial.json').exists() and core.read(progress)['update'] >= 9:
                    verify_first_resumed_update();first_checked = True
                time.sleep(2)
            code = child.returncode
        except BaseException:
            if child.poll() is None:child.terminate()
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:pass
            raise
    assert code == 0 and not stopping and first_checked,'Preserve actual guard/update failure; no automatic restart'
    check_sources(expected);return core.verify_job(output,steps)


def verify_training(job):
    import numpy as np
    import torch
    from quadsmash.evaluation_v4 import independent_episode
    inputs=core.read(TRAIN/'INPUTS.json');metrics=core.read(TRAIN/'metrics.json');source_inputs=core.read(SOURCE/'INPUTS.json')
    assert inputs['resume_checkpoint_sha256']==core.sha(SOURCE/'checkpoint.pt')
    assert inputs['initial_actor_checkpoint_sha256']is None and not inputs['resize_ground_requested']
    assert inputs['contract']==source_inputs['contract'] and inputs['configuration_sha256']==core.sha(CONFIG)
    assert inputs['contract']['base_controller_mode']=='frozen_public_policy' and inputs['contract']['base_policy_sha256']==core.sha(BASE)
    native=torch.load(TRAIN/'checkpoint.pt',map_location='cpu',weights_only=False)
    source=torch.load(SOURCE/'checkpoint.pt',map_location='cpu',weights_only=False)
    policy=torch.load(TRAIN/'policy.pt',map_location='cpu',weights_only=False)
    base=torch.load(BASE,map_location='cpu',weights_only=False)
    assert source['iter']==8 and source['native']['canonical_restart'] and native['iter']==40 and native['native']['canonical_restart']
    assert native['native']['total_timesteps']==CUMULATIVE_STEPS and source['native']['total_timesteps']+STEPS==CUMULATIVE_STEPS
    assert native['native']['environment']['num_envs']==source['native']['environment']['num_envs']==2048
    assert native['native']['training_signature']==source['native']['training_signature']
    assert not native['native']['environment']['tensors']['episode_length_buf'].any()
    assert all(torch.isfinite(v).all()for v in native['model_state_dict'].values()if torch.is_tensor(v))
    assert all(torch.isfinite(v).all()for state in native['optimizer_state_dict']['state'].values()for v in state.values()if torch.is_tensor(v))
    initialization=inputs['initialization'];assert initialization==metrics['initialization']==native['infos']['learner_initialization']
    assert initialization==dict(mode='strict_canonical_ground_resume',num_envs=2048,initial_iteration=8)
    assert native['infos']['residual_contract']==inputs['contract']
    before=source['optimizer_state_dict']['state'];after=native['optimizer_state_dict']['state'];assert set(before)==set(after) and len(after)==17
    assert all(float(after[k]['step'])-float(before[k]['step'])==32*20 for k in before)
    for name in ['actor_obs_normalizer.count','critic_obs_normalizer.count']:
        assert int(native['model_state_dict'][name])-int(source['model_state_dict'][name])==STEPS
    assert any(not torch.equal(v,source['model_state_dict'][name])for name,v in native['model_state_dict'].items()if name.startswith('actor.')and torch.is_tensor(v))
    assert policy['native_execution']and policy['residual_base_mode']=='frozen_public_policy'and not policy['performance_inherited']
    assert policy['base_controller']['residual_actor_jit']==base['residual_actor_jit']
    meta=policy['residual_training_metadata'];assert meta['completed_updates']==40 and meta['actual_control_env_steps']==STEPS
    assert meta['base_policy_sha256']==core.sha(BASE) and meta['initialization']==initialization
    assert core.finite(meta['export_actor_max_error']) and meta['export_actor_max_error']<=2e-5
    assert hashlib.sha256(policy['residual_actor_jit']).hexdigest()==meta['actor_jit_sha256']
    assert core.sha(TRAIN/'policy.pt')==metrics['policy_sha256']
    assert metrics['completed_updates']==len(metrics['history'])==32 and core.finite(metrics['history'])
    assert [entry['update']for entry in metrics['history']]==list(range(9,41))
    assert metrics['history'][-1]['actual_control_env_steps']==STEPS
    assert all(not entry['actor_warmup']for entry in metrics['history'])
    ground=core.read(TRAIN/'initial_ground_states.json');protocol=core.read(ROOT/'configs/g07_full_racket_eval_v4_r3.json');protocol['_root']=str(ROOT)
    roots=np.asarray(ground['root_state_local']);assert roots.shape==(2048,13)and np.isfinite(roots).all()
    assert ground['all_episode_clocks_zero']and ground['source_training_indices']==list(range(128))*16
    assert np.max(abs(roots[:,:3]-np.asarray(protocol['initialization']['root_position_m'])))<=1e-5
    assert np.max(abs(roots[:,3:7]-np.array([1.,0.,0.,0.])))<=1e-5 and np.max(abs(roots[:,7:]))<=1e-5
    assert np.max(abs(np.asarray(ground['joint_velocity_rad_s'])))<=1e-5
    rows=core.read(TRAIN/'episodes.json');assert core.sha(TRAIN/'episodes.json')==core.sha(TRAIN/'episodes.partial.json')
    chains=verify_episode_chains(source,native,rows,2048)
    assert all(row['ground_init']and row['source_training_index']==row['env_id']%128 for row in rows)
    outcomes=[independent_episode(row,protocol)for row in rows]
    counts=dict(completed_episodes=len(rows),air=sum(v['airborne_contact']for v in outcomes),
        conditional_recovery_count=sum(v['conditional_recovery']for v in outcomes),complete=sum(v['complete']for v in outcomes),
        numerical_failures=sum(row['numerical_failures']for row in rows))
    assert counts==metrics['counts']and counts['numerical_failures']==0
    return dict(checked_at=datetime.now(timezone.utc).isoformat(),actual_control_env_steps=STEPS,cumulative_new_learner_steps=CUMULATIVE_STEPS,
        actual_GPU_hours=job['charged_gpu_hours'],source_checkpoint_sha256=core.sha(SOURCE/'checkpoint.pt'),
        checkpoint_sha256=core.sha(TRAIN/'checkpoint.pt'),policy_sha256=core.sha(TRAIN/'policy.pt'),fixed_base_policy_sha256=core.sha(BASE),
        initial_iteration=8,final_iteration=40,all17_Adam_states_advanced640_steps=True,both_normalizer_counts_preserved_and_advanced=STEPS,
        source_actor_normalizer_count=int(source['model_state_dict']['actor_obs_normalizer.count']),
        source_critic_normalizer_count=int(source['model_state_dict']['critic_obs_normalizer.count']),
        actor_normalizer_count=int(native['model_state_dict']['actor_obs_normalizer.count']),
        critic_normalizer_count=int(native['model_state_dict']['critic_obs_normalizer.count']),
        actor_weights_changed=True,actual_episode_chains=chains,independent_changing_policy_counts=counts,
        frozen_policy_performance=False,formal_acceptance=False,goal_complete=False)



def cpu(command,label,required=True):
    global child
    environment=dict(os.environ,PYTHONPATH='/tmp/unitree_video_deps',LIBGL_ALWAYS_SOFTWARE='1',LP_NUM_THREADS='2',
        OMP_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='',__EGL_VENDOR_LIBRARY_FILENAMES='/usr/share/glvnd/egl_vendor.d/50_mesa.json')
    with (HERE/(label+'.log')).open('x')as log:
        child=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
        try:code=child.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            child.terminate()
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            raise
    if required:assert code==0 and not stopping,'CPU follow-up failed: '+label
    return code


def video_spec(summary):
    import numpy as np
    from quadsmash.evaluation_v4 import independent_episode
    protocol=core.read(ROOT/'configs/g07_full_racket_eval_v4_r3.json');protocol['_root']=str(ROOT)
    rows=core.read(EVALUATION/'episodes.json');cases=[]
    for index in PRESELECTED:
        row=rows[index];assert row['source_training_index']==row['env_id']==index and row['episode_index']==0
        outcome=independent_episode(row,protocol);velocity=np.asarray(row['outgoing_velocity_mps']);landing=np.asarray(row['first_landing_m'])
        kind='合法空中拍线' if outcome['airborne_contact']else '拍框接触' if row['frame_hit']else '未通过合格空中拍线判定'
        outgoing=f'出球{np.linalg.norm(velocity):.2f}米/秒'if np.isfinite(velocity).all()else'未记录有效出球'
        landed=f'首落[{landing[0]:.2f},{landing[1]:.2f}]米'if np.isfinite(landing).all()else'未观察到首次落点'
        cases.append(dict(run=EVALUATION.name,source_type='evaluation',env_id=index,
            title=f'固定TRAIN{index}｜{kind}',subtitle='固定旧8基线｜落点反馈0.75阶段40｜实际原生全身轨迹',
            terminal_status=('完整扣杀成立'if outcome['complete']else'完整扣杀失败')+('｜合格恢复成立'if outcome['conditional_recovery']else'｜触球与恢复联合条件未通过'),
            limitation=f'{outgoing}；{landed}；稳定窗口{row["jump"]["recovery_s"]:.2f}秒'))
    counts=f'空中{summary["airborne_contacts"]}、条件恢复{summary["conditional_recovery_count"]}、完整{summary["complete"]}'
    return dict(scope='Frozen075 stage40 after strict stage8 resume, unchanged fixed old8 base; original TRAIN128, not changing-policy or formal A/B performance.',
        cases=cases,opening_title='落点反馈0.75｜阶段PPO40',
        opening_lines=['本阶段canonical8严格续训32次至40；保留价值网络、Adam、归一化和RNG。','全128真实结果：'+counts+'；数值故障'+str(summary['numerical_failures'])+'。','训练前登记六例，全部实际结果保留。'],
        closing_title='固定TRAIN128诊断｜原门槛不变',
        closing_lines=[counts+'；每例为独立原生执行。','固定基线成绩不继承，训练成功不代替冻结表现。','正式达标仍需同一冻结策略通过A/B各200回合。'],
        evaluation_footer='落点反馈0.75阶段40｜'+counts+'｜非正式A/B',
        caveats=['Actual native50Hz full-body poses and exact contact/landing events; no generated or interpolated motion.',
            'All six comparison cases were registered before the original centered pilot and retained for this continuation. Not blind/random sampling.',
            'Same075 stage8-to40 canonical continuation with prior actor-normalizer offset preserved; distinct from prior025 centered40 or mutable-actor40.',
            'Last valid50Hz sample is explicitly distinguished from exact terminal or landing substeps.'])


def main():
    assert PLAN.exists() and not STATE.exists();plan=core.read(PLAN)
    expected=dict(plan['source_hashes']);expected[str(PLAN.relative_to(ROOT))]=core.sha(PLAN);check_sources(expected)
    train,evaluate=commands();assert train==plan['training_command'] and evaluate==plan['evaluation_command']
    write('WAITING_FOR_VERIFIED_PARENT',observed_parent_pid=plan['verified_live_parent_pid'])
    while core.live(plan['verified_live_parent_pid'],'centered_placement075_pilot8_queue01/run.py'):
        assert not stopping,'Observer cancelled'
        check_sources(expected);time.sleep(10)
    parent_state=core.read(PARENT/'state.json')
    assert parent_state['state']=='READY_FOR_VISUAL_REVIEW','Parent not complete: '+parent_state['state']
    assert not any(v['status']=='RUNNING'for v in core.read(ROOT/'status/GPU_LEDGER.json')['runs'])
    for folder in [SOURCE,PARENT_EVAL]:
        prior_job = core.read(folder/'job.json')
        for name in ['pid','child_pid']:
            assert not core.live(prior_job[name],folder.name),'Prior native source owner remains live'
    parent_spec=importlib.util.spec_from_file_location('checked_centered_pilot',PARENT/'run.py')
    parent=importlib.util.module_from_spec(parent_spec);parent_spec.loader.exec_module(parent)
    parent_job=core.verify_job(SOURCE,8*2048*128);parent_receipt=parent.verify_training(parent_job)
    parent_summary=core.verify_evaluation(PARENT_EVAL,SOURCE/'policy.pt')
    audit_attempt=core.read(PARENT/'TRACE_AUDIT_ATTEMPT.json')
    trace_ok=audit_attempt['passed']and audit_attempt['exit_code']==0
    if trace_ok:
        audited=core.read(PARENT/'TRAINING_TRACE_AUDIT.json')
        trace_ok=audited['run']==str(SOURCE)and audited['all_selected_trajectories_checked']and audited['selected_episodes_checked']>0
        assert all(core.sha(Path(name))==digest for name,digest in audited['source_sha256'].items())
    next_action=decision(parent_summary,trace_ok)
    save_once(HERE/'PARENT_VERIFIED_DECISION.json',dict(checked_at=datetime.now(timezone.utc).isoformat(),
        parent_training=parent_receipt,independent_fixed_TRAIN128=parent_summary,selected_trace_audit_passed=trace_ok,
        next_action=next_action,formal_acceptance=False,goal_complete=False))
    if next_action!='CONTINUE_STRICT_SAME_CONTRACT_32_UPDATES':
        write(next_action,independent_fixed_TRAIN128=parent_summary,new_native_job_started=False);return
    check_sources(expected)
    job=guarded(train,TRAIN,STEPS,'RUNNING_NATIVE_PLACEMENT075_RESUME32',expected)
    receipt=verify_training(job);save_once(HERE/'TRAINING_RECEIPT.json',receipt)
    audit_code=cpu([PYTHON,'scripts/check_full_racket_training_traces_v4.py','--run',str(TRAIN),
        '--output',str(HERE/'TRAINING_TRACE_AUDIT.json')],'trace_audit',required=False)
    save_once(HERE/'TRACE_AUDIT_ATTEMPT.json',dict(exit_code=audit_code,passed=audit_code==0,
        scope='Selected actual changing-policy training traces only; not frozen or formal acceptance.'))
    guarded(evaluate,EVALUATION,102400,'RUNNING_FIXED_PLACEMENT075_ITER40_TRAIN128',expected)
    summary=core.verify_evaluation(EVALUATION,TRAIN/'policy.pt');save_once(HERE/'FIXED_TRAIN128_RESULT.json',summary)
    for run,label in [(TRAIN,'centered_training_episodes'),(EVALUATION,'centered_fixed_episodes')]:
        cpu([PYTHON,'scripts/export_full_racket_train_csv_v4.py','--run',str(run),'--output-prefix',str(HERE/label)],label)
    save_once(VIDEO_SPEC,video_spec(summary));write('PREPARING_ACTUAL_FROZEN_VIDEO',independent_fixed_TRAIN128=summary)
    cpu([PYTHON,'scripts/prepare_full_racket_latest_video_v4.py','--spec',str(VIDEO_SPEC),'--output',str(SNAPSHOT)],'video_prepare')
    write('RENDERING_ACTUAL_VIDEO_ON_CPU')
    cpu([PYTHON,'scripts/render_full_racket_latest_effect_v4.py','--snapshot',str(SNAPSHOT),'--output',str(VIDEO)],'video_render')
    old=REPORTS/'joint_placement_iter40_video_queue01/run.py'
    spec=importlib.util.spec_from_file_location('preserved_video_decode',old);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cpu([PYTHON,'-c',module.DECODE,str(VIDEO)],'video_decode')
    check_sources(expected)
    write('READY_FOR_VISUAL_REVIEW',training=receipt,independent_fixed_TRAIN128=summary,
        next_action=decision(summary,audit_code==0),selected_training_trace_audit_passed=audit_code==0,video=str(VIDEO),video_sha256=core.sha(VIDEO),
        automatic_upload_performed=False,scope='Actual new learning and frozen TRAIN diagnostics; full original G07 remains unproven.')


def stop(*_):
    global stopping
    stopping=True
    if child is not None and child.poll()is None:child.terminate()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    with (HERE/'observer.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.prepare_only:prepare()
        else:
            signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
            try:main()
            except BaseException as error:
                (HERE/'error.txt').write_text(traceback.format_exc());write('FAILED_NATIVE_OR_FOLLOWUP',error=repr(error));raise
