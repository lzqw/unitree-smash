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
PARENT=REPORTS/'centered_residual_pilot8_queue02'
SOURCE=REPORTS/'residual_ppo_centered_base8_n2048_probe8_02'
PARENT_EVAL=REPORTS/'residual_ppo_centered_base8_n2048_probe8_fixed_train128_02'
TRAIN=REPORTS/'residual_ppo_centered_base8_n2048_resume32_01'
EVALUATION=REPORTS/'residual_ppo_centered_base8_n2048_iter40_fixed_train128_01'
BASE=REPORTS/'residual_ppo_joint_placement_actor32_n2048_probe8_01/policy.pt'
CONFIG=ROOT/'configs/g07_full_racket_residual_ppo_centered_v4.json'
PLAN=HERE/'PREDECLARED_PLAN.json';STATE=HERE/'state.json'
VIDEO=REPORTS/'video/Current_effect_centered_residual_base8_iter40_20261026.mp4'
SNAPSHOT=REPORTS/'video_snapshot_centered_residual_base8_iter40_20261026'
VIDEO_SPEC=REPORTS/'video_spec_centered_residual_base8_iter40_20261026.json'
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


def verify_episode_chains(source,final,rows,count):
    import numpy as np
    initial=source['native']['environment']['tensors']['episode_index'].cpu().numpy()
    terminal=final['native']['environment']['tensors']['episode_index'].cpu().numpy()
    assert initial.shape==terminal.shape==(count,)
    seen=[[]for _ in range(count)]
    for row in rows:
        assert type(row['env_id'])is int and 0<=row['env_id']<count
        seen[row['env_id']].append(row['episode_index'])
    assert all(seen),'Each actual lane must have a closed episode'
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
        ['--base-checkpoint',str(BASE),'--config',str(CONFIG),'--num-envs','2048','--updates','32','--resume',str(SOURCE/'checkpoint.pt'),'--seed','748701'])
    evaluate=command(EVALUATION,2700,102400,'scripts/evaluate_full_racket_train_v4.py',
        ['--checkpoint',str(TRAIN/'policy.pt'),'--noise','none','--seed','724201'])
    return train,evaluate


def prepare():
    import torch
    assert not any(path.exists()for path in [PLAN,STATE,TRAIN,EVALUATION,VIDEO,SNAPSHOT,VIDEO_SPEC])
    parent=core.read(PARENT/'state.json');assert core.live(parent['queue_pid'],'centered_residual_pilot8_queue02/run.py')
    active=[v for v in core.read(ROOT/'status/GPU_LEDGER.json')['runs']if v['status']=='RUNNING']
    assert len(active)==1 and Path(active[0]['output'])==SOURCE
    assert core.live(active[0]['pid'],SOURCE.name) and core.live(active[0]['child_pid'],SOURCE.name)
    assert not PARENT_EVAL.exists(),'Register before the new frozen assessment starts'
    expected=dict(core.read(PARENT/'PREDECLARED_PLAN.json')['source_hashes']);check_sources(expected)
    # Exercise the corrected ID-chain verifier on a completed real prior resume,
    # including its real nonzero source IDs. This is CPU regression, not new learning.
    old_source=torch.load(REPORTS/'residual_ppo_joint_placement_actor32_n2048_probe8_01/checkpoint.pt',map_location='cpu',weights_only=False)
    old_final=torch.load(REPORTS/'residual_ppo_joint_placement_actor32_n2048_resume32_01/checkpoint.pt',map_location='cpu',weights_only=False)
    old_rows=core.read(REPORTS/'residual_ppo_joint_placement_actor32_n2048_resume32_01/episodes.json')
    regression=verify_episode_chains(old_source,old_final,old_rows,2048)
    old8=core.read(REPORTS/'centered_residual_pilot8_queue01/BASELINE_DECISION_AND_SOURCE_EXIT.json')['selected']['summary']
    old40=core.read(REPORTS/'residual_ppo_joint_placement_actor32_n2048_iter40_fixed_train128_01/metrics.json')
    assert decision(old8,True)=='CONTINUE_STRICT_SAME_CONTRACT_32_UPDATES'
    assert decision(old40,True)=='REVIEW_FROZEN_TRAIN_COLLAPSE'
    assert decision(old8,False)=='REVIEW_MISSING_OR_FAILED_TRAINING_TRACE_AUDIT'
    dependencies=[REPORTS/'residual_ppo_joint_placement_actor32_n2048_probe8_01/checkpoint.pt',
        REPORTS/'residual_ppo_joint_placement_actor32_n2048_resume32_01/checkpoint.pt',
        REPORTS/'residual_ppo_joint_placement_actor32_n2048_resume32_01/episodes.json',
        REPORTS/'centered_residual_pilot8_queue01/BASELINE_DECISION_AND_SOURCE_EXIT.json',
        REPORTS/'residual_ppo_joint_placement_actor32_n2048_iter40_fixed_train128_01/metrics.json']
    save_once(HERE/'ACTUAL_PRIOR_RESUME_CPU_CHECK.json',dict(checked_at=datetime.now(timezone.utc).isoformat(),
        actual_old40_episode_chains=regression,old8_frozen_decision=decision(old8,True),old40_frozen_decision=decision(old40,True),
        source_sha256={str(path.relative_to(ROOT)):core.sha(path)for path in dependencies},
        new_native_execution=False,formal_cases_read=False,formal_acceptance=False))
    for path in [Path(__file__).resolve(),PARENT/'PREDECLARED_PLAN.json',HERE/'ACTUAL_PRIOR_RESUME_CPU_CHECK.json',*dependencies]:
        expected[str(path.relative_to(ROOT))]=core.sha(path)
    check_sources(expected);train,evaluate=commands()
    save_once(PLAN,dict(created_at=datetime.now(timezone.utc).isoformat(),source_hashes=expected,
        verified_live_parent_pid=parent['queue_pid'],verified_live_guard_pid=active[0]['pid'],verified_live_native_pid=active[0]['child_pid'],
        parent_frozen_assessment_not_started=True,source_checkpoint=str(SOURCE/'checkpoint.pt'),
        source_checkpoint_must_be_validated_after_actual_success=True,training_command=train,evaluation_command=evaluate,
        source_iteration=8,additional_updates=32,target_iteration=40,num_envs=2048,distinct_TRAIN_cases=128,copies_per_case=16,
        additional_control_env_steps=STEPS,expected_cumulative_new_learner_steps=CUMULATIVE_STEPS,
        same_config_sha256=core.sha(CONFIG),fixed_base_policy_sha256=core.sha(BASE),
        internal_continuation_gate=dict(minimum_airborne_contacts=16,minimum_conditional_recoveries=8,numerical_failures=0,
            selected_native_training_trace_audit_required=True,scope='Retain a useful baseline before more learning, not formal acceptance.'),
        original_formal_readiness_gate=dict(complete_success_rate=.5,airborne_contact_rate=.7,conditional_recovery_rate=.85,numerical_failures=0),
        decision_order='Wait for actual parent process exit and READY_FOR_VISUAL_REVIEW; independently verify completed pilot and all128 frozen episodes. Preserve a ready frozen policy for formal review; otherwise continue only if both trace audit and internal collapse gate pass.',
        rationale='The8-update pilot includes2 critic-warmup and6 actor updates. Allow32 further whole-body updates if measured frozen performance remains useful; never assume longer learning improves it.',
        restart_semantics='Strict same-contract canonical resume, same2048 lanes, preserved actor/critic/Adam/normalizers/RNG/counters. Real ground restart; no old in-contact PhysX motion is claimed.',
        episode_identifier_verification='Derive each exact source-inclusive to final-exclusive chain from real canonical checkpoint tensors; never require episode0 after a resume.',
        physical_model_rules_cases_hardware_and_original_acceptance_unchanged=True,
        video_case_preselection=[19,23,107,0,8,58],selected_from='Retained from original centered pilot before its new results.',
        video_upload_requires_root_visual_review=True,source_custody='Keep all parent and continuation sources unchanged until both actual workflows finish.',
        resource_authorization='Original shared main GPU0 guard/ledger only, unlimited cumulative authorization, finite job time/step caps, no cloud.',
        formal_cases_read=False,formal_acceptance=False,goal_complete=False))
    print(json.dumps(dict(prepared=str(PLAN),verified_live_native_pid=active[0]['child_pid'],sources=len(expected),actual_prior_chain_check=regression,new_native_job_started=False)),flush=True)



def guarded(command,output,steps,label,expected):
    global child
    assert not stopping and not output.exists();check_sources(expected)
    with (HERE/(label+'.guard.log')).open('x')as log:
        child=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2'),
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(label,guard_pid=child.pid)
        code=child.wait()
    assert code==0 and not stopping,'Guard failed/cancelled; preserve artifacts, no automatic restart'
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
            title=f'固定TRAIN{index}｜{kind}',subtitle='固定已测8版基线｜新增量学习器40｜实际原生全身轨迹',
            terminal_status=('完整扣杀成立'if outcome['complete']else'完整扣杀失败')+('｜合格恢复成立'if outcome['conditional_recovery']else'｜触球与恢复联合条件未通过'),
            limitation=f'{outgoing}；{landed}；稳定窗口{row["jump"]["recovery_s"]:.2f}秒'))
    counts=f'空中{summary["airborne_contacts"]}、条件恢复{summary["conditional_recovery_count"]}、完整{summary["complete"]}'
    return dict(scope='New frozen centered residual learner40 over measured fixed8; original TRAIN128, not changing-policy or formal A/B performance.',
        cases=cases,opening_title='保留已测基线｜新增量PPO40',
        opening_lines=['固定基线8；18关节增量网络从8严格续训32次至40。','全128真实结果：'+counts+'；数值故障'+str(summary['numerical_failures'])+'。','训练前登记六例，全部实际结果保留。'],
        closing_title='固定TRAIN128诊断｜原门槛不变',
        closing_lines=[counts+'；每例为独立原生执行。','固定基线成绩不继承，训练成功不代替冻结表现。','正式达标仍需同一冻结策略通过A/B各200回合。'],
        evaluation_footer='固定8基线＋新增量PPO40｜'+counts+'｜非正式A/B',
        caveats=['Actual native50Hz full-body poses and exact contact/landing events; no generated or interpolated motion.',
            'All six comparison cases were registered before the original centered pilot and retained for this continuation. Not blind/random sampling.',
            'This is a new frozen outer learner40 over fixed measured8, not the earlier mutable-actor learner8/40.',
            'Last valid50Hz sample is explicitly distinguished from exact terminal or landing substeps.'])


def main():
    assert PLAN.exists() and not STATE.exists();plan=core.read(PLAN);check_sources(plan['source_hashes'])
    train,evaluate=commands();assert train==plan['training_command'] and evaluate==plan['evaluation_command']
    parent_launch=core.read(PARENT/'LAUNCH_OBSERVATION.json')
    write('WAITING_FOR_VERIFIED_PARENT',observed_parent_pid=plan['verified_live_parent_pid'])
    while core.live(plan['verified_live_parent_pid'],'centered_residual_pilot8_queue02/run.py'):
        assert not stopping,'Observer cancelled'
        check_sources(plan['source_hashes']);time.sleep(10)
    parent_state=core.read(PARENT/'state.json')
    assert parent_state['state']=='READY_FOR_VISUAL_REVIEW','Parent not complete: '+parent_state['state']
    assert not any(v['status']=='RUNNING'for v in core.read(ROOT/'status/GPU_LEDGER.json')['runs'])
    for name in ['guard_pid','native_child_pid']:
        assert not core.live(parent_launch[name],SOURCE.name),'Prior native source owner remains live'
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
    check_sources(plan['source_hashes'])
    job=guarded(train,TRAIN,STEPS,'RUNNING_NATIVE_CENTERED_RESUME32',plan['source_hashes'])
    receipt=verify_training(job);save_once(HERE/'TRAINING_RECEIPT.json',receipt)
    audit_code=cpu([PYTHON,'scripts/check_full_racket_training_traces_v4.py','--run',str(TRAIN),
        '--output',str(HERE/'TRAINING_TRACE_AUDIT.json')],'trace_audit',required=False)
    save_once(HERE/'TRACE_AUDIT_ATTEMPT.json',dict(exit_code=audit_code,passed=audit_code==0,
        scope='Selected actual changing-policy training traces only; not frozen or formal acceptance.'))
    guarded(evaluate,EVALUATION,102400,'RUNNING_FIXED_CENTERED_TRAIN128',plan['source_hashes'])
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
    check_sources(plan['source_hashes'])
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
