"""Run the reviewed centered residual pilot, independent TRAIN128 and real video.

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

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
REPORTS=ROOT/'reports/G07_full_racket_v4';sys.path.insert(0,str(ROOT))
PYTHON='/home/autodrive/Isaac/env_isaaclab/bin/python'
spec=importlib.util.spec_from_file_location('preserved_native_checks',REPORTS/'residual_resume24_queue01/run.py')
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
TRAIN=REPORTS/'residual_ppo_centered_base8_n2048_probe8_01'
EVALUATION=REPORTS/'residual_ppo_centered_base8_n2048_probe8_fixed_train128_01'
BASE=REPORTS/'residual_ppo_joint_placement_actor32_n2048_probe8_01/policy.pt'
CONFIG=ROOT/'configs/g07_full_racket_residual_ppo_centered_v4.json'
PLAN=HERE/'PREDECLARED_PLAN_V2.json';STATE=HERE/'state.json'
VIDEO=REPORTS/'video/Current_effect_centered_residual_base8_probe8_20261026.mp4'
SNAPSHOT=REPORTS/'video_snapshot_centered_residual_base8_probe8_20261026'
VIDEO_SPEC=REPORTS/'video_spec_centered_residual_base8_probe8_20261026.json'
PRESELECTED=[19,23,107,0,8,58]
STEPS=8*2048*128
stopping=False;child=None


def save_once(path,value):
    assert not path.exists(),'Preserve existing evidence: '+str(path)
    core.atomic(path,value)


def write(state,**extra):
    value=dict(state=state,queue_pid=os.getpid(),updated_at=datetime.now(timezone.utc).isoformat(),
        training_output=str(TRAIN),evaluation_output=str(EVALUATION),
        formal_cases_read=False,formal_acceptance=False,goal_complete=False,**extra)
    core.atomic(STATE,value);print(json.dumps(value),flush=True)


def sources():
    paths=list(ROOT.glob('quadsmash/**/*.py'))+list(ROOT.glob('configs/*.json'))+list(ROOT.glob('assets/*.usd'))
    paths += [ROOT/'scripts'/name for name in ['run_gpu_job.py','train_full_racket_residual_ppo_v4.py',
        'evaluate_full_racket_train_v4.py','export_full_racket_train_csv_v4.py','check_centered_residual_cpu_v4.py',
        'check_full_racket_training_traces_v4.py','prepare_full_racket_latest_video_v4.py',
        'render_full_racket_latest_effect_v4.py','render_full_racket_r3_effect.py','render_current_v4_progress.py']]
    paths += [Path(__file__).resolve(),BASE,HERE/'BASELINE_DECISION_AND_SOURCE_EXIT.json',
        HERE/'MAIN_CPU_TEST_RESULT.xml',HERE/'MAIN_REAL_BASE_CPU_CHECK.json',
        REPORTS/'residual_resume24_queue01/run.py',REPORTS/'joint_placement_iter40_video_queue01/run.py']
    protocol=core.read(ROOT/'configs/g07_full_racket_eval_v4_r3.json')
    paths.append(ROOT/protocol['training_suite']['path'])
    paths += [BASE.parent/name for name in ['job.json','completed.json','INPUTS.json','metrics.json']]
    paths += [Path(name)for name in core.read(BASE.parent/'asset_dependencies.json')['resolved_layers']]
    return {str(p.relative_to(ROOT))if p.is_relative_to(ROOT)else str(p):core.sha(p)for p in sorted(set(paths))}


def check_sources(expected):
    for path,digest in expected.items():assert core.sha(ROOT/path)==digest,'Pinned source changed: '+path


def command(output,seconds,steps,script,arguments):
    return [PYTHON,str(ROOT/'scripts/run_gpu_job.py'),'--goal','G07',
        '--budget-window','g07-full-racket-unlimited-20261024','--max-seconds',str(seconds),
        '--env-steps-cap',str(steps),'--output',str(output),'--require-artifact','completed.json','--',
        PYTHON,str(ROOT/script),'--output',str(output),*arguments,'--headless']


def commands():
    train=command(TRAIN,7200,STEPS,'scripts/train_full_racket_residual_ppo_v4.py',
        ['--base-checkpoint',str(BASE),'--config',str(CONFIG),'--num-envs','2048','--updates','8','--seed','748701'])
    evaluate=command(EVALUATION,2700,102400,'scripts/evaluate_full_racket_train_v4.py',
        ['--checkpoint',str(TRAIN/'policy.pt'),'--noise','none','--seed','724201'])
    return train,evaluate


def prepare():
    import xml.etree.ElementTree as ET
    assert not any(path.exists()for path in [PLAN,STATE,TRAIN,EVALUATION,VIDEO,SNAPSHOT,VIDEO_SPEC])
    decision=core.read(HERE/'BASELINE_DECISION_AND_SOURCE_EXIT.json');assert decision['selected']['name']=='frozen8'
    assert decision['original_old_source_owners_exited'] and decision['original_ledger_has_no_running_native_job']
    assert not any(v['status']=='RUNNING'for v in core.read(ROOT/'status/GPU_LEDGER.json')['runs'])
    suite=ET.parse(HERE/'MAIN_CPU_TEST_RESULT.xml').getroot().find('testsuite')
    assert suite.attrib['tests']=='83' and suite.attrib['failures']==suite.attrib['errors']=='0'
    checked=core.read(HERE/'MAIN_REAL_BASE_CPU_CHECK.json')
    assert all(v['zero_new_residual_vs_fixed_base_bitwise_equal']for v in checked['actual_saved_input_checks'])
    assert checked['baseline_source']['source_policy_sha256']==core.sha(BASE)
    config=core.read(CONFIG);assert config['base_controller_mode']=='frozen_public_policy'
    assert config['joint_scale_rad']==[.03]*12+[.15]*6 and config['episode_trace']
    train,evaluate=commands();expected=sources();check_sources(expected)
    save_once(PLAN,dict(created_at=datetime.now(timezone.utc).isoformat(),source_hashes=expected,
        candidate_source_commit='12de0e7',baseline_policy=str(BASE),baseline_policy_sha256=core.sha(BASE),
        baseline_fixed_TRAIN128=decision['selected']['summary'],
        training_command=train,evaluation_command=evaluate,config_sha256=core.sha(CONFIG),
        fresh_outer_actor_critic_normalizers_Adam_RNG_and_counters=True,
        fixed_base_actor_and_normalizer=True,trainable_action_joints=18,
        new_leg_increment_bound_rad=.03,new_arm_increment_bound_rad=.15,
        updates=8,num_envs=2048,all_original_TRAIN_cases=128,copies_per_case=16,
        control_env_steps=STEPS,critic_warmup_updates=2,
        source_custody='All current runtime/config/selected-source files stay unchanged through the actual training, frozen assessment and CPU video.',
        training_trace_selection=config['episode_trace'],
        trace_audit_failure_policy='Retain explicit failure and still assess the frozen policy; do not substitute missing training motion or claim a passed trace audit.',
        video_preselection_before_new_training=PRESELECTED,original_formal_thresholds_unchanged=True,
        physical_asset_rules_cases_hardware_and_acceptance_unchanged=True,
        resource_authorization='Original main GPU0 guard/ledger, unlimited cumulative authorization, finite per-job caps; no candidate-local ledger/cloud.',
        formal_cases_read=False,formal_acceptance=False,goal_complete=False))
    print(json.dumps(dict(prepared=str(PLAN),sources=len(expected),new_native_job_started=False)),flush=True)


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
    import torch
    from quadsmash.evaluation_v4 import independent_episode
    inputs=core.read(TRAIN/'INPUTS.json');metrics=core.read(TRAIN/'metrics.json')
    assert inputs['initialization']['mode']=='fresh_zero_mean_actor_over_frozen_public_base'
    assert inputs['resume_checkpoint_sha256'] is None and inputs['initial_actor_checkpoint_sha256'] is None
    assert inputs['contract']['base_controller_mode']=='frozen_public_policy'
    assert inputs['contract']['base_policy_sha256']==core.sha(BASE)
    assert inputs['contract']['target_event_requires_actual_attack'] and not inputs['contract']['fixed_upper_uv_position_shaping']
    native=torch.load(TRAIN/'checkpoint.pt',map_location='cpu',weights_only=False)
    policy=torch.load(TRAIN/'policy.pt',map_location='cpu',weights_only=False)
    base=torch.load(BASE,map_location='cpu',weights_only=False)
    assert native['iter']==8 and native['native']['canonical_restart'] and native['native']['total_timesteps']==STEPS
    assert native['native']['environment']['num_envs']==2048
    assert not native['native']['environment']['tensors']['episode_length_buf'].any()
    assert all(torch.isfinite(v).all()for v in native['model_state_dict'].values()if torch.is_tensor(v))
    assert all(torch.isfinite(v).all()for state in native['optimizer_state_dict']['state'].values()for v in state.values()if torch.is_tensor(v))
    assert policy['native_execution'] and policy['residual_base_mode']=='frozen_public_policy'
    assert not policy['performance_inherited']
    assert policy['base_controller']['residual_actor_jit']==base['residual_actor_jit']
    assert policy['residual_training_metadata']['base_policy_sha256']==core.sha(BASE)
    assert metrics['completed_updates']==len(metrics['history'])==8
    assert [entry['update']for entry in metrics['history']]==list(range(1,9))
    assert metrics['history'][-1]['actual_control_env_steps']==STEPS
    assert core.sha(TRAIN/'policy.pt')==metrics['policy_sha256']
    protocol=core.read(ROOT/'configs/g07_full_racket_eval_v4_r3.json');protocol['_root']=str(ROOT)
    rows=core.read(TRAIN/'episodes.json');checked=[independent_episode(row,protocol)for row in rows]
    counts=dict(completed_episodes=len(rows),air=sum(v['airborne_contact']for v in checked),
        conditional_recovery_count=sum(v['conditional_recovery']for v in checked),complete=sum(v['complete']for v in checked),
        numerical_failures=sum(row['numerical_failures']for row in rows))
    assert counts==metrics['counts'] and counts['numerical_failures']==0
    assert {row['env_id']for row in rows if row['episode_index']==0}==set(range(2048))
    assert all(row['ground_init'] and row['source_training_index']==row['env_id']%128 for row in rows)
    return dict(checked_at=datetime.now(timezone.utc).isoformat(),actual_control_env_steps=STEPS,
        actual_GPU_hours=job['charged_gpu_hours'],checkpoint_sha256=core.sha(TRAIN/'checkpoint.pt'),
        policy_sha256=core.sha(TRAIN/'policy.pt'),fixed_base_policy_sha256=core.sha(BASE),
        independent_changing_policy_counts=counts,frozen_policy_performance=False,formal_acceptance=False,goal_complete=False)


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
            title=f'固定TRAIN{index}｜{kind}',subtitle='固定已测8版基线｜新增量学习器8｜实际原生全身轨迹',
            terminal_status=('完整扣杀成立'if outcome['complete']else'完整扣杀失败')+('｜合格恢复成立'if outcome['conditional_recovery']else'｜触球与恢复联合条件未通过'),
            limitation=f'{outgoing}；{landed}；稳定窗口{row["jump"]["recovery_s"]:.2f}秒'))
    counts=f'空中{summary["airborne_contacts"]}、条件恢复{summary["conditional_recovery_count"]}、完整{summary["complete"]}'
    return dict(scope='New frozen centered residual learner8 over measured fixed8; original TRAIN128, not changing-policy or formal A/B performance.',
        cases=cases,opening_title='保留已测基线｜新增量PPO8',
        opening_lines=['固定基线8；全新18关节增量网络完成8次更新。','全128真实结果：'+counts+'；数值故障'+str(summary['numerical_failures'])+'。','训练前登记六例，全部实际结果保留。'],
        closing_title='固定TRAIN128诊断｜原门槛不变',
        closing_lines=[counts+'；每例为独立原生执行。','固定基线成绩不继承，训练成功不代替冻结表现。','正式达标仍需同一冻结策略通过A/B各200回合。'],
        evaluation_footer='固定8基线＋新增量PPO8｜'+counts+'｜非正式A/B',
        caveats=['Actual native50Hz full-body poses and exact contact/landing events; no generated or interpolated motion.',
            'All six comparison cases were registered before this new centered training. Not blind/random sampling.',
            'This is a new frozen outer learner8 over fixed measured8, not the earlier mutable-actor learner8/40.',
            'Last valid50Hz sample is explicitly distinguished from exact terminal or landing substeps.'])


def main():
    assert PLAN.exists() and not STATE.exists();plan=core.read(PLAN);check_sources(plan['source_hashes'])
    train,evaluate=commands();assert train==plan['training_command'] and evaluate==plan['evaluation_command']
    job=guarded(train,TRAIN,STEPS,'RUNNING_NATIVE_CENTERED_PILOT8',plan['source_hashes'])
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
        selected_training_trace_audit_passed=audit_code==0,video=str(VIDEO),video_sha256=core.sha(VIDEO),
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
