"""Prospective frozen40 retest of both newly complete changing-TRAIN cases.

Selected from actual training updates34/35 before fixed40 outcomes exist.
Retain both outcomes; these clips cannot recreate the unrecorded training
motion or establish a frozen population/formal success rate.
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
REPORTS = ROOT / 'reports/G07_full_racket_v4'
PYTHON = '/home/autodrive/Isaac/env_isaaclab/bin/python'
CORE = REPORTS / 'residual_resume24_queue01/run.py'
spec = importlib.util.spec_from_file_location('original_fixed_TRAIN_checks', CORE)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
PARENT = REPORTS / 'joint_placement_resume32_queue01'
TRAINING_SUCCESS = REPORTS / 'joint_placement_first_training_complete_20261026/REPORT.json'
TRAIN = REPORTS / 'residual_ppo_joint_placement_actor32_n2048_resume32_01'
EVALUATION = REPORTS / 'residual_ppo_joint_placement_actor32_n2048_iter40_fixed_train128_01'
SPEC = REPORTS / 'video_spec_residual_ppo_joint_placement_iter40_TRAIN65_6_20261026.json'
SNAPSHOT = REPORTS / 'video_snapshot_residual_ppo_joint_placement_iter40_TRAIN65_6_20261026'
VIDEO = REPORTS / 'video/Current_effect_residual_PPO_joint_placement_iter40_TRAIN65_6_20261026.mp4'
STATE = HERE / 'state.json'
PLAN = HERE / 'PREDECLARED_PLAN.json'
PRESELECTED = [65, 6]
stopping = False
child = None


def check_sources(expected):
    for name, digest in expected.items():
        assert core.sha(ROOT / name) == digest, 'Source changed: ' + name


def write(state, **extra):
    value = dict(state=state, queue_pid=os.getpid(), updated_at=datetime.now(timezone.utc).isoformat(),
        preselected_TRAIN_indices=PRESELECTED, evaluation_output=str(EVALUATION),
        video=str(VIDEO), renderer_uses_gpu=False, formal_cases_read=False,
        formal_acceptance=False, goal_complete=False, **extra)
    core.atomic(STATE, value)
    print(json.dumps({k: v for k, v in value.items() if k != 'source_hashes'}), flush=True)


def prepare():
    assert not any(p.exists() for p in [PLAN, STATE, EVALUATION, SPEC, SNAPSHOT, VIDEO])
    parent = core.read(PARENT / 'state.json')
    assert parent['state'] == 'RUNNING_SAME_REWARD_RESUME32'
    assert core.live(parent['queue_pid'], 'joint_placement_resume32_queue01/run.py')
    assert core.live(parent['guard_pid'], TRAIN.name)
    original = core.read(TRAINING_SUCCESS)
    assert [row['source_training_index'] for row in original['complete_episodes']] == PRESELECTED
    assert original['snapshot_update'] == 35 and original['all_closed_training_counts']['complete'] == 2
    expected = dict(core.read(PARENT / 'PREDECLARED_PLAN.json')['source_hashes'])
    for path in [Path(__file__).resolve(), PARENT / 'PREDECLARED_PLAN.json', TRAINING_SUCCESS,
                 ROOT / 'scripts/prepare_full_racket_latest_video_v4.py',
                 ROOT / 'scripts/render_full_racket_latest_effect_v4.py',
                 ROOT / 'scripts/render_full_racket_r3_effect.py',
                 ROOT / 'scripts/render_current_v4_progress.py']:
        expected[str(path.relative_to(ROOT))] = core.sha(path)
    check_sources(expected)
    progress = core.read(TRAIN / 'progress.json')
    assert 8 < progress['update'] < 40
    # Check the already installed CPU renderer without rendering any invented
    # trajectory or taking a GPU context from the existing native learner.
    smoke = '''import json,moderngl,imageio_ffmpeg
from pathlib import Path
from PIL import ImageFont
ctx=moderngl.create_standalone_context(backend='egl',device_index=1)
renderer=ctx.info['GL_RENDERER'];assert 'llvmpipe' in renderer
font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
ImageFont.truetype(font,20)
ffmpeg=imageio_ffmpeg.get_ffmpeg_exe();assert Path(ffmpeg).is_file()
print(json.dumps(dict(renderer=renderer,font=font,ffmpeg=ffmpeg,renderer_uses_gpu=False)))
ctx.release()
'''
    smoke_environment = dict(os.environ, PYTHONPATH='/tmp/unitree_video_deps', LIBGL_ALWAYS_SOFTWARE='1',
        LP_NUM_THREADS='2', OMP_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='',
        __EGL_VENDOR_LIBRARY_FILENAMES='/usr/share/glvnd/egl_vendor.d/50_mesa.json')
    renderer_check = json.loads(subprocess.check_output([PYTHON, '-c', smoke], cwd=ROOT,
        env=smoke_environment, text=True, timeout=30))
    plan = dict(created_at=datetime.now(timezone.utc).isoformat(),
        verified_live_parent_pid=parent['queue_pid'], source_hashes=expected,
        training_progress_at_registration={k: v for k, v in progress.items() if k != 'losses'},
        preselected_TRAIN_indices=PRESELECTED,
        selected_after_actual_training_success_before_fixed40_assessment=True,
        training_success_report_sha256=core.sha(TRAINING_SUCCESS),
        fixed_assessment_not_started_at_registration=True,
        original_TRAIN128_all_results_required=True, source_run=str(EVALUATION),
        video=str(VIDEO), renderer_uses_gpu=False,
        action='Wait for actual learner40 and all frozen TRAIN128 outcomes. Independently recompute them, then render BOTH selected TRAIN65/6 fixed first episodes with side/top views, actual contact, landing and recovery, regardless of whether either fixed retest succeeds. Decode all frames and leave for visual review before upload.',
        scope='Targeted prospective frozen40 retest of TRAIN65/6, selected from actual changing-policy successes in updates34/35. These are new fixed executions, not the unrecorded successful training motions. No random sampling or formal A/B score.',
        formal_cases_read=False, formal_acceptance=False, goal_complete=False)
    core.atomic(PLAN, plan)
    core.atomic(HERE / 'PREPARATION_CHECKS.json', dict(
        checked_at=datetime.now(timezone.utc).isoformat(), verified_live_parent_pid=parent['queue_pid'],
        source_fingerprints_verified=len(expected), fixed_assessment_not_started=True,
        both_actual_training_success_cases_selected=True, renderer_uses_gpu=False,
        native_or_video_render_job_executed_by_this_preparation=False,
        actual_CPU_renderer_capability_check=renderer_check,
        plan_sha256=core.sha(PLAN), formal_acceptance=False, goal_complete=False))
    print(json.dumps(dict(prepared=str(PLAN), source_hashes=len(expected),
        parent_pid=parent['queue_pid'], preselected=PRESELECTED)), flush=True)


def make_spec(summary, policy):
    import numpy as np
    from quadsmash.evaluation_v4 import independent_episode
    protocol = core.read(ROOT / 'configs/g07_full_racket_eval_v4_r3.json')
    protocol['_root'] = str(ROOT)
    rows = core.read(EVALUATION / 'episodes.json')
    cases = []
    for index in PRESELECTED:
        row = rows[index]
        assert row['source_training_index'] == row['env_id'] == index and row['episode_index'] == 0
        outcome = independent_episode(row, protocol)
        kind = ('合法空中拍线击球' if outcome['airborne_contact'] else
            ('拍框接触' if row['frame_hit'] else ('拍线接触、未通过空中判定' if row['string_hit'] else '没有球拍接触')))
        velocity = np.asarray(row['outgoing_velocity_mps'])
        outgoing = f'出球{np.linalg.norm(velocity):.2f}米/秒' if np.isfinite(velocity).all() else '未记录有效出球'
        landing = np.asarray(row['first_landing_m'])
        landed = f'首落[{landing[0]:.2f},{landing[1]:.2f}]米' if np.isfinite(landing).all() else '未观察到首次落点'
        cases.append(dict(run=EVALUATION.name, source_type='evaluation', env_id=index,
            title=f'冻结复测TRAIN{index}｜{kind}', subtitle='联合奖励新学习器40｜训练成功来球的固定复测',
            terminal_status=('本回合完整扣杀成立' if outcome['complete'] else '本回合完整扣杀失败') +
                ('｜合格恢复成立' if outcome['conditional_recovery'] else '｜接触与恢复联合条件未通过'),
            limitation=f'{outgoing}；{landed}；稳定窗口{row["jump"]["recovery_s"]:.2f}秒'))
    counts = f'空中{summary["airborne_contacts"]}、条件恢复{summary["conditional_recovery_count"]}、完整{summary["complete"]}'
    return dict(scope='Frozen new joint-placement-reward PPO learner after40 updates, original TRAIN128. No changing-policy or formal A/B footage.',
        policy_sha256=core.sha(policy), preselected_before_fixed_evaluation=True,
        selection_reason='Both TRAIN65/6 were selected after actual changing-policy training completions at updates34/35, before fixed40 assessment. Preserve both actual retest outcomes. Not random sampling and not the original training-success footage.',
        opening_title='训练成功来球｜冻结40复测',
        opening_lines=['同一固定策略重新执行TRAIN65、6。',
            '完整TRAIN128：' + counts + '；数值故障' + str(summary['numerical_failures']) + '。',
            '由训练34/35成功选定，保留本次全部真实结果。'],
        closing_title='新固定TRAIN128结果｜原门槛不变',
        closing_lines=[counts + '；不将多个独立回合拼成成功。',
            '完整球拍、原球场和规则；实际接触、首落与恢复分别显示。',
            '正式达标仍需同一冻结策略完成A/B各200回合。'],
        evaluation_footer='PPO40训练成功来球复测｜' + counts + '｜非训练原片',
        caveats=['Actual native50Hz full-body poses and exact contact events; no generated or interpolated motion.',
            'Both cases were selected from actual training successes after updates34/35, before fixed40 outcomes existed. This is not blind/random sampling.',
            'These are new frozen40 executions. The original successful training episodes crossed multiple actor/normalizer updates and had no saved full-motion trajectory.',
            'This learner starts at iteration0 with transferred public actor32 only; it is not the original PPO iter8/32/96.',
            'Full128 outcomes come from this one exported frozen policy; training successes are not substituted.',
            'The last valid50Hz sample is shown at the terminal image, not an invented exact terminal pose.',
            'Original formal acceptance thresholds remain unchanged.'], cases=cases)


def run_cpu(command, label):
    global child
    assert not stopping
    environment = dict(os.environ, PYTHONPATH='/tmp/unitree_video_deps', LIBGL_ALWAYS_SOFTWARE='1',
        LP_NUM_THREADS='2', OMP_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='',
        __EGL_VENDOR_LIBRARY_FILENAMES='/usr/share/glvnd/egl_vendor.d/50_mesa.json')
    with (HERE / (label + '.log')).open('x') as log:
        child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        try:
            code = child.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            raise
    assert code == 0 and not stopping, 'CPU render/decode failed or cancelled: ' + label


DECODE = '''import hashlib,json,sys
from pathlib import Path
import imageio.v2 as imageio
p=Path(sys.argv[1]);metadata=json.loads(p.with_suffix('.json').read_text())
with imageio.get_reader(str(p)) as reader:
    fps=reader.get_meta_data()['fps'];count=0
    for frame in reader:
        assert frame.shape==(720,1280,3);count+=1
assert count==metadata['frames'] and fps==50
assert metadata['native_recorded_motion'] and not metadata['renderer_uses_gpu']
assert 'llvmpipe' in metadata['renderer']
digest=hashlib.sha256(p.read_bytes()).hexdigest();assert digest==metadata['sha256']
p.with_name(p.stem+'_verification.json').write_text(json.dumps(dict(
    video=str(p),sha256=digest,frames_decoded=count,fps=fps,all_frames_decoded=True,
    resolution=[1280,720],renderer_uses_gpu=False,formal_acceptance=False),indent=2)+chr(10))
'''


def stop(*_):
    global stopping
    stopping = True
    if child is not None and child.poll() is None:
        child.terminate()


def main():
    assert PLAN.exists() and not any(p.exists() for p in [STATE, SPEC, SNAPSHOT, VIDEO])
    plan = core.read(PLAN)
    check_sources(plan['source_hashes'])
    write('WAITING_FOR_FROZEN_JOINT_REWARD_TRAIN128', verified_parent_pid=plan['verified_live_parent_pid'])
    while not stopping and core.live(plan['verified_live_parent_pid'], 'joint_placement_resume32_queue01/run.py'):
        time.sleep(10)
    if stopping:
        raise InterruptedError('Video observer cancelled; native jobs untouched')
    parent = core.read(PARENT / 'state.json')
    if parent['state'] != 'SUCCEEDED':
        write('SKIPPED_PARENT_NOT_SUCCEEDED', parent_state=parent['state'])
        return
    check_sources(plan['source_hashes'])
    policy = TRAIN / 'policy.pt'
    summary = core.verify_evaluation(EVALUATION, policy)
    core.atomic(SPEC, make_spec(summary, policy))
    write('PREPARING_NATIVE_SNAPSHOT')
    run_cpu([PYTHON, 'scripts/prepare_full_racket_latest_video_v4.py', '--spec', str(SPEC), '--output', str(SNAPSHOT)], 'prepare')
    write('RENDERING_ON_CPU')
    run_cpu([PYTHON, 'scripts/render_full_racket_latest_effect_v4.py', '--snapshot', str(SNAPSHOT), '--output', str(VIDEO)], 'render')
    write('VERIFYING_ALL_VIDEO_FRAMES')
    run_cpu([PYTHON, '-c', DECODE, str(VIDEO)], 'decode')
    check_sources(plan['source_hashes'])
    write('READY_FOR_VISUAL_REVIEW', video_sha256=core.sha(VIDEO),
        independent_fixed_TRAIN128=summary, automatic_upload_performed=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    with (HERE / 'observer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.prepare_only:
            prepare()
        else:
            signal.signal(signal.SIGTERM, stop)
            signal.signal(signal.SIGINT, stop)
            try:
                main()
            except InterruptedError as error:
                write('CANCELLED', error=str(error))
                sys.exit(2)
            except BaseException as error:
                (HERE / 'error.txt').write_text(traceback.format_exc())
                write('FAILED_VALIDATION_OR_CPU_VIDEO', error=repr(error))
                raise
