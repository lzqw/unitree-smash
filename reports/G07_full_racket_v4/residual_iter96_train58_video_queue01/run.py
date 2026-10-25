"""Render the prospective fixed iter96 TRAIN58 retest on the CPU after the existing GPU workflow."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
REPORTS = ROOT / 'reports/G07_full_racket_v4'
PYTHON = '/home/autodrive/Isaac/env_isaaclab/bin/python'
CORE = REPORTS / 'residual_resume24_queue01/run.py'
module_spec = importlib.util.spec_from_file_location('residual_checks', CORE)
core = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(core)
PARENT_PID = 1576853
PARENT = REPORTS / 'residual_resize2048_queue01'
TRAIN = REPORTS / 'residual_ppo_n2048_resume64_01'
EVALUATION = REPORTS / 'residual_ppo_fixed_train128_iter96_n2048_01'
SPEC = REPORTS / 'video_spec_residual_ppo_iter96_TRAIN58_20261025.json'
SNAPSHOT = REPORTS / 'video_snapshot_residual_ppo_iter96_TRAIN58_20261025'
VIDEO = REPORTS / 'video/Current_effect_residual_PPO_iter96_TRAIN58_20261025.mp4'
STATE = HERE / 'state.json'
PRESELECTED = [58]
stopping = False
child = None


def write(state, **extra):
    value = dict(state=state, queue_pid=os.getpid(), updated_at=datetime.now(timezone.utc).isoformat(),
        preselected_TRAIN_indices=PRESELECTED, source_run=str(EVALUATION), video=str(VIDEO),
        formal_acceptance=False, renderer_uses_gpu=False, **extra)
    core.atomic(STATE, value)
    print(json.dumps({k: v for k, v in value.items() if k != 'source_hashes'}), flush=True)


def hashes():
    paths = [CORE, Path(__file__).resolve(), PARENT / 'run.py',
        ROOT / 'scripts/prepare_full_racket_latest_video_v4.py',
        ROOT / 'scripts/render_full_racket_latest_effect_v4.py',
        ROOT / 'scripts/render_full_racket_r3_effect.py', ROOT / 'scripts/render_current_v4_progress.py']
    return dict(core.source_hashes(), **{str(p): core.sha(p) for p in paths})


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
            title=f'固定TRAIN{index}｜{kind}', subtitle='同一PPO第96次更新策略｜原生全身轨迹',
            terminal_status=('本回合完整扣杀成立' if outcome['complete'] else '本回合完整扣杀失败') +
                ('｜合格恢复成立' if outcome['conditional_recovery'] else '｜接触与恢复联合条件未通过'),
            limitation=f'{outgoing}；{landed}；稳定窗口{row["jump"]["recovery_s"]:.2f}秒'))
    counts = f'空中{summary["airborne_contacts"]}、条件恢复{summary["conditional_recovery_count"]}、完整{summary["complete"]}'
    return dict(scope='Fixed residual PPO iteration96, original TRAIN128; not changing-policy training or formal A/B.',
        policy_sha256=core.sha(policy), preselected_before_fixed_evaluation=True,
        selection_reason='TRAIN58 was selected after the changing-policy TRAIN58_LANE1338_EP2 completion was independently observed at update46, before iter96 frozen assessment. This is a prospective targeted retest, not random sampling or a claim that the training success belongs to one frozen actor.',
        opening_title='TRAIN58训练成功案例｜第96次更新固定复测',
        opening_lines=['TRAIN58曾在变动策略训练中完成全链条；本片检查同一来球的冻结策略。',
            '完整TRAIN128：' + counts + '；数值故障' + str(summary['numerical_failures']) + '。',
            '此例在固定评估前登记；保留实际结果，不继承训练中的成功。'],
        closing_title='固定TRAIN128结果｜原门槛不变',
        closing_lines=[counts + '；本例只代表一例复测，不替代完整128回合统计。',
            '完整球拍、原球场和规则；首次落点与恢复分别显示。',
            '正式达标仍需同一冻结策略完成A/B各200回合。'],
        evaluation_footer='PPO iter96固定TRAIN128｜' + counts + '｜非正式A/B',
        caveats=['Actual native50Hz body poses and exact contact events. No generated or interpolated motion.',
            'TRAIN58 was fixed before this iter96 assessment, after a changing-policy training success. Its actual frozen outcome is retained; it is not blind/random population sampling.',
            'These are fixed-policy evaluation episodes, distinct from the earlier changing-policy TRAIN19_LANE275_EP1 success.',
            'The terminal image uses the last valid50Hz sample, not an invented exact terminal pose.',
            'The original formal A/B acceptance thresholds remain unchanged.'], cases=cases)


def run(command, label):
    global child
    if stopping:
        raise InterruptedError('Cancelled before CPU rendering step')
    environment = dict(os.environ, PYTHONPATH='/tmp/unitree_video_deps', LIBGL_ALWAYS_SOFTWARE='1',
        LP_NUM_THREADS='2', OMP_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='')
    with (HERE / (label + '.log')).open('w') as log:
        child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        code = child.wait(timeout=1800)
    assert code == 0 and not stopping, f'CPU {label} stopped or failed; inspect its log'


def stop(*_):
    global stopping
    stopping = True
    if child is not None and child.poll() is None:
        child.terminate()


def main():
    assert not any(p.exists() for p in [STATE, SPEC, SNAPSHOT, VIDEO])
    assert core.live(PARENT_PID, 'residual_resize2048_queue01/run.py')
    assert not EVALUATION.exists(), 'Preselect before fixed evaluation begins'
    expected = hashes()
    progress = core.read(TRAIN / 'progress.json')
    assert progress['update'] < 96
    core.atomic(HERE / 'PREDECLARED_PLAN.json', dict(
        created_at=datetime.now(timezone.utc).isoformat(),
        verified_live_parent_pid=PARENT_PID, verified_parent_command='residual_resize2048_queue01/run.py',
        training_progress_at_registration={k: v for k, v in progress.items() if k != 'losses'},
        preselected_TRAIN_indices=PRESELECTED, evaluation_output_did_not_exist_at_registration=True,
        source_hashes=expected, renderer_uses_gpu=False, gpu_workflow_untouched=True,
        action='After the existing native workflow actually succeeds, independently verify fixed TRAIN128 and render the selected TRAIN58 first episode on CPU. Decode all frames, then require visual review before upload.',
        formal_cases_read=False, formal_acceptance=False, goal_complete=False))
    write('WAITING_FOR_NATIVE_WORKFLOW', verified_parent_pid=PARENT_PID, source_hashes=expected,
        selection_reason='TRAIN58 is a targeted retest after TRAIN58_LANE1338_EP2 independently completed during learning at update46. Selected before iter96 frozen assessment; not random or before all training.')
    while not stopping and core.live(PARENT_PID, 'residual_resize2048_queue01/run.py'):
        time.sleep(10)
    if stopping:
        raise InterruptedError('Cancelled while waiting; no GPU process touched')
    parent = core.read(PARENT / 'state.json')
    if parent['state'] != 'SUCCEEDED':
        write('SKIPPED_PARENT_NOT_SUCCEEDED', parent_state=parent['state'])
        return
    assert hashes() == expected, 'Source changed; review before rendering'
    policy = TRAIN / 'policy.pt'
    summary = core.verify_evaluation(EVALUATION, policy)
    core.atomic(SPEC, make_spec(summary, policy))
    write('PREPARING_NATIVE_SNAPSHOT', source_hashes=expected)
    run([PYTHON, 'scripts/prepare_full_racket_latest_video_v4.py', '--spec', str(SPEC), '--output', str(SNAPSHOT)], 'prepare')
    write('RENDERING_ON_CPU', source_hashes=expected)
    run([PYTHON, 'scripts/render_full_racket_latest_effect_v4.py', '--snapshot', str(SNAPSHOT), '--output', str(VIDEO)], 'render')
    write('VERIFYING_VIDEO_DECODE', source_hashes=expected)
    verify_code = '''import json,sys
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
p.with_name(p.stem+'_verification.json').write_text(json.dumps(dict(
    video=str(p),sha256=metadata['sha256'],frames_decoded=count,fps=fps,all_frames_decoded=True,
    resolution=[1280,720],renderer_uses_gpu=False,formal_acceptance=False),indent=2)+'\\n')
'''
    run([PYTHON, '-c', verify_code, str(VIDEO)], 'decode')
    assert hashes() == expected
    metadata = core.read(VIDEO.with_suffix('.json'))
    assert metadata['sha256'] == core.sha(VIDEO)
    write('READY_FOR_VISUAL_REVIEW', video_sha256=metadata['sha256'], duration_s=metadata['duration_s'],
        frames=metadata['frames'], previews=metadata['previews'], independent_fixed_TRAIN128=summary,
        note='Inspect contact/first-landing/terminal previews before uploading. Workflow success is not G07 acceptance.')


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        main()
    except InterruptedError as error:
        write('CANCELLED', error=str(error))
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate()
        write('FAILED_RENDER_OR_VALIDATION', error=repr(error))
        raise
