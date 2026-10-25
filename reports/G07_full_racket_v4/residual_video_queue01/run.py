"""Prepare a CPU replay of three preselected cases after the pilot TRAIN evaluation."""
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
module_spec = importlib.util.spec_from_file_location('residual_continuation', CORE)
core = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(core)
SPEC = REPORTS / 'video_spec_residual_ppo_iter8_20261025.json'
SNAPSHOT = REPORTS / 'video_snapshot_residual_ppo_iter8_20261025'
VIDEO = REPORTS / 'video/Current_effect_residual_PPO_iter8_20261025.mp4'
STATE = HERE / 'state.json'
PRESELECTED = [23, 107, 0]
stopping = False
child = None


def write(state, **extra):
    data = dict(state=state, queue_pid=os.getpid(),
        updated_at=datetime.now(timezone.utc).isoformat(),
        preselected_TRAIN_indices=PRESELECTED, source_run=str(core.PRIOR_EVAL),
        video=str(VIDEO), formal_acceptance=False, renderer_uses_gpu=False, **extra)
    core.atomic(STATE, data)
    print(json.dumps({k: v for k, v in data.items() if k != 'source_hashes'}), flush=True)


def hashes():
    files = [CORE, Path(__file__).resolve(),
        ROOT / 'scripts/prepare_full_racket_latest_video_v4.py',
        ROOT / 'scripts/render_full_racket_latest_effect_v4.py',
        ROOT / 'scripts/render_full_racket_r3_effect.py',
        ROOT / 'scripts/render_current_v4_progress.py',
        ROOT / 'configs/court_net_v4.json', ROOT / 'configs/g07_full_racket_eval_v4_r3.json']
    files += sorted(ROOT.glob('quadsmash/**/*.py'))
    return {str(p): core.sha(p) for p in files}


def make_spec(run, summary, policy):
    from quadsmash.evaluation_v4 import independent_episode
    import numpy as np
    protocol = core.read(ROOT / 'configs/g07_full_racket_eval_v4_r3.json')
    protocol['_root'] = str(ROOT)
    rows = core.read(run / 'episodes.json')
    cases = []
    for index in PRESELECTED:
        row = rows[index]
        assert row['source_training_index'] == index and row['env_id'] == index
        result = independent_episode(row, protocol)
        contact = ('合法空中拍线击球' if result['airborne_contact'] else
                   ('拍框接触' if row['frame_hit'] else ('拍线接触、未通过合法空中判定' if row['string_hit'] else '没有球拍接触')))
        velocity = np.asarray(row['outgoing_velocity_mps'])
        outgoing = f'出球{np.linalg.norm(velocity):.2f}米/秒' if np.isfinite(velocity).all() else '未记录有效出球'
        landing = np.asarray(row['first_landing_m'])
        landing_text = f'首落[{landing[0]:.2f},{landing[1]:.2f}]米' if np.isfinite(landing).all() else '未观察到首次落点'
        recovered = '合格恢复成立' if result['conditional_recovery'] else '接触与恢复联合条件未通过'
        cases.append(dict(run=run.name, source_type='evaluation', env_id=index,
            title=f'固定TRAIN{index}｜{contact}',
            subtitle='同一PPO第8次更新策略｜独立回合原生全身轨迹',
            terminal_status=('本回合完整扣杀成立' if result['complete'] else '本回合完整扣杀失败') + '｜' + recovered,
            limitation=f'{outgoing}；{landing_text}；稳定窗口{row["jump"]["recovery_s"]:.2f}秒'))
    counts = f'空中{summary["airborne_contacts"]}、条件恢复{summary["conditional_recovery_count"]}、完整{summary["complete"]}'
    return dict(scope='Fixed residual PPO iteration8 on the original TRAIN128; preselected cases23,107,0 match the earlier baseline video. Not formal A/B.',
        policy_sha256=core.sha(policy), preselected_before_evaluation=True,
        opening_title='残差PPO第8次更新｜固定案例实际效果',
        opening_lines=['同一固定策略，依次展示TRAIN23、TRAIN107、TRAIN0。',
            '完整TRAIN128：' + counts + '；数值故障' + str(summary['numerical_failures']) + '。',
            '侧视和俯视同步播放真实全身轨迹、触球和首次落点。'],
        closing_title='固定TRAIN128结果｜继续联合优化',
        closing_lines=[counts + '；这三个回合保持独立。',
            '案例编号与上一段基线视频相同，失败结果按实保留。',
            '最终仍需原规则与两组各200回合独立验收。'],
        evaluation_footer='PPO iter8固定TRAIN128｜' + counts + '｜非正式A/B',
        caveats=['Actual native50Hz whole-body poses, with exact native contact events; no generated or interpolated motion.',
            'TRAIN indices23,107,0 were fixed before this evaluation finished and match the earlier baseline video.',
            'This frozen policy is distinct from changing-policy training episodes and later continuation weights.',
            'Terminal pictures use the last valid50Hz sample, not a fabricated exact terminal substep.',
            f'FullTRAIN128 counts: {counts}; numerical failures{summary["numerical_failures"]}.'],
        cases=cases)


def run(command, label, environment=None):
    global child
    if stopping:
        raise InterruptedError('Cancelled before CPU rendering step')
    with (HERE / (label + '.log')).open('w') as log:
        child = subprocess.Popen(command, cwd=ROOT, env=environment,
            stdout=log, stderr=subprocess.STDOUT)
        code = child.wait(timeout=1800)
    assert code == 0, f'{label} failed with exit{code}; inspect its log'


def stop(*_):
    global stopping
    stopping = True
    if child is not None and child.poll() is None:
        child.terminate()


def main():
    assert not STATE.exists() and not SPEC.exists() and not SNAPSHOT.exists() and not VIDEO.exists()
    assert core.live(core.PRIOR_PID, 'residual_eval_queue_1250/run.py')
    expected = hashes()
    write('WAITING_FOR_VERIFIED_FIXED_TRAIN_EVALUATION',
          verified_previous_queue_pid=core.PRIOR_PID, source_hashes=expected)
    while not stopping and core.live(core.PRIOR_PID, 'residual_eval_queue_1250/run.py'):
        time.sleep(10)
    if stopping:
        raise InterruptedError('Cancelled while waiting; no GPU job touched')
    assert core.read(core.PRIOR_QUEUE / 'state.json')['state'] == 'SUCCEEDED'
    assert hashes() == expected, 'Native evaluation or renderer source changed'
    policy = core.PILOT / 'policy.pt'
    summary = core.verify_evaluation(core.PRIOR_EVAL, policy)
    core.atomic(SPEC, make_spec(core.PRIOR_EVAL, summary, policy))
    write('PREPARING_NATIVE_SNAPSHOT', source_hashes=expected)
    run([PYTHON, 'scripts/prepare_full_racket_latest_video_v4.py', '--spec', str(SPEC),
         '--output', str(SNAPSHOT)], 'prepare')
    environment = dict(os.environ, PYTHONPATH='/tmp/unitree_video_deps',
        LIBGL_ALWAYS_SOFTWARE='1', LP_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='')
    write('RENDERING_ON_CPU', source_hashes=expected)
    run([PYTHON, 'scripts/render_full_racket_latest_effect_v4.py', '--snapshot', str(SNAPSHOT),
         '--output', str(VIDEO)], 'render', environment)
    write('VERIFYING_VIDEO_DECODE', source_hashes=expected)
    verify_code = '''import json,sys
from pathlib import Path
import imageio.v2 as imageio
p=Path(sys.argv[1]);metadata=json.loads(p.with_suffix('.json').read_text())
with imageio.get_reader(str(p)) as reader:
    fps=reader.get_meta_data()['fps'];count=0;shape=None
    for frame in reader:
        assert frame.shape==(720,1280,3);shape=frame.shape;count+=1
assert count==metadata['frames'] and fps==50
assert metadata['native_recorded_motion'] and not metadata['renderer_uses_gpu']
assert 'llvmpipe' in metadata['renderer']
p.with_name(p.stem+'_verification.json').write_text(json.dumps(dict(
    video=str(p),frames_decoded=count,fps=fps,all_frames_decoded=True,
    renderer_uses_gpu=False,formal_acceptance=False),indent=2)+'\\n')
'''
    run([PYTHON, '-c', verify_code, str(VIDEO)], 'decode', environment)
    assert hashes() == expected
    metadata = core.read(VIDEO.with_suffix('.json'))
    assert metadata['sha256'] == core.sha(VIDEO)
    write('READY_FOR_VISUAL_REVIEW', video_sha256=metadata['sha256'],
        duration_s=metadata['duration_s'], frames=metadata['frames'],
        previews=metadata['previews'], independent_fixed_TRAIN128=summary,
        note='View the rendered native previews, then upload the video and exact evidence to the existing GitHub release.')


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
        write('FAILED', error=repr(error))
        raise
