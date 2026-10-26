"""CPU replay of the same six cases after the actual fixed_actor_body_gen2 parent exits."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
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
MAIN = ROOT.parent/'unitree_smash'
HERE = Path(__file__).resolve().parent
REPORTS = MAIN/'reports/G07_full_racket_v4'
PARENT = REPORTS/'fixed_actor_body_cem_gen2_queue01'
TRAIN = REPORTS/'fixed_actor_body_cem1280_gen2_01'
FROZEN = REPORTS/'fixed_actor_body_cem1280_gen2_fixed_train128_01'
SPEC = REPORTS/'video_spec_fixed_actor_body_cem_gen2_20261026.json'
SNAPSHOT = REPORTS/'video_snapshot_fixed_actor_body_cem_gen2_20261026'
VIDEO = ROOT/'reports/G07_full_racket_v4/video/Current_effect_fixed_actor_body_cem_gen2_orthographic_v3_20261026.mp4'
PREVIEW = HERE/'Preview_fixed_actor_body_gen2_orthographic_v3.mp4'
PLAN,STATE = HERE/'PLAN.json',HERE/'state.json'
PYTHON = '/home/autodrive/Isaac/env_isaaclab/bin/python'
CASES = [19,23,107,0,8,58]
sys.path.insert(0,str(MAIN))
module_spec = importlib.util.spec_from_file_location('actual_fixed_actor_body_gen2_native_owner',PARENT/'run.py')
owner = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(owner)
core = owner.core
child = None
stopping = False


def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def check(expected):
    for name,digest in expected.items(): assert sha(Path(name)) == digest,name
def save(path,value):
    assert not path.exists(),str(path)
    core.atomic(path,value)
def write(state,**extra):
    core.atomic(STATE,dict(state=state,observer_pid=os.getpid(),observed_at=datetime.now(timezone.utc).isoformat(),
        CPU_only=True,new_native_execution=False,automatic_upload=False,formal_acceptance=False,goal_complete=False,**extra))


def prepare():
    assert not any(p.exists() for p in [PLAN,STATE,SPEC,SNAPSHOT,VIDEO])
    assert core.live(1794481,'fixed_actor_body_cem_gen2_queue01/run.py') and core.live(1794497,TRAIN.name)
    assert not (FROZEN/'metrics.json').exists()
    main_plan = read(PARENT/'PLAN.json')
    held = {str(MAIN/name):digest for name,digest in main_plan['source_hashes'].items()}
    held[str(PARENT/'PLAN.json')] = sha(PARENT/'PLAN.json'); check(held)
    reviewed = ROOT/'reports/G07_full_racket_v4/orthographic_leg06_warmup13_after_native_queue01'
    prior = read(reviewed/'ROOT_VISUAL_REVIEW.json')
    assert prior['approved_for_authorized_GitHub_video_upload'] and read(reviewed/'UPLOAD_RECEIPT.json')['download_verified']
    paths = [Path(__file__),HERE/'ACTUAL_CURRENT13_SCHEMA_CHECK.json',ROOT/'scripts/render_full_racket_actual_landing_labels_v4.py',
        ROOT/'scripts/native_video_framing_v4.py',ROOT/'scripts/render_full_racket_r3_effect.py',
        ROOT/'scripts/render_current_v4_progress.py',ROOT/'configs/court_net_v4.json',
        ROOT/'configs/g07_full_racket_eval_v4_r3.json',reviewed/'ROOT_VISUAL_REVIEW.json',reviewed/'UPLOAD_RECEIPT.json',
        MAIN/'scripts/prepare_full_racket_latest_video_v4.py']
    local = {str(path):sha(path) for path in paths}
    save(PLAN,dict(created_at=datetime.now(timezone.utc).isoformat(),actual_parent_pid=1794481,
        actual_current_native_pid=1794497,main_inputs=held,own_inputs=local,cases=CASES,
        cases_declared_before_new_frozen_results=True,new_training_already_in_progress_at_registration=True,
        native_launch_unchanged=True,new_GPU_job_started=False,
        wait_condition='Actual1794481 exits COMPLETED_FIXED_ACTOR_BODY_SEARCH_AND_COLD128, then independently verify actual completed search receipt and original all128 before rendering.',
        source_custody='Main380 sources checked at registration/admission only. Root must wait ACTUAL_NATIVE_INPUT_CHECK before any later main-runtime revision. After admission only own renderer, snapshot and completed immutable data are held.',
        prior_fixed13_review_is_schema_reference_only=True,automatic_upload=False,
        new_native_execution=False,formal_cases_read=False,formal_acceptance=False,goal_complete=False))
    print(json.dumps(dict(prepared=str(PLAN),main_inputs=len(held),new_GPU_job_started=False)),flush=True)


def cpu(arguments,label,cwd=ROOT):
    global child
    env = dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',PYTHONPATH='/tmp/unitree_video_deps',
        LIBGL_ALWAYS_SOFTWARE='1',LP_NUM_THREADS='2',__EGL_VENDOR_LIBRARY_FILENAMES='/usr/share/glvnd/egl_vendor.d/50_mesa.json')
    with (HERE/(label+'.log')).open('x') as log:
        child = subprocess.Popen([PYTHON,*arguments],cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT)
        write('RUNNING_CPU_'+label.upper(),renderer_pid=child.pid)
        try: code = child.wait(timeout=1800)
        except BaseException:
            if child.poll() is None: child.terminate(); child.wait(timeout=30)
            raise
        assert code == 0 and not stopping


def video_spec(summary):
    import numpy as np
    from quadsmash.evaluation_v4 import independent_episode
    protocol = read(MAIN/'configs/g07_full_racket_eval_v4_r3.json'); protocol['_root'] = str(MAIN)
    raw = read(FROZEN/'episodes.json'); cases = []
    for index in CASES:
        row = raw[index]; outcome = independent_episode(row,protocol)
        velocity = np.asarray(row['outgoing_velocity_mps']); landing = np.asarray(row['first_landing_m'])
        outgoing = f'出球{np.linalg.norm(velocity):.2f}米/秒' if np.isfinite(velocity).all() else '未记录有效出球'
        landed = f'首落[{landing[0]:.2f},{landing[1]:.2f}]米' if np.isfinite(landing).all() else '未观察到首次落点'
        cases.append(dict(run=FROZEN.name,source_type='evaluation',env_id=index,
            title=f'固定TRAIN{index}｜'+('合法空中拍线' if outcome['airborne_contact'] else '未通过合格空中拍线判定'),
            subtitle='保留两层策略网络｜全身参数搜索2代｜原128冷评估',
            terminal_status=('完整扣杀成立' if outcome['complete'] else '完整扣杀失败')+('｜合格恢复成立' if outcome['conditional_recovery'] else '｜触球与恢复联合条件未通过'),
            limitation=f'{outgoing}；{landed}；稳定窗口{row["jump"]["recovery_s"]:.2f}秒'))
    return dict(scope='Actual fixed actor body CEMgen2 TRAIN128, no formal A/B or inherited performance.',cases=cases,
        opening_title='保留策略网络｜全身拦截与起跳搜索',opening_lines=['两层已学策略保留；全身站位、拦截与起跳参数搜索。',
            f'全128真实结果：空中{summary["airborne_contacts"]}、恢复{summary["conditional_recovery_count"]}、完整{summary["complete"]}。',
            '原六例全部保留；完整球拍、原规则及成功条件保持。'],closing_title='固定TRAIN128诊断｜原门槛不变',
        closing_lines=[f'空中{summary["airborne_contacts"]}、条件恢复{summary["conditional_recovery_count"]}、完整{summary["complete"]}。',
            '正式达标仍需同一冻结策略通过A/B各200回合。'],
        evaluation_footer=f'全身搜索2代｜固定原128｜空中{summary["airborne_contacts"]}、恢复{summary["conditional_recovery_count"]}、完整{summary["complete"]}｜非正式A/B',
        caveats=['Actual native50Hz full-body poses and exact contact/landing events; no generated or interpolated motion.',
            'Same six cases declared before new body-search frozen outcomes; training had already started. Not blind or random sampling.',
            'Original0758 andBASE846 actor exports, normalizers, scales and mixers retained; only declared analytic reference parameters searched.',
            'Last valid50Hz sample is explicitly distinguished from exact terminal or landing substeps.'])


def framing(metadata,cases):
    assert len(metadata['camera_framing_checks']) == len(cases) == 6
    for case,coverage in zip(cases,metadata['camera_framing_checks']):
        meshes = read(SNAPSHOT/case['run']/'geometry.json')['meshes']
        assert coverage['rendered_meshes'] == len(meshes)
        assert coverage['rendered_bodies'] == len({m['body'] for m in meshes})
        assert all(view['all_fitted_points_in_viewport'] for view in coverage['views'].values())
    layout = metadata['pane_layout']; assert layout['top']+layout['height'] < layout['time_band_top']


def main():
    assert PLAN.is_file() and not STATE.exists()
    plan = read(PLAN); own = {**plan['own_inputs'],str(PLAN):sha(PLAN)}
    check({**own,**plan['main_inputs']}); write('WAITING_FOR_ACTUAL_BODY_SEARCH_PARENT_EXIT',parent_pid=1794481)
    while core.live(1794481,'fixed_actor_body_cem_gen2_queue01/run.py'):
        assert not stopping; time.sleep(10)
    assert read(PARENT/'state.json')['state'] == 'COMPLETED_FIXED_ACTOR_BODY_SEARCH_AND_COLD128'
    check({**own,**plan['main_inputs']})
    receipt = read(PARENT/'SEARCH_INDEPENDENT_CHECK.json')
    assert receipt['all2560_actual_selected_episode_rows_independently_checked']
    assert receipt['actual_control_env_steps'] == read(TRAIN/'completed.json')['env_steps']
    assert read(TRAIN/'job.json')['status'] == 'SUCCEEDED'
    assert receipt['public_policy_sha256'] == sha(TRAIN/'best_candidate.pt')
    summary = core.verify_evaluation(FROZEN,TRAIN/'best_candidate.pt')
    assert read(FROZEN/'INPUTS.json')['reset_noise_seed'] == 724201
    save(SPEC,video_spec(summary))
    cpu([str(MAIN/'scripts/prepare_full_racket_latest_video_v4.py'),'--spec',str(SPEC),'--output',str(SNAPSHOT)],'snapshot',MAIN)
    from scripts.initialize_centered_policy_v4 import same_tree
    raw = read(FROZEN/'episodes.json'); cases = read(SNAPSHOT/'cases.json')['cases']
    assert [c['env_id'] for c in cases] == CASES and all(same_tree(c['row'],raw[c['env_id']]) for c in cases)
    assert any(not c['row']['complete_smash'] for c in cases) or summary['complete'] == 128
    inputs = [p for p in SNAPSHOT.rglob('*') if p.is_file()]+[SPEC,TRAIN/'best_candidate.pt',PARENT/'SEARCH_INDEPENDENT_CHECK.json']
    inputs += [FROZEN/name for name in ['job.json','completed.json','INPUTS.json','metrics.json','episodes.json']]
    native = {str(path):sha(path) for path in inputs}
    save(HERE/'ACTUAL_NATIVE_INPUT_CHECK.json',dict(checked_at=datetime.now(timezone.utc).isoformat(),
        actual_completed_search=receipt,original_all128_summary=summary,six_actual_rows_deep_matched=True,
        main380_sources_and_PLAN_verified_at_admission=True,sources=native,new_native_execution=False,formal_acceptance=False))
    renderer = str(ROOT/'scripts/render_full_racket_actual_landing_labels_v4.py')
    cpu([renderer,'--snapshot',str(SNAPSHOT),'--output',str(PREVIEW),'--preview-only'],'preview')
    framing(read(PREVIEW.with_suffix('.preview.json')),cases)
    cpu([renderer,'--snapshot',str(SNAPSHOT),'--output',str(VIDEO)],'render')
    metadata = read(VIDEO.with_suffix('.json')); framing(metadata,cases)
    assert not metadata['renderer_uses_gpu'] and not metadata['interpolated_or_generated_motion']
    assert 'llvmpipe' in metadata['renderer'].lower() and all(same_tree(c['row'],raw[c['env_id']]) for c in metadata['cases'])
    import imageio.v2 as imageio
    from PIL import Image
    samples = HERE/'decoded_samples'; samples.mkdir(exist_ok=False)
    total = metadata['frames']; selected = {0,total//6,total//3,total//2,2*total//3,5*total//6,total-50,total-1}
    reader = imageio.get_reader(VIDEO); count = 0
    for frame in reader:
        assert frame.shape == (720,1280,3)
        if count in selected: Image.fromarray(frame).save(samples/f'frame_{count:06d}.jpg',quality=94)
        count += 1
    fps = float(reader.get_meta_data()['fps']); reader.close()
    assert count == total and fps == 50. and metadata['duration_s'] == count/fps
    check({**own,**native})
    for name,digest in metadata['sources'].items(): assert sha(Path(name) if Path(name).is_absolute() else ROOT/name) == digest
    verification = dict(checked_at=datetime.now(timezone.utc).isoformat(),video=str(VIDEO),sha256=sha(VIDEO),bytes=VIDEO.stat().st_size,
        frames=count,duration_s=count/fps,all_frames_decoded=True,same_six_cases=True,original_all128_summary=summary,
        root_visual_review_pending=True,automatic_upload=False,new_native_execution=False,formal_acceptance=False,goal_complete=False)
    save(HERE/'DECODE_AND_NATIVE_CUSTODY_CHECK.json',verification); write('READY_FOR_ROOT_VISUAL_REVIEW',verification=verification)


def stop(*_):
    global stopping
    stopping = True
    if child is not None and child.poll() is None: child.terminate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--prepare-only',action='store_true'); args = parser.parse_args()
    with (HERE/'observer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.prepare_only: prepare()
        else:
            signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
            try: main()
            except BaseException as error:
                (HERE/'error.txt').write_text(traceback.format_exc()); write('FAILED_CPU_VIDEO_ONLY_NATIVE_UNTOUCHED',error=repr(error)); raise
