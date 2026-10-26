"""CPU-only fitted-view replay after the actual leg06 train/fixed/video owner exits.

No new policy execution, main-source edits, GPU scheduling, automatic upload,
or inherited visual approval. The existing six native cases remain unchanged.
"""
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
HERE = Path(__file__).resolve().parent
MAIN = ROOT.parent/'unitree_smash'
REPORTS = MAIN/'reports/G07_full_racket_v4'
PARENT = REPORTS/'centered_leg06_after075_intermediates_pilot8_queue01'
SELECTOR = PARENT
TRAIN = REPORTS/'residual_ppo_centered_leg06_after075_intermediates_n2048_probe8_01'
FROZEN = REPORTS/'residual_ppo_centered_leg06_after075_intermediates_n2048_probe8_fixed_train128_01'
POLICY = TRAIN/'policy.pt'
SNAPSHOT = REPORTS/'video_snapshot_centered_leg06_after075_intermediates_pilot8_20261026'
VIDEO = ROOT/'reports/G07_full_racket_v4/video/Current_effect_centered_leg06_pilot8_orthographic_v3_20261026.mp4'
PREVIEW = HERE/'Preview_leg06_pilot8_orthographic_v3.mp4'
PLAN, STATE = HERE/'PLAN.json', HERE/'state.json'
PYTHON = '/home/autodrive/Isaac/env_isaaclab/bin/python'
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('preserved_native40_video_checks', REPORTS/'residual_resume24_queue01/run.py')
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
child = None
stopping = False


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def atomic(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    os.replace(temporary, path)
def write(state, **extra):
    atomic(STATE, dict(state=state, observed_at=datetime.now(timezone.utc).isoformat(),
        observer_pid=os.getpid(), renderer_uses_gpu=False, new_native_execution=False,
        formal_cases_read=False, formal_acceptance=False, goal_complete=False, **extra))
def check_sources(expected):
    for name, digest in expected.items():assert sha(Path(name)) == digest, name


def local_sources():
    paths = [Path(__file__), HERE/'ACTUAL_REFERENCE_SCHEMA_CHECK.json', ROOT/'configs/court_net_v4.json', ROOT/'configs/g07_full_racket_eval_v4_r3.json']
    paths += [ROOT/'scripts'/name for name in ['render_full_racket_actual_landing_labels_v4.py',
        'native_video_framing_v4.py', 'render_full_racket_r3_effect.py', 'render_current_v4_progress.py']]
    paths += [ROOT/'reports/G07_full_racket_v4/orthographic075_iter40_landing_labels_20261026'/name
        for name in ['ROOT_VISUAL_REVIEW.json', 'DECODE_AND_NATIVE_CUSTODY_CHECK.json', 'CAPTION_CHANGE.json']]
    return {str(p): sha(p) for p in paths}


def prepare():
    assert ROOT == Path('/home/autodrive/LZQW/unitree_smash_video_framing')
    assert not any(p.exists() for p in [PLAN, STATE, VIDEO, VIDEO.with_suffix('.json')])
    assert core.live(1768388, 'centered_leg06_after075_intermediates_pilot8_queue01/run.py')
    assert core.live(1768415, TRAIN.name)
    old = read(ROOT/'reports/G07_full_racket_v4/orthographic075_iter40_landing_labels_20261026/ROOT_VISUAL_REVIEW.json')
    assert old['approved_for_authorized_GitHub_video_upload'] and old['root_visually_reviewed_native_event_images'] == 15
    prior_caption = read(ROOT/'reports/G07_full_racket_v4/orthographic075_iter40_landing_labels_20261026/CAPTION_CHANGE.json')
    assert prior_caption['new_renderer_sha256'] == sha(ROOT/'scripts/render_full_racket_actual_landing_labels_v4.py')
    parent_plan = read(PARENT/'PREDECLARED_PLAN.json')
    assert parent_plan['video_preselection_before_new_training'] == [19, 23, 107, 0, 8, 58]
    held = {str(MAIN/name): digest for name, digest in parent_plan['source_hashes'].items()}
    held[str(PARENT/'PREDECLARED_PLAN.json')] = sha(PARENT/'PREDECLARED_PLAN.json')
    check_sources(held)
    atomic(PLAN, dict(created_at=datetime.now(timezone.utc).isoformat(), local_sources=local_sources(),
        original_main_held_sources=held, actual_parent_pid=1768388, actual_native_pid=1768415,
        same_fixed_native_cases=[19, 23, 107, 0, 8, 58],
        launch_condition='Actual parent1768388 exits READY_FOR_VISUAL_REVIEW after original leg06 training/fixed8/audit/video; verify successful native receipts and original128 independently, then render saved poses on CPU.',
        original_GPU_guard_or_ledger_used_by_this_workflow=False, main_runtime_modified=False,
        main_source_custody='Verify the296 main inputs at registration and native admission only. Afterward hold own renderer and completed immutable data; do not extend main runtime ownership while replaying saved geometry.',
        actual_leg06_results_absent_at_registration=not (FROZEN/'metrics.json').exists(),
        old_stage40_visual_review_is_reference_only_not_new_leg06_approval=True,
        dynamic_frames_from_actual_metadata=True, automatic_upload=False,
        new_native_execution=False, formal_acceptance=False, goal_complete=False))
    print(json.dumps(dict(prepared=str(PLAN), main_inputs_checked=len(held), new_GPU_job_started=False)))


def cpu(arguments, label):
    global child
    environment = dict(os.environ, PYTHONPATH='/tmp/unitree_video_deps', CUDA_VISIBLE_DEVICES='',
        OMP_NUM_THREADS='2', LP_NUM_THREADS='2', LIBGL_ALWAYS_SOFTWARE='1',
        __EGL_VENDOR_LIBRARY_FILENAMES='/usr/share/glvnd/egl_vendor.d/50_mesa.json')
    with (HERE/(label+'.log')).open('x') as stream:
        child = subprocess.Popen([PYTHON, *arguments], cwd=ROOT, env=environment,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        write('RUNNING_CPU_'+label.upper(), renderer_pid=child.pid)
        try:code = child.wait(timeout=1800)
        except BaseException:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=30)
            raise
    assert code == 0 and not stopping


def framing_check(metadata, cases):
    assert len(metadata['camera_framing_checks']) == len(cases) == 6
    for case, coverage in zip(cases, metadata['camera_framing_checks']):
        meshes = read(SNAPSHOT/case['run']/'geometry.json')['meshes']
        assert coverage['rendered_meshes'] == len(meshes)
        assert coverage['rendered_bodies'] == len({mesh['body'] for mesh in meshes})
        assert all(view['all_fitted_points_in_viewport'] for view in coverage['views'].values())
    layout = metadata['pane_layout']
    assert layout['top']+layout['height'] < layout['time_band_top']


def main():
    assert PLAN.exists() and not STATE.exists() and not VIDEO.exists()
    plan = read(PLAN)
    expected = {**plan['local_sources'], **plan['original_main_held_sources'], str(PLAN): sha(PLAN)}
    check_sources(expected)
    write('WAITING_FOR_ACTUAL_LEG06_PARENT_EXIT', parent_pid=1768388, new_GPU_job_started=False)
    while core.live(1768388, 'centered_leg06_after075_intermediates_pilot8_queue01/run.py'):
        assert not stopping
        time.sleep(10)
    parent = read(PARENT/'state.json')
    assert parent['state'] == 'READY_FOR_VISUAL_REVIEW' and parent['selected_training_trace_audit_passed']
    check_sources(expected)
    core.verify_job(TRAIN, 2097152)
    receipt = read(PARENT/'TRAINING_RECEIPT.json')
    assert receipt['policy_sha256'] == sha(POLICY) and receipt['checkpoint_sha256'] == sha(TRAIN/'checkpoint.pt')
    assert receipt['actual_control_env_steps'] == 2097152
    assert receipt['actor_normalizer_count'] == receipt['source_actor_normalizer_count'] + 2097152
    assert receipt['critic_normalizer_count'] == 2097152
    assert receipt['fresh_actor_Adam_steps'] == 120 and receipt['fresh_critic_Adam_steps'] == 160
    assert receipt['all2048_fresh_episode_chains_exact']
    assert receipt['source_public_policy_sha256'] == read(PARENT/'PREDECLARED_PLAN.json')['selected_policy_sha256']
    summary = core.verify_evaluation(FROZEN, POLICY)
    assert read(FROZEN/'INPUTS.json')['reset_noise_seed'] == 724201
    sys.path.insert(0, str(MAIN))
    from scripts.initialize_centered_policy_v4 import same_tree
    raw = read(FROZEN/'episodes.json')
    cases = read(SNAPSHOT/'cases.json')['cases']
    assert [c['env_id'] for c in cases] == plan['same_fixed_native_cases']
    assert all(same_tree(c['row'], raw[c['env_id']]) for c in cases)
    assert any(not c['row']['complete_smash'] for c in cases) or summary['complete'] == 128, 'Retain an actual failure when the measured cohort has failures'
    inputs = [p for p in SNAPSHOT.rglob('*') if p.is_file()]
    inputs += [POLICY, TRAIN/'checkpoint.pt', PARENT/'TRAINING_RECEIPT.json']
    inputs += [FROZEN/name for name in ['INPUTS.json', 'metrics.json', 'job.json', 'completed.json', 'episodes.json']]
    native_sources = {str(p): sha(p) for p in inputs}
    atomic(HERE/'ACTUAL_FROZEN8_INPUT_CHECK.json', dict(checked_at=datetime.now(timezone.utc).isoformat(),
        original_all128_summary=summary, six_actual_cases_deep_matched=True, sources=native_sources,
        original_main296_sources_verified_at_admission=True,
        new_native_execution=False, formal_acceptance=False, goal_complete=False))
    expected = {**plan['local_sources'], str(PLAN): sha(PLAN), **native_sources}
    renderer = str(ROOT/'scripts/render_full_racket_actual_landing_labels_v4.py')
    cpu([renderer, '--snapshot', str(SNAPSHOT), '--output', str(PREVIEW), '--preview-only'], 'preview')
    framing_check(read(PREVIEW.with_suffix('.preview.json')), cases)
    cpu([renderer, '--snapshot', str(SNAPSHOT), '--output', str(VIDEO)], 'render')
    check_sources(expected)
    metadata = read(VIDEO.with_suffix('.json'))
    assert not metadata['renderer_uses_gpu'] and not metadata['interpolated_or_generated_motion']
    assert 'llvmpipe' in metadata['renderer'].lower()
    assert all(same_tree(c['row'], raw[c['env_id']]) for c in metadata['cases'])
    framing_check(metadata, cases)
    write('DECODING_ALL_ACTUAL_FRAMES_ON_CPU')
    import imageio.v2 as imageio
    from PIL import Image
    samples = HERE/'decoded_samples'
    samples.mkdir(exist_ok=False)
    total = metadata['frames']
    indices = {0, total//6, total//3, total//2, 2*total//3, 5*total//6, max(0, total-50), total-1}
    reader = imageio.get_reader(VIDEO)
    count = 0
    for frame in reader:
        assert frame.shape == (720, 1280, 3)
        if count in indices:Image.fromarray(frame).save(samples/f'frame_{count:06d}.jpg', quality=94)
        count += 1
    fps = float(reader.get_meta_data()['fps'])
    reader.close()
    assert count == total and fps == 50. and metadata['duration_s'] == count/fps
    for name, digest in metadata['sources'].items():
        assert sha(Path(name) if Path(name).is_absolute() else ROOT/name) == digest
    check_sources(expected)
    verification = dict(checked_at=datetime.now(timezone.utc).isoformat(), video=str(VIDEO), bytes=VIDEO.stat().st_size,
        sha256=sha(VIDEO), all_frames_decoded=True, frames=count, fps=fps, duration_s=count/fps,
        original_all128_summary=summary, exact_same_six_native_cases=True, source_hashes_checked=len(expected),
        root_visual_review_pending=True, automatic_upload_performed=False,
        new_native_execution=False, formal_acceptance=False, goal_complete=False)
    atomic(HERE/'DECODE_AND_NATIVE_CUSTODY_CHECK.json', verification)
    write('READY_FOR_ROOT_VISUAL_REVIEW', verification=verification)


def stop(*_):
    global stopping
    stopping = True
    if child is not None and child.poll() is None:child.terminate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    with (HERE/'observer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.prepare_only:prepare()
        else:
            signal.signal(signal.SIGTERM, stop)
            signal.signal(signal.SIGINT, stop)
            try:main()
            except BaseException as error:
                (HERE/'error.txt').write_text(traceback.format_exc())
                write('FAILED_CPU_VIDEO_ONLY_NATIVE_UNTOUCHED', error=repr(error))
                raise
