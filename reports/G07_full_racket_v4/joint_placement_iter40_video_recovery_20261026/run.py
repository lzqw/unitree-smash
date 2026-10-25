"""Preserve the two original selections and render after recovered TRAIN128.

Historical skipped queue states remain unchanged. Neither case selection nor
the actual native evaluation is repeated or replaced by training successes.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
REPORTS=ROOT/'reports/G07_full_racket_v4'
sys.path.insert(0,str(ROOT))


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


recovery=module('joint40_native_recovery',REPORTS/'joint_placement_iter40_recovery_20261026/run.py')
core=recovery.core
videos=[module('original_six_video',REPORTS/'joint_placement_iter40_video_queue01/run.py'),
    module('original_targeted_video',REPORTS/'joint_placement_iter40_targeted_video_queue01/run.py')]
PLAN=HERE/'RECOVERY_PLAN.json';STATE=HERE/'state.json'


def write(state,**extra):
    value=dict(state=state,queue_pid=os.getpid(),updated_at=datetime.now(timezone.utc).isoformat(),
        renderer_uses_gpu=False,formal_cases_read=False,formal_acceptance=False,goal_complete=False,**extra)
    core.atomic(STATE,value);print(json.dumps(value),flush=True)


def prepare():
    assert not PLAN.exists() and not STATE.exists()
    parent=core.read(recovery.STATE)
    assert parent['state']=='RUNNING_FROZEN40_ORIGINAL_TRAIN128'
    assert core.live(parent['queue_pid'],'joint_placement_iter40_recovery_20261026/run.py')
    assert core.live(parent['guard_pid'],recovery.EVALUATION.name)
    expected=dict(core.read(recovery.PLAN)['source_hashes'])
    selections=[]
    for video in videos:
        historical=core.read(video.STATE)
        assert historical['state']=='SKIPPED_PARENT_NOT_SUCCEEDED'
        assert not core.live(historical['queue_pid'],str(video.HERE.relative_to(ROOT)/'run.py'))
        assert not any(path.exists() for path in [video.SPEC,video.SNAPSHOT,video.VIDEO])
        original=core.read(video.PLAN)
        expected.update(original['source_hashes'])
        expected[str(video.PLAN.relative_to(ROOT))]=core.sha(video.PLAN)
        selections.append(dict(original_plan=str(video.PLAN),indices=video.PRESELECTED,
            original_registration_preserved=True,video=str(video.VIDEO)))
    assert [entry['indices']for entry in selections]==[[19,23,107,0,8,58],[65,6]]
    expected[str(Path(__file__).relative_to(ROOT))]=core.sha(Path(__file__))
    expected[str(recovery.PLAN.relative_to(ROOT))]=core.sha(recovery.PLAN)
    recovery.old.check_sources(expected)
    core.atomic(PLAN,dict(created_at=datetime.now(timezone.utc).isoformat(),source_hashes=expected,
        verified_live_recovery_pid=parent['queue_pid'],selections=selections,
        reason='Original video observers skipped because a post-training verifier incorrectly required episode0. New observer consumes the recovered single frozen40 assessment.',
        selection_changes=False,historical_states_modified=False,GPU_jobs_started=False,
        actual_frame_visual_review_required_before_upload=True,formal_acceptance=False,goal_complete=False))
    print(json.dumps(dict(prepared=str(PLAN),sources=len(expected),selections=selections)),flush=True)


def run_cpu(command,label):
    environment=dict(os.environ,PYTHONPATH='/tmp/unitree_video_deps',LIBGL_ALWAYS_SOFTWARE='1',
        LP_NUM_THREADS='2',OMP_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='',
        __EGL_VENDOR_LIBRARY_FILENAMES='/usr/share/glvnd/egl_vendor.d/50_mesa.json')
    with (HERE/(label+'.log')).open('x')as log:
        subprocess.run(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,
            check=True,timeout=1800)


def main():
    assert PLAN.exists() and not STATE.exists()
    plan=core.read(PLAN);recovery.old.check_sources(plan['source_hashes'])
    write('WAITING_FOR_RECOVERED_NATIVE_FROZEN40',verified_parent_pid=plan['verified_live_recovery_pid'])
    while core.live(plan['verified_live_recovery_pid'],'joint_placement_iter40_recovery_20261026/run.py'):
        time.sleep(10)
    parent=core.read(recovery.STATE)
    assert parent['state']=='SUCCEEDED', parent
    recovery.old.check_sources(plan['source_hashes'])
    summary=core.verify_evaluation(recovery.EVALUATION,recovery.TRAIN/'policy.pt')
    results=[]
    for index,video in enumerate(videos):
        assert not any(path.exists()for path in [video.SPEC,video.SNAPSHOT,video.VIDEO])
        core.atomic(video.SPEC,video.make_spec(summary,recovery.TRAIN/'policy.pt'))
        write('PREPARING_NATIVE_SNAPSHOT',selection=index,indices=video.PRESELECTED)
        run_cpu([recovery.PYTHON,'scripts/prepare_full_racket_latest_video_v4.py',
            '--spec',str(video.SPEC),'--output',str(video.SNAPSHOT)],f'prepare{index}')
        write('RENDERING_ON_CPU',selection=index,indices=video.PRESELECTED)
        run_cpu([recovery.PYTHON,'scripts/render_full_racket_latest_effect_v4.py',
            '--snapshot',str(video.SNAPSHOT),'--output',str(video.VIDEO)],f'render{index}')
        write('VERIFYING_ALL_VIDEO_FRAMES',selection=index)
        run_cpu([recovery.PYTHON,'-c',video.DECODE,str(video.VIDEO)],f'decode{index}')
        results.append(dict(video=str(video.VIDEO),sha256=core.sha(video.VIDEO),
            indices=video.PRESELECTED,automatic_upload_performed=False))
    recovery.old.check_sources(plan['source_hashes'])
    write('READY_FOR_VISUAL_REVIEW',videos=results,independent_fixed_TRAIN128=summary,
        historical_skipped_states_preserved=True,automatic_upload_performed=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    with (HERE/'observer.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.prepare_only:prepare()
        else:
            try:main()
            except BaseException as error:
                (HERE/'error.txt').write_text(traceback.format_exc());write('FAILED_VIDEO_RECOVERY',error=repr(error));raise
