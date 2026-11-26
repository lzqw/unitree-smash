"""One guarded native-teacher Recovery update and paired full-tail fixed128 evaluation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from scripts.run_supermimic_recovery_baseline import (
    ROOT, approach_jobs, dump, gpu_state, stop_native, summarize, verify)


def run_stage(command, folder, lock, status):
    for attempt in range(1, 10):
        dest = folder.with_name(f'{folder.name}_{attempt:02d}')
        while True:
            free, gpu = gpu_state()
            if not gpu - approach_jobs() and free >= 9000 and shutil.disk_usage(folder.parent).free > 15 * 2**30:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    pass
            dump(status, dict(state='waiting_for_resources', stage=folder.name, free_mib=free))
            time.sleep(15)
        try:
            with dest.with_suffix('.log').open('xb') as log:
                child = subprocess.Popen([sys.executable, '-u', *command, '--output', str(dest)],
                                         cwd=ROOT, env=dict(os.environ, PYTHONPATH='.:scripts'),
                                         stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
                interrupted = False
                try:
                    while child.poll() is None:
                        free, gpu = gpu_state()
                        if free < 5000:
                            # Child owns this process group. Release CUDA immediately for Approach.
                            try:
                                os.killpg(child.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                            child.wait()
                            raise RuntimeError(f'Recovery stopped for GPU reserve ({free} MiB free)')
                        if gpu - approach_jobs() - {child.pid}:
                            interrupted = True
                            break
                        dump(status, dict(state='running', stage=folder.name, pid=child.pid, free_mib=free))
                        time.sleep(3)
                finally:
                    stop_native(child)
                if interrupted:
                    continue
                if child.returncode:
                    raise RuntimeError(f'{dest} exited {child.returncode}')
                return dest
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
    raise RuntimeError(f'Repeated GPU preemption at {folder}')


def main():
    p = argparse.ArgumentParser()
    for name in ('checkpoint','companion','training-panel','output','gpu-lock','baseline'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--existing-probe', type=Path)
    a = p.parse_args()
    verify([a.checkpoint, a.companion])
    a.output.mkdir(parents=True, exist_ok=True)
    status = a.output/'status.json'
    native = ['scripts/evaluate_supermimic_v1.py','--headless','--physics-device','cuda:0','--device','cuda:0',
              '--kit_args=--renderer/multiGpu/enabled=false','--recovery-full-tail']
    try:
        with a.gpu_lock.open('a') as lock:
            probe = a.existing_probe or run_stage([*native,'--checkpoint',str(a.checkpoint),'--recovery-training-probes',
                               str(a.training_panel),'--recovery-probe-companion',str(a.companion)],
                              a.output/'probe', lock, status)
            from scripts.score_recovery_probe_round import score_folder
            score = score_folder(probe)
            dump(a.output/'pilot_comparison.json', score)
            if not score['expand']:
                dump(status, dict(state='pilot_complete_no_expansion', comparison=score))
                return
            probes = [probe]
            for shard in range(1,8):
                probes.append(run_stage([*native,'--checkpoint',str(a.checkpoint),
                                         '--recovery-training-probes',str(a.training_panel),
                                         '--recovery-probe-shard',str(shard),
                                         '--recovery-probe-companion',str(a.companion)],
                                        a.output/f'shard{shard}',lock,status))
            inputs = [arg for folder in probes for arg in ('--probe',str(folder))]
            fitted = run_stage(['scripts/distill_native_recovery.py',*inputs,
                                '--checkpoint',str(a.checkpoint)], a.output/'fit', lock, status)
            result = run_stage([*native,'--checkpoint',str(fitted/'policy.pt'),'--research-fixed128'],
                               a.output/'fixed128', lock, status)
        after = summarize(result)
        before = summarize(a.baseline)
        if json.loads((result/'initial_states.json').read_text()) != json.loads((a.baseline/'initial_states.json').read_text()):
            raise ValueError('Fixed panel reset states differ')
        def by_case(folder):
            rows = json.loads((folder/'episodes_summary.json').read_text())
            out = {r['case_id']: r for r in rows}
            assert len(out)==128
            return out
        b, c = by_case(a.baseline), by_case(result)
        assert b.keys()==c.keys()
        flips = {}
        for k in ('qualified_airborne_string_contact','strict_physical_recovery','physical_recovery_success',
                  'body_ground_seen','robot_net_fault','ball_side_complete_except_recovery','complete5'):
            flips[k] = {name:[i for i in b if bool(b[i][k])==x and bool(c[i][k])==y]
                        for name,x,y in [('gained',False,True),('lost',True,False)]}
        dump(a.output/'comparison.json', dict(before=before, after=after, flips=flips))
        dump(status, dict(state='complete', baseline=str(a.baseline), candidate=str(result)))
    except Exception as exc:
        dump(status, dict(state='failed', error=str(exc)))
        raise


if __name__=='__main__':
    main()
