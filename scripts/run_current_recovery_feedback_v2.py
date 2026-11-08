"""After a verified current-state search, check full feedback and its end state.

Bounded CPU-only continuation. Never retries a lost predecessor, launches
native physics, promotes an incomplete path or creates expert labels.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def identity(pid):
    try:
        return Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()[19]
    except FileNotFoundError:
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['search', 'configuration', 'fixture-run', 'output', 'coordination']:
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    read = lambda f: json.loads(f.read_text())
    if a.coordination.exists() or a.output.exists():
        raise ValueError('Fresh continuation outputs required')
    current = read(a.search/'status.json')
    done = current['state'] == 'COMPLETED_ACTUAL_RECOVERY_SEQUENCE_SEARCH'
    pid = current.get('pid')
    start_ticks = None
    if not done:
        arguments = Path('/proc', str(pid), 'cmdline').read_bytes().split(b'\0')
        arguments = [x.decode() for x in arguments if x]
        if 'scripts/search_current_native_recovery_v4.py' not in arguments:
            raise ValueError('Unexpected live predecessor')
        observed = Path(arguments[arguments.index('--output')+1]).resolve()
        if observed != a.search.resolve():
            raise ValueError('Predecessor writes a different search')
        start_ticks = identity(pid)
        if start_ticks is None:
            raise ValueError('Predecessor identity unavailable')
    files = [Path(__file__), a.configuration, a.search/'inputs.json',
        ROOT/'scripts/audit_replanned_recovery_feedback_v1.py', ROOT/'scripts/audit_feedback_end_observation_v2.py']
    pins = {str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    a.coordination.mkdir(parents=True)
    def status(name, **values):
        f = a.coordination/'status.json'
        temporary = f.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(state=name,pid=os.getpid(),
            time_utc=datetime.now(timezone.utc).isoformat(),**values),indent=2)+'\n')
        temporary.replace(f)
    (a.coordination/'inputs.json').write_text(json.dumps(dict(source_sha256=pins,
        search_pid=pid,search_start_ticks=start_ticks,max_wait_seconds=3060,
        max_feedback_seconds=2000,native_execution=False,training=False),indent=2)+'\n')
    try:
        deadline = time.monotonic()+3060
        status('WAITING_VERIFIED_CURRENT_SEARCH',search_pid=pid)
        while read(a.search/'status.json')['state'] != 'COMPLETED_ACTUAL_RECOVERY_SEQUENCE_SEARCH':
            if identity(pid) != start_ticks:
                raise ValueError('Predecessor ended without completion; not restarting')
            if time.monotonic() > deadline:
                raise TimeoutError('Bounded observation expired; not restarting predecessor')
            time.sleep(5)
        for f,h in pins.items():
            if hashlib.sha256(Path(f).read_bytes()).hexdigest() != h:
                raise ValueError('Changed continuation source '+f)
        search = read(a.search/'result.json')
        if not any(r['model_path_complete'] and r['failure'] is None for r in search['results']):
            status('COMPLETED_NO_FULL_MODEL_CANDIDATE',search_result=search)
            return
        env = dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        command = ['bash','scripts/python.sh','-u','scripts/audit_replanned_recovery_feedback_v1.py',
            '--configuration',str(a.configuration.resolve()),'--fixture-run',str(a.fixture_run.resolve()),
            '--search',str(a.search.resolve()),'--output',str(a.output.resolve())]
        status('RUNNING_CURRENT_PLAN_FEEDBACK')
        with (a.coordination/'feedback.log').open('x') as log:
            subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=2000)
        end = a.coordination/'end_observation.json'
        command = ['bash','scripts/python.sh','scripts/audit_feedback_end_observation_v2.py',
            '--search',str(a.search.resolve()),'--feedback',str(a.output.resolve()),'--output',str(end.resolve())]
        with (a.coordination/'end_observation.log').open('x') as log:
            subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
        status('COMPLETED_CURRENT_FEEDBACK_AND_END_CHECK',result=read(a.output/'result.json'),
            end_observation=read(end)['results'],native_success=False,valid_expert_labels=False)
    except BaseException as error:
        status('FAILED',error=repr(error))
        raise


if __name__ == '__main__':
    main()
