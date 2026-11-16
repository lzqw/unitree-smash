#!/usr/bin/env python3
"""Persistent per-line supervisor with an auditable child process boundary.
It never signals another line and records identity/exit evidence."""
from __future__ import annotations
import argparse, datetime as dt, json, os, pathlib, signal, subprocess, sys, time


def now(): return dt.datetime.now(dt.timezone.utc).isoformat()
def proc_start(pid):
    try:
        fields = pathlib.Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
        return int(fields[19])
    except Exception: return None
def cgroup(pid):
    try: return pathlib.Path(f'/proc/{pid}/cgroup').read_text().splitlines()
    except Exception: return []
def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n')
    tmp.replace(path)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--line', required=True, choices=('supermimic','teacher'))
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--output', type=pathlib.Path, required=True)
    ap.add_argument('--cwd', type=pathlib.Path, required=True)
    ap.add_argument('--command-json', type=pathlib.Path, required=True)
    a=ap.parse_args()
    out=a.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    command=json.loads(a.command_json.read_text())
    if not isinstance(command, list) or not command or not all(isinstance(x,str) for x in command):
        raise SystemExit('command-json must contain a non-empty string list')
    identity={
        'schema':'dual-line-runtime-v1','line':a.line,'run_id':a.run_id,
        'supervisor_pid':os.getpid(),'supervisor_ppid':os.getppid(),
        'supervisor_pgid':os.getpgid(0),'supervisor_sid':os.getsid(0),
        'supervisor_start_ticks':proc_start(os.getpid()),'started_at':now(),
        'cwd':str(a.cwd.resolve()),'command':command,'command_sha256':__import__('hashlib').sha256(json.dumps(command,separators=(',',':')).encode()).hexdigest(),
        'supervisor_cgroup':cgroup(os.getpid()),
    }
    write(out/'identity.json',identity)
    child=subprocess.Popen(command,cwd=str(a.cwd),start_new_session=True,
                           stdout=(out/'stdout.log').open('ab'),stderr=subprocess.STDOUT,
                           env=os.environ.copy())
    # The observer reads this file only; it is never used to signal the child.
    (out/'child.pid').write_text(f'{child.pid}\n')
    identity.update({'child_pid':child.pid,'child_pgid':os.getpgid(child.pid),
                     'child_sid':os.getsid(child.pid),'child_start_ticks':proc_start(child.pid),
                     'child_cgroup':cgroup(child.pid),'child_started_at':now()})
    write(out/'identity.json',identity)
    stopping={'requested':False}
    def stop(sig, _frame):
        if stopping['requested']: return
        stopping['requested']=True
        try: os.killpg(os.getpgid(child.pid), signal.SIGTERM)
        except ProcessLookupError: pass
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
    rc=child.wait()
    result={**identity,'finished_at':now(),'exit_code':rc,
            'terminated_by_signal':(-rc if rc<0 else None),
            'stop_requested':stopping['requested'],
            'terminal':'normal' if rc==0 else ('signal' if rc<0 else 'nonzero')}
    write(out/'exit.json',result)
    return 0 if rc==0 else 1
if __name__=='__main__': raise SystemExit(main())
