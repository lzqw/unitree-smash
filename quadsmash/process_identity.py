"""Dependency-free Linux identity checks prevent signaling a recycled PID."""
import hashlib
import json
import os
from pathlib import Path

def identity(pid):
    base=Path('/proc')/str(pid)
    stat=(base/'stat').read_text().rsplit(') ',1)[1].split()
    cmdline=(base/'cmdline').read_bytes()
    return dict(pid=pid,start_ticks=int(stat[19]),uid=base.stat().st_uid,
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        cmdline_sha256=hashlib.sha256(cmdline).hexdigest(),cwd=os.readlink(base/'cwd'))

def write_identity(run_dir,pid):
    directory=Path(run_dir).resolve()
    record=dict(process=identity(pid),run_dir=str(directory))
    temporary=directory/'process.json.tmp'
    temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(directory/'process.json')

def checked_pid(run_dir):
    directory=Path(run_dir).resolve()
    record=json.loads((directory/'process.json').read_text())
    if record['run_dir']!=str(directory):raise RuntimeError('process record belongs to a different run')
    pid=int((directory/'run.pid').read_text().strip())
    if pid!=record['process']['pid'] or pid==os.getpid():raise RuntimeError('invalid recorded process PID')
    if identity(pid)!=record['process']:raise RuntimeError('stale or changed process identity; refusing to signal')
    return pid
