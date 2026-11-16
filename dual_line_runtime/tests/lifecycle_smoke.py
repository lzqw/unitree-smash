#!/usr/bin/env python3
from __future__ import annotations
import json, os, pathlib, signal, subprocess, sys, time
root=pathlib.Path(__file__).resolve().parents[1]
runs=root/'runs'/'lifecycle_smoke'
runs.mkdir(parents=True,exist_ok=True)
cmd=[sys.executable,'-c',"import time; time.sleep(30)"]
(runs/'a-command.json').write_text(json.dumps(cmd)); (runs/'b-command.json').write_text(json.dumps(cmd))
a=subprocess.Popen([str(root/'bin/line_supervisor.py'),'--line','supermimic','--run-id','smoke-a','--output',str(runs/'a'),'--cwd','/tmp','--command-json',str(runs/'a-command.json')])
b=subprocess.Popen([str(root/'bin/line_supervisor.py'),'--line','teacher','--run-id','smoke-b','--output',str(runs/'b'),'--cwd','/tmp','--command-json',str(runs/'b-command.json')])
# supervisors may start after command files are written; wait for identity and assert distinct sessions/groups.
for _ in range(50):
    if (runs/'a'/'identity.json').exists() and (runs/'b'/'identity.json').exists(): break
    time.sleep(.1)
ia=json.loads((runs/'a'/'identity.json').read_text()); ib=json.loads((runs/'b'/'identity.json').read_text())
assert ia['child_pgid'] != ib['child_pgid'] and ia['child_sid'] != ib['child_sid']
os.kill(a.pid, signal.SIGTERM)
assert a.wait(timeout=5) != None
# b must remain alive after stopping a.
time.sleep(.5); assert b.poll() is None
b.send_signal(signal.SIGTERM); b.wait(timeout=5)
assert (runs/'a'/'exit.json').exists() and (runs/'b'/'exit.json').exists()
(runs/'result.json').write_text(json.dumps({'passed':True,'distinct_pgid':True,'distinct_sid':True,'stop_a_preserved_b':True},indent=2)+'\n')
print(json.dumps(json.loads((runs/'result.json').read_text())))
