"""Exercise a real CPU planning request and parent-exit worker cleanup.

The child intentionally exits after saving its result without calling close.
The outer process checks the service and candidate workers stop executing.
Only this test's own process group is affected; no Isaac process is started.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_native_replanned_suffix_v1 import validate
from scripts.check_constrained_recovery_suffix_v1 import source_context,state_from_row
from scripts.search_preserved_early_support_v1 import initialize
from scripts.current_state_recovery_service_v1 import CurrentStateRecoveryServiceV1


def process_identity(pid):
    try:
        fields=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=pid,state=fields[0],process_group=int(fields[2]),start_ticks=fields[19])
    except FileNotFoundError:return None


def descendants(pid):
    result=[];pending=[pid]
    while pending:
        current=pending.pop();row=process_identity(current)
        if row is None:continue
        result.append(row)
        p=Path('/proc',str(current),'task',str(current),'children')
        if p.exists():pending.extend(int(x) for x in p.read_text().split())
    return result


def child(source_path,output):
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    source=json.loads(source_path.read_text());start,expected=validate(source)
    _,knots,_,audit,initargs=source_context(source['configuration']['source']);initialize(*initargs)
    tick=700;row=source['best']['result']['trajectory'][tick-start-1];state=state_from_row(row)
    plan=dict(original_knots_rad=knots,corrections=[dict(control_tick=start,
        previous_command=source['query']['target'],knots_rad=source['best']['corrections_rad'])])
    pins=dict(source['source_sha256'])
    for path in [source_path.resolve(),Path(__file__).resolve(),ROOT/'scripts/current_state_recovery_service_v1.py',
                 ROOT/'scripts/current_state_recovery_planner_v1.py']:pins[str(path)]=sha(path)
    service=CurrentStateRecoveryServiceV1(initargs,audit,pins,max_seconds=180.,workers=2)
    result=service.solve(state,tick,plan)
    if not result['accepted']:raise ValueError('Known complete suffix did not pass through service')
    trajectory=result['prediction']['trajectory']
    exact=all(np.array_equal(np.asarray([r[k] for r in trajectory]),expected[k][tick-start:])
        for k in ['q','v','target','contacts'])
    if not exact:raise ValueError('Service changed the full model prediction')
    owned=descendants(service.process.pid)
    if len(owned)<2 or any(r['process_group']!=service.process.pid for r in owned):
        raise ValueError('Expected isolated service and candidate workers')
    output.write_text(json.dumps(dict(scope=__doc__,ready=service.ready,
        control_tick=tick,completed_model_steps=result['prediction']['completed_model_steps'],
        independent_replay_exact=result['independent_replay_exact'],reference_suffix_exact=exact,
        query_preserved=True,wall_seconds=result['wall_seconds'],cpu_service=result['cpu_service'],
        owned_processes=owned,source_sha256=pins,native_execution=False,valid_expert_labels=False),indent=2)+'\n')
    # Exercise the actual parent-death path rather than substituting normal
    # shutdown. The CPU service must clean up its own group on this exit.
    os._exit(0)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--child',action='store_true');a=p.parse_args()
    if a.output.exists():raise ValueError('Fresh result required')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.child:child(a.source,a.output);return
    stage=a.output.with_suffix('.child.json')
    if stage.exists():raise ValueError('Preserve prior child evidence')
    with a.output.with_suffix('.child.log').open('x') as log:
        proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--source',str(a.source.resolve()),
            '--output',str(stage.resolve()),'--child'],cwd=ROOT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=proc.wait(timeout=270.)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=10.);raise
    if code!=0:raise ValueError('CPU service child failed; inspect '+str(a.output.with_suffix('.child.log')))
    result=json.loads(stage.read_text());until=time.monotonic()+10.;states=[]
    while True:
        states=[];active=[]
        for prior in result['owned_processes']:
            now=process_identity(prior['pid'])
            if now is not None and now['start_ticks']!=prior['start_ticks']:
                states.append(dict(pid=prior['pid'],state='original_process_exited_pid_reused'));continue
            states.append(now or dict(pid=prior['pid'],state='exited'))
            if now is not None and now['state'] not in ['Z','X']:active.append(now)
        if not active:break
        if time.monotonic()>=until:raise ValueError('CPU descendants survived parent exit: '+repr(active))
        time.sleep(.1)
    result.update(parent_exit_cleanup_verified=True,post_parent_exit_processes=states,
        all_original_cpu_processes_inactive=True,normal_close_not_used=True)
    result['source_sha256'][str(stage.resolve())]=hashlib.sha256(stage.read_bytes()).hexdigest()
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['reference_suffix_exact','completed_model_steps','wall_seconds',
        'parent_exit_cleanup_verified','all_original_cpu_processes_inactive']}))


if __name__=='__main__':main()
