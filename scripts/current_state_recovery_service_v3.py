"""Private CPU planning service that never spawns the Isaac entrypoint.

Only inherited local sockets are accepted. A dedicated process group bounds
the service and its candidate workers; CUDA is disabled in that group.
"""
import argparse
import hashlib
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.parallel_landing_solver_v1 import _send,_receive


class CurrentStateRecoveryServiceV3:
    def __init__(self,initargs,audit,source_hashes,*,workers=4,max_seconds=600.):
        if not 1<=workers<=4 or not 120<=max_seconds<=1800:raise ValueError('Bounded CPU planner required')
        self.process=None;self.sock=None;self.closed=False;self.request_id=0;self.max_seconds=max_seconds
        self.last_tick=-1
        path=Path(__file__).resolve();pins=dict(source_hashes)
        for dependency in [path,ROOT/'scripts/current_state_recovery_planner_v3.py']:
            pins[str(dependency)]=hashlib.sha256(dependency.read_bytes()).hexdigest()
        for p,h in pins.items():
            if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:raise ValueError('Changed service input '+p)
        parent,child=socket.socketpair()
        try:
            self.process=subprocess.Popen([sys.executable,str(path),'--worker-fd',str(child.fileno()),'--parent-pid',str(os.getpid())],
                cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),
                stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                pass_fds=(child.fileno(),),start_new_session=True)
            child.close();self.sock=parent
            deadline=time.monotonic()+120
            _send(parent,dict(initargs=initargs,audit=audit,pins=pins,workers=workers,max_seconds=max_seconds),deadline)
            answer=_receive(parent,deadline)
            if (not answer.get('ready') or answer['pid']!=self.process.pid or answer['cuda_visible_devices']!=''
                    or answer['isaac_modules_loaded'] or not answer.get('parent_death_group_cleanup')
                    or answer['worker_sha256']!=pins[str(path)]):
                raise ValueError('Unexpected CPU service startup '+repr(answer))
            self.ready=answer
        except BaseException:
            child.close();parent.close();self.close();raise

    def solve(self,state,tick,plan,*,preceding_stable_ticks=0):
        if self.closed or self.process.poll() is not None:raise RuntimeError('CPU service is not running')
        if type(tick) is not int or tick<=self.last_tick:raise ValueError('Increasing actual planning ticks required')
        self.request_id+=1;deadline=time.monotonic()+self.max_seconds+60
        request=dict(request_id=self.request_id,state=state,tick=tick,plan=plan,stable_ticks=preceding_stable_ticks)
        try:
            _send(self.sock,request,deadline);answer=_receive(self.sock,deadline)
            if answer.get('request_id')!=self.request_id:raise ValueError('Stale planning response')
            if not answer.get('ok'):raise RuntimeError('CPU planning failed: '+str(answer.get('error')))
            result=answer['result'];self.last_tick=tick
            if result['control_tick']!=tick:raise ValueError('Planner clock differs')
            import numpy as np
            for key,value in [('q',state.q),('v',state.v),('contacts',state.contacts),('target',state.previous_command_target)]:
                if not np.array_equal(result['query'][key],value):raise ValueError('Planner substituted query '+key)
            result['cpu_service']=dict(pid=self.process.pid,request_id=self.request_id,cuda_visible_devices='',
                isaac_modules_loaded=answer['isaac_modules_loaded'])
            if answer['isaac_modules_loaded']:raise ValueError('CPU service imported Isaac')
            return result
        except BaseException:self.close();raise

    def close(self):
        if self.closed:return
        self.closed=True
        if self.sock is not None:
            try:
                if self.process is not None and self.process.poll() is None:
                    _send(self.sock,dict(close=True),time.monotonic()+1.)
            except (OSError,TimeoutError,EOFError):pass
            self.sock.close()
        if self.process is not None:
            try:self.process.wait(timeout=5.)
            except subprocess.TimeoutExpired:
                try:os.killpg(self.process.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:self.process.wait(timeout=5.)
                except subprocess.TimeoutExpired:
                    try:os.killpg(self.process.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    self.process.wait(timeout=5.)


def worker(fd,parent_pid):
    import ctypes
    if os.getpgrp()!=os.getpid():raise ValueError('Dedicated CPU service process group required')
    def terminate_group(signum,frame):
        signal.signal(signal.SIGTERM,signal.SIG_DFL)
        os.killpg(os.getpgrp(),signal.SIGTERM)
    signal.signal(signal.SIGTERM,terminate_group)
    # If the guarded Isaac parent exits, terminate this separate CPU group as
    # well. Otherwise its new session would outlive a native process-group kill.
    if ctypes.CDLL(None,use_errno=True).prctl(1,signal.SIGTERM,0,0,0)!=0:
        raise OSError(ctypes.get_errno(),'Cannot install parent-death signal')
    if os.getppid()!=parent_pid:terminate_group(signal.SIGTERM,None)
    from scripts.current_state_recovery_planner_v3 import CurrentStateRecoveryPlannerV3
    sock=socket.socket(fileno=fd);planner=None
    def isaac_loaded():return any(n=='isaaclab' or n.startswith(('isaaclab.','omni.isaac')) for n in sys.modules)
    try:
        payload=_receive(sock,time.monotonic()+120)
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or isaac_loaded():raise ValueError('Require CPU-only service')
        for p,h in payload['pins'].items():
            if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:raise ValueError('Changed worker input '+p)
        planner=CurrentStateRecoveryPlannerV3(payload['initargs'],payload['audit'],
            workers=payload['workers'],max_seconds=payload['max_seconds'])
        _send(sock,dict(ready=True,pid=os.getpid(),cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
            isaac_modules_loaded=isaac_loaded(),parent_death_group_cleanup=True,
            worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
            time.monotonic()+120)
        last=0
        while True:
            request=_receive(sock,time.monotonic()+7200)
            if request.get('close'):break
            if request['request_id']!=last+1:raise ValueError('Request sequence changed')
            last=request['request_id']
            try:
                result=planner.solve(request['state'],request['tick'],request['plan'],
                    preceding_stable_ticks=request['stable_ticks'])
                response=dict(ok=True,request_id=last,result=result,isaac_modules_loaded=isaac_loaded())
            except Exception as error:response=dict(ok=False,request_id=last,error=repr(error))
            _send(sock,response,time.monotonic()+60)
    except (EOFError,ConnectionError):pass
    finally:
        if planner is not None:planner.close()
        sock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker-fd',type=int,required=True)
    p.add_argument('--parent-pid',type=int,required=True);a=p.parse_args();worker(a.worker_fd,a.parent_pid)
