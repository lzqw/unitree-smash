"""CPU-only, lane-owned MPC subprocesses for synchronous simulation batches.

No native integration yet. Each lane always belongs to the same process;
observed reference memory and reset operations never cross lane boundaries.
Private inherited Unix sockets carry only trusted local Python objects. Workers
start this standalone script, never the Isaac entrypoint, and cannot use CUDA.
"""
import argparse
import os
from pathlib import Path
import pickle
import socket
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MAX_MESSAGE = 128 * 1024 * 1024


def _read(sock, count, deadline):
    pieces = []
    while count:
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError('MPC worker response deadline exceeded')
        sock.settimeout(remaining)
        data = sock.recv(min(count, 1024*1024))
        if not data:
            raise EOFError('MPC worker connection closed')
        pieces.append(data); count -= len(data)
    return b''.join(pieces)


def _receive(sock, deadline):
    size = struct.unpack('!Q', _read(sock, 8, deadline))[0]
    if not 0 < size <= MAX_MESSAGE:
        raise ValueError('MPC worker message size outside bound')
    return pickle.loads(_read(sock, size, deadline))


def _send(sock, value, deadline):
    data = pickle.dumps(value, protocol=5)
    if len(data) > MAX_MESSAGE:
        raise ValueError('MPC worker message too large')
    remaining = deadline-time.monotonic()
    if remaining <= 0:
        raise TimeoutError('MPC worker send deadline exceeded')
    sock.settimeout(remaining)
    sock.sendall(struct.pack('!Q', len(data))+data)


def build_controller(configuration):
    import numpy as np
    from scripts.whole_body_support_qp_v1 import PositionPDInterface, SupportSettings
    from scripts.landing_support_mpc_v2 import LandingMPCSettingsV2
    from scripts.landing_full_command_interval_mpc_v1 import LandingFullCommandIntervalMPCV1
    from scripts.landing_feasible_command_cost_mpc_v1 import LandingFeasibleCommandCostMPCV1
    family = configuration['family']
    if family not in ('full_interval', 'feasible_command_cost'):
        raise ValueError('Undeclared controller family')
    interface = PositionPDInterface(**{k:np.asarray(v) if k!='friction' else v
                                      for k,v in configuration['interface'].items()})
    kwargs = dict(qp_settings=SupportSettings(**configuration['qp_settings']),
                  settings=LandingMPCSettingsV2(**configuration['settings']),
                  arm_target_rate_rad_s=configuration['arm_target_rate_rad_s'])
    cls = LandingFullCommandIntervalMPCV1
    if family == 'feasible_command_cost':
        cls = LandingFeasibleCommandCostMPCV1
        kwargs['command_change_weight'] = configuration['command_change_weight']
    controller = cls(interface, **kwargs)
    controller.set_posture_target(np.asarray(configuration['posture_target']))
    return controller


def execute(controller, request):
    lane = request['lane']
    if request['operation'] == 'reset':
        controller.reset(lane=lane)
        return None
    if request['operation'] != 'observe':
        raise ValueError('Unknown MPC worker operation')
    state = request['state']
    reference = controller.admit_command_target(state, request['reference'])
    controller.observe(state, reference, lane=lane, control_tick=request['control_tick'])
    if not request['solve']:
        return None
    return controller.solve(state, reference, lane=lane,
                            control_tick=request['control_tick'], force=request['force'])


class ParallelLandingSolverV1:
    def __init__(self, configuration, *, workers=4, timeout_s=180.):
        if type(workers) is not int or not 1 <= workers <= 8 or not 0 < timeout_s <= 600:
            raise ValueError('Require 1–8 workers and a bounded positive batch deadline')
        self.workers = workers; self.timeout_s = timeout_s
        self.processes = []; self.sockets = []; self.closed = False
        try:
            for _ in range(workers):
                parent, child = socket.socketpair()
                env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                           CUDA_VISIBLE_DEVICES='')
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                    '--worker-fd', str(child.fileno())], cwd=ROOT, env=env,
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL, pass_fds=(child.fileno(),))
                child.close(); self.processes.append(process); self.sockets.append(parent)
                deadline = time.monotonic()+timeout_s
                _send(parent, configuration, deadline)
                ready = _receive(parent, deadline)
                if not ready.get('ready') or ready['pid'] != process.pid or ready['cuda_visible_devices'] != '':
                    raise RuntimeError('MPC worker did not initialize correctly: '+repr(ready))
        except BaseException:
            self.close(); raise

    def batch(self, requests):
        if self.closed:
            raise RuntimeError('MPC pool is closed')
        groups = [[] for _ in range(self.workers)]
        for index, request in enumerate(requests):
            lane = request['lane']
            if type(lane) is not int or lane < 0:
                raise ValueError('Nonnegative integer lane required')
            groups[lane % self.workers].append((index, request))
        deadline = time.monotonic()+self.timeout_s
        result = [None]*len(requests)
        try:
            for sock, group in zip(self.sockets, groups):
                _send(sock, group, deadline)
            for sock, group in zip(self.sockets, groups):
                response = _receive(sock, deadline)
                if not response['ok']:
                    raise RuntimeError('MPC worker rejected batch: '+response['error'])
                if [x[0] for x in response['results']] != [x[0] for x in group]:
                    raise ValueError('MPC worker response order changed')
                for index, value in response['results']:
                    result[index] = value
            return result
        except BaseException:
            self.close(); raise

    def close(self):
        if self.closed:
            return
        self.closed = True
        for sock in self.sockets:
            sock.close()
        for process in self.processes:
            try:
                process.wait(timeout=2.)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=2.)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait(timeout=2.)

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()


def worker(fd):
    sock = socket.socket(fileno=fd)
    try:
        configuration = _receive(sock, time.monotonic()+60.)
        controller = build_controller(configuration)
        _send(sock, dict(ready=True,pid=os.getpid(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')),
              time.monotonic()+60.)
        while True:
            group = _receive(sock,time.monotonic()+600.)
            try:
                results = [(index,execute(controller,request)) for index,request in group]
                response = dict(ok=True,results=results)
            except Exception as error:
                response = dict(ok=False,error=repr(error))
            _send(sock,response,time.monotonic()+60.)
    except (EOFError, BrokenPipeError, ConnectionResetError):
        pass
    finally:
        sock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker-fd',type=int,required=True)
    worker(parser.parse_args().worker_fd)
