"""CPU candidate workers isolated from the Isaac application entrypoint.

Private inherited sockets carry trusted local state/command arrays. Workers
start this guarded script with CUDA disabled, never fork or spawn Isaac.
"""
import argparse
import hashlib
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.parallel_landing_solver_v1 import ParallelLandingSolverV1, _send, _receive, build_controller


class FrictionalCandidatePoolV1(ParallelLandingSolverV1):
    def __init__(self, configuration, spheres, *, workers=4, timeout_s=180.):
        if type(workers) is not int or not 1 <= workers <= 4 or not 0 < timeout_s <= 600:
            raise ValueError('Require bounded CPU worker pool')
        self.workers = workers
        self.timeout_s = timeout_s
        self.processes = []
        self.sockets = []
        self.closed = False
        path = Path(__file__).resolve()
        geometry = ROOT/'configs/go2_piper_authored_ground_geometry_v1.json'
        payload = dict(configuration=configuration, spheres=spheres,
            worker_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            geometry_sha256=hashlib.sha256(geometry.read_bytes()).hexdigest())
        try:
            for _ in range(workers):
                parent, child = socket.socketpair()
                environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
                try:
                    process = subprocess.Popen([sys.executable, str(path), '--worker-fd', str(child.fileno())],
                        cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, pass_fds=(child.fileno(),))
                except BaseException:
                    parent.close()
                    raise
                finally:
                    child.close()
                self.processes.append(process)
                self.sockets.append(parent)
                deadline = time.monotonic()+timeout_s
                _send(parent, payload, deadline)
                ready = _receive(parent, deadline)
                if ready != dict(ready=True, pid=process.pid, cuda_visible_devices='', worker_sha256=payload['worker_sha256']):
                    raise ValueError('Unexpected candidate worker identity')
        except BaseException:
            self.close()
            raise


def worker(fd):
    import copy
    import numpy as np
    import scripts.search_frictional_recovery_cem_v1 as core
    from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance
    from scripts.extend_frictional_recovery_v1 import set_state
    sock = socket.socket(fileno=fd)
    try:
        payload = _receive(sock, time.monotonic()+60.)
        digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        geometry_path = ROOT/'configs/go2_piper_authored_ground_geometry_v1.json'
        if digest != payload['worker_sha256'] or hashlib.sha256(geometry_path.read_bytes()).hexdigest() != payload['geometry_sha256']:
            raise ValueError('Worker implementation or geometry changed')
        c = build_controller(payload['configuration'])
        geometry = AuthoredGroundClearance(geometry_path, c.qp.model, margin_m=0.)
        spheres = payload['spheres']
        if [s['body'] for s in spheres] != c.qp.model.feet:
            raise ValueError('Foot sphere order differs')
        _send(sock, dict(ready=True, pid=os.getpid(), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), worker_sha256=digest), time.monotonic()+60.)
        while True:
            group = _receive(sock, time.monotonic()+600.)
            try:
                results = []
                for index, request in group:
                    state = request['state']
                    seed = np.asarray(request['seed'], float)
                    correction = np.asarray(request['correction'], float)
                    horizon = len(seed)
                    if horizon not in (40,80) or seed.shape != (horizon,18) or correction.shape != seed.shape:
                        raise ValueError('Require full declared recovery horizon')
                    if not np.isfinite(seed).all() or not np.isfinite(correction).all():
                        raise ValueError('Nonfinite recovery command')
                    actual = set_state(state,c.qp.model,state.q,state.v,state.contacts,state.previous_command_target)
                    if not np.isfinite(actual.q).all() or not np.isfinite(actual.v).all():
                        raise ValueError('Nonfinite observed state')
                    core.CONTEXT = (c,actual,actual.previous_command_target,seed,spheres,geometry)
                    answer = core.evaluate(correction, bool(request.get('save_states',False)))
                    answer['input_q'] = actual.q.copy()
                    answer['input_v'] = actual.v.copy()
                    answer['input_previous_command'] = actual.previous_command_target.copy()
                    results.append((index,answer))
                response = dict(ok=True, results=results)
            except Exception as error:
                response = dict(ok=False, error=repr(error))
            _send(sock,response,time.monotonic()+600.)
    except (EOFError,ConnectionError):
        pass
    finally:
        sock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker-fd',type=int,required=True)
    worker(parser.parse_args().worker_fd)
