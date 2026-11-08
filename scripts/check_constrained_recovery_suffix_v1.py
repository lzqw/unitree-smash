"""Execute full-resolution suffix equivalence and changed-state checks.

Uses completed model evidence as the regression oracle; never resets native
physics. A successful check validates clock/state semantics, not recovery
robustness, MPC optimization, or a teacher demonstration.
"""
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np
import pinocchio as pin

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.prepare_native_constrained_tracking_v1 import validated_candidate
from scripts.search_preserved_early_support_v1 import initialize
from scripts.search_landing_support_tail_v3 import _WORKER
from scripts.hip_mount_path_constraint_v1 import HipMountPathConstraintV1
from scripts.terminal_attitude_feedback_v1 import TerminalAttitudeFeedbackV1
from scripts.feedback_tail_rollout_v1 import terminal_measures, kinematically_stable
from scripts.constrained_recovery_suffix_v1 import evaluate_suffix
from scripts.probe_constrained_support_domain_v1 import identity
from scripts.audit_joint_sequence_proposal_v1 import plain


def source_context(path):
    read = lambda p: json.loads(Path(p).read_text())
    search = read(path); knots, expected = validated_candidate(search)
    previous = read(search['configuration']['source'])
    old = read(previous['configuration']['search']); source = read(old['configuration']['source'])
    audit = read(previous['configuration']['collision_audit'])
    paths = [Path(p) for p in source['source_sha256'] if Path(p).name == 'refine_support_tail_fine160_v1.json']
    if len(paths) != 1: raise ValueError('Unique fixed feedback reference required')
    fine = read(paths[0]); tail = read(fine['configuration']['source'])
    parent = read(tail['configuration']['source_search']); cfg = parent['configuration']
    query = read(cfg['query']); manifest = read(Path(cfg['bundle'])/'manifest.json')
    states = fine['independent_replay']['trajectory']
    rq = np.asarray([query['initial_q']]+[r['q'] for r in states[:-1]])
    rv = np.asarray([query['initial_v']]+[r['v'] for r in states[:-1]])
    return search, knots, expected, audit, (manifest['configuration'], query,
        fine['fixed_early_knots_rad'], cfg['sphere_audit'], rq, rv, source['configuration'])


def state_from_row(row):
    w = _WORKER; model = w['controller'].qp.model
    state = copy.deepcopy(w['initial'])
    state.q = np.asarray(row['q']).copy(); state.v = np.asarray(row['v']).copy()
    state.contacts = np.asarray(row['contacts'], bool).copy()
    state.root_rotation_world = pin.Quaternion(state.q[3:7]).matrix()
    state.root_position_world = state.q[:3].copy()
    state.joint_position = state.q[model.q_indices].copy()
    state.joint_velocity = state.v[model.v_indices].copy()
    state.previous_command_target = np.asarray(row['target']).copy()
    return state


def check_suffix(tick, changed, knots, trajectory, audit, deadline):
    w = _WORKER; index = tick-280; state = state_from_row(trajectory[index-1])
    controller = w['controller']; preceding = 0
    for row in reversed(trajectory[:index]):
        sample = state_from_row(row)
        if sample.q[2] < .12 or not kinematically_stable(terminal_measures(controller, sample), w['protocol']):
            break
        preceding += 1
    if changed:
        # Diagnostic perturbation of the prediction input only, never native
        # teleportation. Keep the original reference and command history.
        state.v[3] += .01
        if not kinematically_stable(terminal_measures(controller, state), w['protocol']): preceding = 0
    policy = TerminalAttitudeFeedbackV1(w['factory'](controller.qp.model,
        controller.qp.interface, w['initial'], knots), proportional=.3,
        damping_s=.04, maximum_joint_delta_rad=.08)
    geometry = HipMountPathConstraintV1(w['geometry'], audit, margin_rad=.02)
    q_before = state.q.copy(); v_before = state.v.copy(); command_before = state.previous_command_target.copy()
    result = evaluate_suffix(controller, state, policy, w['spheres'], geometry, w['protocol'],
        control_tick=tick, substeps=160, deadline=deadline,
        preceding_stable_ticks=preceding, save=True)
    if not all(np.array_equal(x, y) for x, y in [(state.q,q_before),(state.v,v_before),
        (state.previous_command_target,command_before),(result['initial_q'],q_before),
        (result['initial_v'],v_before),(result['initial_previous_command'],command_before)]):
        raise ValueError('Suffix mutated or replaced the supplied state/history')
    expected = trajectory[index:]; actual = result['trajectory']
    exact = len(actual) == len(expected) and all(np.array_equal(a[k], b[k])
        for a,b in zip(actual,expected) for k in ['q','v','target','contacts'])
    if not changed and (not exact or not result['accepted']):
        raise ValueError('Original reference suffix did not reproduce exactly')
    first_changed = None
    for i,(a,b) in enumerate(zip(actual,expected)):
        if any(not np.array_equal(a[k],b[k]) for k in ['q','v','target','contacts']):
            first_changed = tick+i; break
    if changed and first_changed is None:
        raise ValueError('Perturbed prediction state had no verified response')
    return dict(control_tick=tick, changed_state=changed, original_suffix_exact=exact,
        first_changed_control_tick=first_changed, input_and_history_preserved=True, result=result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--wait-pid', type=int); p.add_argument('--wait-start-ticks')
    a = p.parse_args()
    if a.output.exists(): raise ValueError('Fresh output required')
    if bool(a.wait_pid) != bool(a.wait_start_ticks): raise ValueError('Explicit waiting process identity required')
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    own_sha = sha(__file__)
    if a.wait_pid:
        until = time.monotonic()+1800
        print(json.dumps(dict(state='WAITING_FOR_CPU_OWNER', pid=a.wait_pid)), flush=True)
        while True:
            current = identity(a.wait_pid)
            if current is None: break
            if current != a.wait_start_ticks: raise ValueError('Waiting process identity changed')
            if time.monotonic() >= until: raise TimeoutError('CPU wait limit')
            time.sleep(5)
    if sha(__file__) != own_sha: raise ValueError('Waiting program changed')
    search, knots, expected, audit, initargs = source_context(a.source)
    hashes = dict(search['source_sha256'])
    for path in [a.source.resolve(),Path(__file__).resolve(),ROOT/'scripts/constrained_recovery_suffix_v1.py']:
        hashes[str(path)] = sha(path)
    for path,h in hashes.items():
        if sha(path) != h: raise ValueError('Changed source '+path)
    began = time.monotonic(); deadline = began+900
    trajectory = search['best']['result']['trajectory']
    with ProcessPoolExecutor(max_workers=3, mp_context=mp.get_context('spawn'), initializer=initialize,
            initargs=initargs) as pool:
        jobs = [pool.submit(check_suffix,tick,changed,knots,trajectory,audit,deadline)
                for tick,changed in [(350,False),(700,False),(350,True)]]
        rows = [future.result() for future in jobs]
    for path,h in hashes.items():
        if sha(path) != h: raise ValueError('Changed source during check '+path)
    result = dict(scope=__doc__, source_sha256=hashes, rows=rows,
        all_checks_passed=True, wall_seconds=time.monotonic()-began,
        native_execution=False, online_mpc=False, valid_expert_labels=False)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(plain(result),indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(all_checks_passed=True, wall_seconds=result['wall_seconds'],
        rows=[{k:v for k,v in r.items() if k!='result'} for r in rows])),flush=True)


if __name__ == '__main__': main()
