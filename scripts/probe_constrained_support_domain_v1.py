"""Re-evaluate nearby states using the new constrained full-episode controller.

Only the completed source and its independent fine replay admit this experiment.
All probes retain authored hip/mount checks at every prediction microstep.
This is model evidence, not native robustness, a certified region or an expert.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_native_constrained_tracking_v1 import validated_candidate
from scripts.search_preserved_early_support_v1 import initialize
from scripts.search_landing_support_tail_v3 import _WORKER
from scripts.search_early_landing_trajectory_v1 import evaluate
from scripts.perturb_early_support_domain_v1 import perturb_state
from scripts.hip_mount_path_constraint_v1 import HipMountPathConstraintV1
from scripts.audit_joint_sequence_proposal_v1 import plain


def identity(pid):
    try: return Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()[19]
    except FileNotFoundError: return None


def run_probe(spec, knots, audit, deadline):
    w = _WORKER
    state = perturb_state(w['initial'], spec)
    geometry = HipMountPathConstraintV1(w['geometry'], audit, margin_rad=.02)
    result = evaluate(w['controller'], state, knots, w['spheres'], geometry, w['protocol'],
        substeps=160, deadline=deadline, save=True, policy_factory=w['factory'], control_tick=280)
    if geometry.last_violation is not None:
        result['failure'].update(reason='authored_hip_mount_constraint', collision=geometry.last_violation)
    return dict(probe=spec, initial_q=state.q, initial_v=state.v, result=result,
                geometry_checks=geometry.calls)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wait-pid', type=int)
    parser.add_argument('--wait-start-ticks')
    parser.add_argument('--wait-limit-s', type=float, default=1200.)
    parser.add_argument('--max-seconds', type=float, default=1200.)
    parser.add_argument('--workers', type=int, default=4)
    a = parser.parse_args()
    if (a.output.exists() or not 300 <= a.max_seconds <= 1800 or
            not 1 <= a.workers <= 4 or not 0 < a.wait_limit_s <= 1800):
        raise ValueError('Fresh output and bounded computation required')
    read = lambda p: json.loads(Path(p).read_text())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    own_sha = sha(__file__)
    if bool(a.wait_pid) != bool(a.wait_start_ticks):
        raise ValueError('Wait requires explicit process identity')
    if a.wait_pid:
        print(json.dumps(dict(state='WAITING_FOR_SOURCE_PROCESS', pid=a.wait_pid,
                              start_ticks=a.wait_start_ticks)), flush=True)
        until = time.monotonic() + a.wait_limit_s
        while True:
            actual = identity(a.wait_pid)
            if actual is None: break
            if actual != a.wait_start_ticks: raise ValueError('Waiting PID reused')
            if time.monotonic() >= until: raise TimeoutError('Source wait limit reached')
            time.sleep(5)
    if sha(__file__) != own_sha: raise ValueError('Running probe source changed')
    search = read(a.source)
    knots, expected = validated_candidate(search)
    previous = read(search['configuration']['source'])
    old_search = read(previous['configuration']['search'])
    source = read(old_search['configuration']['source'])
    audit = read(previous['configuration']['collision_audit'])
    paths = [Path(p) for p in source['source_sha256'] if Path(p).name == 'refine_support_tail_fine160_v1.json']
    if len(paths) != 1: raise ValueError('Unique fixed-reference source required')
    fine = read(paths[0]); tail = read(fine['configuration']['source'])
    parent = read(tail['configuration']['source_search']); cfg = parent['configuration']
    query = read(cfg['query']); manifest = read(Path(cfg['bundle'])/'manifest.json')
    states = fine['independent_replay']['trajectory']
    rq = np.asarray([query['initial_q']] + [r['q'] for r in states[:-1]])
    rv = np.asarray([query['initial_v']] + [r['v'] for r in states[:-1]])
    initargs = (manifest['configuration'], query, fine['fixed_early_knots_rad'],
                cfg['sphere_audit'], rq, rv, source['configuration'])
    hashes = dict(search['source_sha256'])
    for p in [a.source.resolve(), Path(__file__).resolve(),
              ROOT/'scripts/prepare_native_constrained_tracking_v1.py',
              ROOT/'scripts/perturb_early_support_domain_v1.py']:
        hashes[str(p)] = sha(p)
    for p, digest in hashes.items():
        if sha(p) != digest: raise ValueError('Changed source ' + p)
    probes = [dict(name='nominal', kind='nominal')]
    for kind, size, unit in [('orientation', np.deg2rad(.5), 'rad'),
                             ('linear_velocity', .02, 'm/s'), ('angular_velocity', .02, 'rad/s')]:
        for axis in (0, 1):
            for sign in (-1, 1):
                probes.append(dict(name=f'{kind}_world_{axis}_{sign:+d}', kind=kind,
                    world_axis=axis, amount=sign*size, unit=unit))
    start = time.monotonic(); deadline = start + a.max_seconds; rows = []
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('spawn'),
            initializer=initialize, initargs=initargs) as pool:
        futures = [pool.submit(run_probe, spec, knots, audit, deadline) for spec in probes]
        for future in as_completed(futures):
            row = future.result(); rows.append(row)
            compact = {k: v for k, v in row['result'].items() if k != 'trajectory'}
            print(json.dumps(plain(dict(probe=row['probe'], result=compact))), flush=True)
            # Preserve completed evidence, including the controller trajectory,
            # rather than only keeping counts while other probes run.
            temporary = a.output.with_suffix('.tmp')
            temporary.write_text(json.dumps(plain(rows), allow_nan=False)+'\n')
            temporary.replace(a.output.with_suffix('.partial.json'))
    nominal = next(r['result'] for r in rows if r['probe']['kind'] == 'nominal')
    if not nominal['accepted'] or len(nominal['trajectory']) != 520:
        raise ValueError('Accepted nominal controller did not reproduce')
    differences = {}
    for field in expected:
        actual = np.asarray([s[field] for s in nominal['trajectory']])
        if not np.array_equal(actual, expected[field]):
            raise ValueError('Nominal replay differs: ' + field)
        differences[field] = 0.
    for p, digest in hashes.items():
        if sha(p) != digest: raise ValueError('Changed source during probes ' + p)
    output = dict(scope=__doc__, source_sha256=hashes, configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        rows=rows, probe_count=len(rows), accepted_model_probes=sum(r['result']['accepted'] for r in rows),
        compute_expired_probes=sum((r['result']['failure'] or {}).get('reason')=='declared_compute_budget' for r in rows),
        nominal_exactly_reproduced=True, nominal_difference=differences, wall_seconds=time.monotonic()-start,
        substeps=160, hip_mount_margin_rad=.02, native_execution=False, valid_expert_labels=False,
        limitations=['Finite axis probes exclude combined, joint, actuator and ball perturbations.',
            'Only audited self-collision pairs are constrained; full collision dynamics are not modeled.',
            'Old unconstrained probe counts are not inherited as new evidence.'])
    a.output.write_text(json.dumps(plain(output), indent=2, allow_nan=False)+'\n')


if __name__ == '__main__': main()
