"""Use verified racket/net distance directions to seed full18 recovery search.

Geometry supplies initial command guesses only. Every guess is rolled out with
the unchanged coupled actuator/contact model to the original episode endpoint.
No geometric projection is called dynamically feasible or a native success.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

import numpy as np
import pinocchio as pin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.search_current_native_recovery_v4 import initialize, rollout, knots_for, WORK
from scripts.retry_budgeted_settling_feedback_v1 import sha, write


def directions(controller, candidate):
    trajectory = candidate['fine']['trajectory']
    checks = []
    for offset in [6, 12]:
        row = trajectory[max(0, len(trajectory)-offset)]
        q = np.asarray(row['q'], float)
        witness = controller.geometry.net.evaluate(q, jacobian=True,
            near_distance_m=.2)['Racket']
        if not witness['exact'] or witness['distance_m'] <= .0005 or 'jacobian' not in witness:
            continue
        gradient = np.asarray(witness['jacobian'])[controller.model.v_indices]
        numeric = []
        epsilon = 1e-5
        for index in controller.model.v_indices:
            tangent = np.zeros(controller.model.model.nv); tangent[index] = epsilon
            plus = pin.integrate(controller.model.model, q, tangent)
            minus = pin.integrate(controller.model.model, q, -tangent)
            distances = [controller.geometry.net.evaluate(x, near_distance_m=.2)['Racket'] for x in [plus, minus]]
            if not all(r['exact'] for r in distances):
                raise ValueError('Cannot differentiate a broad-phase distance bound')
            numeric.append((distances[0]['distance_m']-distances[1]['distance_m'])/(2*epsilon))
        error = float(np.max(abs(gradient-np.asarray(numeric))))
        if error > .002 or np.linalg.norm(gradient) < 1e-5:
            raise ValueError('Racket-distance gradient not independently verified')
        checks.append(dict(step=row['step'], gap_m=witness['distance_m'],
            obstacle=witness['obstacle'], joint_gradient_m_per_rad=gradient,
            central_difference_m_per_rad=numeric, maximum_gradient_error=error,
            finite_difference_epsilon_rad=epsilon, tolerance_m_per_rad=.002))
    if not checks: raise ValueError('No usable exact positive-distance witness')
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['previous', 'configuration', 'run', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(); read = lambda p: json.loads(Path(p).read_text())
    if args.output.exists(): raise ValueError('Fresh search output required')
    previous = read(args.previous/'inputs.json'); result = read(args.previous/'result.json')
    if result['state'] != 'COMPLETED_ACTUAL_RECOVERY_SEQUENCE_SEARCH':
        raise ValueError('Complete previous search required')
    pins = dict(previous['source_sha256'])
    for filename, digest in pins.items():
        if sha(filename) != digest: raise ValueError('Changed original evidence '+filename)
    if (pins.get(str(args.configuration.resolve())) != sha(args.configuration) or
            pins.get(str((args.run/'pure_control_commands.npz').resolve())) != sha(args.run/'pure_control_commands.npz') or
            read(args.run/'job.json')['status'] != 'SUCCEEDED'):
        raise ValueError('Preserve original native observations and hardware interface')
    cfg = read(args.configuration); records = previous['states']
    nominal = read(args.run/'native_command_inputs.json')['default_joint_position_rad']
    initialize(cfg, nominal, records); controller = WORK['c']
    span = np.diff(controller.interface.limits).ravel()
    warm, geometry_checks, files = {}, {}, [Path(__file__), args.previous/'inputs.json', args.previous/'result.json']
    for key in records:
        path = args.previous/('state_'+key.replace(':', '_')+'.json'); candidate = read(path)
        if (candidate['initial_actual_state'] != records[key] or candidate['native_success'] or
                candidate['valid_expert_labels'] or candidate['fine']['substeps'] != 160):
            raise ValueError('Wrong source state or expert label')
        geometry_checks[key] = directions(controller, candidate); files.append(path)
        nodes = np.asarray(candidate['fine']['nodes']); knots = knots_for(records[key])
        if nodes.shape != (len(knots), 18): raise ValueError('Wrong full command parameterization')
        warm[key] = [dict(nodes=nodes, source='previous_best')]
        for check in geometry_checks[key]:
            gradient = np.asarray(check['joint_gradient_m_per_rad'])
            for gap in [.03, .08, .15, .30]:
                correction = gap*gradient/(gradient@gradient+.02**2)
                for ramp in [False, True]:
                    weight = np.minimum(knots/.1, 1.)[:, None] if ramp else np.ones((len(knots), 1))
                    proposed = np.clip(nodes+2*weight*correction/span, -1, 1)
                    warm[key].append(dict(nodes=proposed, source='distance_directed_initial_guess',
                        witness_step=check['step'], requested_local_gap_increase_m=gap, ramp=ramp))
    files += [Path(m.__file__) for n, m in list(sys.modules.items()) if n.startswith('scripts.') and getattr(m, '__file__', None)]
    pins.update({str(f.resolve()):sha(f) for f in files})
    args.output.mkdir(parents=True)
    write(args.output/'inputs.json', dict(scope=__doc__, source_sha256=pins, states=records,
        geometry_checks=geometry_checks, initial_guesses=warm, workers=8, max_seconds=2400,
        reserved_fine_seconds=800, coarse_substeps=16, fine_substeps=160,
        original_episode_end_s=8., native_teleport=False, future_native_actions_used=False,
        training=False, physical_limits_changed=False, first_command_free=True,
        joint_dimension=18, gradient_does_not_certify_dynamics=True))
    started = time.monotonic(); deadline = started+2400
    candidates = {key:[] for key in records}
    rng = {key:np.random.default_rng(889000+i) for i, key in enumerate(records)}
    def progress(state, **extra):
        write(args.output/'status.json', dict(state=state, pid=os.getpid(),
            elapsed_s=time.monotonic()-started,
            completed_coarse=sum(len(v) for v in candidates.values()), **extra))
    progress('SEARCHING_NET_GUIDED_FULL_RECOVERY')
    with ProcessPoolExecutor(max_workers=8, mp_context=mp.get_context('spawn'),
            initializer=initialize, initargs=(cfg, nominal, records)) as pool:
        jobs = {pool.submit(rollout,key,row['nodes'],16,deadline-800):key
            for key in records for row in warm[key]}
        for future in as_completed(jobs): candidates[jobs[future]].append(future.result())
        for generation, sigma in enumerate([.25, .16, .10, .06, .03, .015]):
            if time.monotonic() >= deadline-800: break
            jobs = {}
            for key in records:
                elite = sorted(candidates[key], key=lambda row:row['rank'])[:4]
                for j in range(24):
                    count = len(knots_for(records[key]))
                    noise = rng[key].normal(0, .3*sigma, (count, 18))
                    extension = rng[key].normal(0, sigma, (count, 1))+rng[key].normal(0, .4*sigma, (count, 4))
                    noise[:, 8:12] += extension; noise[:, 4:8] -= .5*extension
                    noise[:, :4] += rng[key].normal(0, .5*sigma, (count, 1))
                    noise[:, 12:] += rng[key].normal(0, sigma, (count, 6))
                    noise[1:] = .5*noise[1:]+.5*noise[:-1]
                    nodes = np.clip(np.asarray(elite[j%len(elite)]['nodes'])+2*noise/span, -1, 1)
                    jobs[pool.submit(rollout,key,nodes,16,deadline-800)] = key
            for future in as_completed(jobs): candidates[jobs[future]].append(future.result())
            progress('SEARCHING_NET_GUIDED_FULL_RECOVERY', generation=generation+1)
        fine = {key:[] for key in records}
        jobs = {pool.submit(rollout,key,np.asarray(row['nodes']),160,deadline):(key, rank)
            for key in records for rank, row in enumerate(sorted(candidates[key], key=lambda row:row['rank'])[:4])}
        progress('CHECKING_FINE_NET_GUIDED_RECOVERY')
        for future in as_completed(jobs):
            key, rank = jobs[future]; row = future.result(); fine[key].append(row)
            write(args.output/('fine_'+key.replace(':','_')+'_'+str(rank)+'.json'), row)
    summary = []
    for key in records:
        chosen = min(fine[key], key=lambda row:row['rank'])
        write(args.output/('state_'+key.replace(':','_')+'.json'), dict(state_key=key,
            initial_actual_state=records[key], fine=chosen,
            coarse_candidates=[{k:v for k,v in row.items() if k not in ['trajectory','terminal']} for row in candidates[key]],
            native_success=False, valid_expert_labels=False))
        summary.append(dict(state_key=key, coarse_candidates=len(candidates[key]),
            steps=chosen['steps'], model_path_complete=chosen['model_path_complete'],
            terminal_stable_s=chosen['terminal_stable_s'], failure=chosen['failure'],
            budget_truncated_coarse=sum((row['failure'] or {}).get('reason')=='compute_budget' for row in candidates[key])))
    for filename, digest in pins.items():
        if sha(filename) != digest: raise ValueError('Changed search input '+filename)
    final = dict(state='COMPLETED_NET_GUIDED_RECOVERY_SEARCH', results=summary,
        elapsed_s=time.monotonic()-started, native_success=False, valid_expert_labels=False)
    write(args.output/'result.json', final); progress(final['state'], results=summary)
    print(json.dumps(final))


if __name__ == '__main__': main()
