"""Continue a frictional-model recovery candidate to the real episode boundary.

Each block uses the same implicit PD and Coulomb contact prediction as the
prefix. All intermediate geometry and hardware constraints and terminal
support tests remain required. This is nominal planning, not native feedback.
"""
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time
import numpy as np
import pinocchio as pin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.search_frictional_recovery_cem_v1 as core
from scripts.search_early_recovery_knots_v1 import expand_corrections
from scripts.predict_frictional_spheres_v1 import predict_frictional_spheres
from scripts.audit_ramped_gravity_continuation_v1 import future_equilibrium_reference, rate_admitted_seed
from scripts.extend_recovery_to_episode_v1 import kinematic_window
from scripts.frictional_static_reference_v1 import sphere_static_reference


def set_state(template, model, q, v, contacts, previous):
    state = copy.deepcopy(template)
    state.q = np.asarray(q, float).copy()
    state.v = np.asarray(v, float).copy()
    state.contacts = np.asarray(contacts, bool).copy()
    state.joint_position = state.q[model.q_indices].copy()
    state.joint_velocity = state.v[model.v_indices].copy()
    state.root_position_world = state.q[:3].copy()
    state.root_rotation_world = pin.Quaternion(state.q[3:7]).matrix()
    state.previous_command_target = np.asarray(previous, float).copy()
    return state


def evaluate_task(task):
    c, template, _, _, spheres, geometry = core.CONTEXT
    state = set_state(template, c.qp.model, task['q'], task['v'], task['contacts'], task['previous'])
    seed = np.asarray(task['seed'], float)
    if seed.shape != (40, 18):
        raise ValueError('Require a complete 0.4 second predictive horizon')
    core.CONTEXT = (c, state, state.previous_command_target, seed, spheres, geometry)
    return core.evaluate(expand_corrections(task['knots']), task.get('save', False))


def replay(c, initial, previous, commands, spheres, geometry):
    state = copy.deepcopy(initial)
    states = []
    for command in commands:
        if np.max(abs(command[12:]-previous[12:])) > .01*c.qp.interface.arm_rate+1e-6:
            raise ValueError('Actual command history violates arm rate')
        item = predict_frictional_spheres(c.qp.model, c.qp.interface, state, command, spheres, geometry=geometry)
        d = item['diagnostics']
        if d['nonfoot_ground_violation_m'] > 1e-6 or any(d[k] > 3e-4 for k in
                ['joint_position_violation_rad', 'joint_velocity_violation_rad_s', 'motor_torque_violation_nm']):
            raise ValueError('Independent replay violates original constraints')
        state = item['state']
        state.previous_command_target = command.copy()
        states.append(copy.deepcopy(state))
        previous = command
    return states


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['bundle', 'sphere-audit', 'prefix-result', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--population', type=int, default=48)
    p.add_argument('--iterations', type=int, default=8)
    p.add_argument('--seed', type=int, default=882098)
    a = p.parse_args()
    if a.output.exists() or not 16 <= a.population <= 128 or not 1 <= a.iterations <= 12:
        raise ValueError('Require bounded new search')
    sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
    prefix = json.loads(a.prefix_result.read_text())
    if not prefix['final']['accepted']:
        raise ValueError('No verified model prefix to continue')
    hashes = dict(prefix['source_sha256'])
    for f, digest in hashes.items():
        if sha(f) != digest:
            raise ValueError('Changed input '+f)
    if str((a.bundle/'manifest.json').resolve()) not in hashes or str(a.sphere_audit.resolve()) not in hashes:
        raise ValueError('Unbound bundle or sphere audit')
    for f in [Path(__file__), a.prefix_result, ROOT/'scripts/audit_ramped_gravity_continuation_v1.py',
              ROOT/'scripts/audit_terminal_gravity_balance_v1.py', ROOT/'configs/evaluation_protocol_v2.json']:
        hashes[str(f.resolve())] = sha(f)
    m = json.loads((a.bundle/'manifest.json').read_text())
    if sha(a.bundle/m['bundle_name']) != m['bundle_sha256']:
        raise ValueError('Changed original bundle')
    row = next(r for r in m['rows'] if r['case'] == prefix['case'])
    core.init_worker(str(a.bundle), str(a.sphere_audit), prefix['case'])
    c, initial, previous, _, spheres, geometry = core.CONTEXT
    for name, module in list(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if name.startswith('scripts.') and path:
            hashes[str(Path(path).resolve())] = sha(path)
    hashes[str(c.qp.model.path.resolve())] = sha(c.qp.model.path)
    initial.previous_command_target = previous.copy()
    commands = np.asarray(prefix['final']['commands'], float)
    if commands.shape != (40, 18):
        raise ValueError('Require complete prefix')
    states = [copy.deepcopy(initial)] + replay(c, initial, previous, commands, spheres, geometry)
    for state, expected in zip(states[1:], prefix['final']['states']):
        if max(np.max(abs(state.q-expected['q'])), np.max(abs(state.v-expected['v']))) > 1e-10:
            raise ValueError('Prefix replay differs')
    with np.load(a.bundle/m['bundle_name']) as z:
        old_commands = z[f"case_{prefix['case']:03d}_commands_rad"].copy()
    needed = 800-row['query_tick']
    if not 40 <= needed <= 800:
        raise ValueError('Invalid original episode boundary')
    rng = np.random.default_rng(a.seed)
    blocks = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('spawn'),
            initializer=core.init_worker, initargs=(str(a.bundle), str(a.sphere_audit), prefix['case'])) as pool:
        while len(commands) < needed:
            current = states[-1]
            previous = commands[-1]
            current.previous_command_target = previous.copy()
            seeds = [('hold_last_command', np.repeat(previous[None], 40, axis=0))]
            gravity = future_equilibrium_reference(c, current)
            if gravity['solved']:
                seeds.append(('current_equilibrium', rate_admitted_seed(c, current, gravity['target'], 40)))
            surface_gravity = sphere_static_reference(c, current, spheres)
            if surface_gravity['solved']:
                seeds.append(('same_sphere_surface_equilibrium', rate_admitted_seed(c, current, surface_gravity['target'], 40)))
            tail = old_commands[len(commands):len(commands)+40]
            if len(tail):
                tail = np.vstack([tail, np.repeat(tail[-1][None], 40-len(tail), axis=0)])
                seeds.append(('old_nominal_seed_only', tail))
            base = dict(q=current.q.tolist(), v=current.v.tolist(), contacts=current.contacts.tolist(),
                previous=previous.tolist(), knots=np.zeros((6, 18)).tolist())
            values = list(pool.map(evaluate_task, [dict(base, seed=x.tolist()) for _, x in seeds]))
            winner = min(range(len(values)), key=lambda i: core.key(values[i]))
            label, seed = seeds[winner]
            task = dict(base, seed=seed.tolist())
            best = values[winner]
            best_knots = np.zeros((6, 18))
            mean = best_knots.copy()
            std = np.tile(np.r_[np.full(12, .02), np.full(6, .01)], (6, 1))
            history = []
            for iteration in range(a.iterations if not best['accepted'] else 0):
                candidates = mean + rng.normal(size=(a.population, 6, 18))*std
                candidates[0] = best_knots
                results = list(pool.map(evaluate_task, [dict(task, knots=x.tolist()) for x in candidates]))
                order = sorted(range(len(results)), key=lambda i: core.key(results[i]))
                elite = candidates[order[:max(6, a.population//6)]]
                mean = .2*mean + .8*elite.mean(0)
                std = np.maximum(.0003, .3*std + .7*elite.std(0))
                if core.key(results[order[0]]) < core.key(best):
                    best = results[order[0]]
                    best_knots = candidates[order[0]].copy()
                record = dict(iteration=iteration, best=best, accepted=sum(r['accepted'] for r in results))
                history.append(record)
                print(json.dumps(dict(start_tick=row['query_tick']+len(commands), **record)), flush=True)
                if best['accepted']:
                    break
            checked = pool.submit(evaluate_task, dict(task, knots=best_knots.tolist(), save=True)).result()
            if core.key(checked) != core.key(best):
                raise ValueError('Independent block replay differs')
            block = dict(start_tick=row['query_tick']+len(commands), seed=label,
                accepted=checked['accepted'], seeds=[dict(name=n, result=v) for (n, _), v in zip(seeds, values)],
                history=history, final={k:v for k,v in checked.items() if k not in ['states', 'commands']})
            blocks.append(block)
            a.output.with_suffix('.partial.json').write_text(json.dumps(blocks, indent=2, allow_nan=False)+'\n')
            print(json.dumps(dict(start_tick=block['start_tick'], accepted=block['accepted'], terminal=checked.get('terminal'))), flush=True)
            if not checked['accepted']:
                break
            chosen = np.asarray(checked['commands'])
            # Verify the whole horizon even for the final truncated export.
            actual = replay(c, current, previous, chosen, spheres, geometry)
            take = min(40, needed-len(commands))
            states.extend(actual[:take])
            commands = np.concatenate([commands, chosen[:take]])
    for f, digest in hashes.items():
        if sha(f) != digest:
            raise ValueError('Input changed during continuation '+f)
    protocol = json.loads((ROOT/'configs/evaluation_protocol_v2.json').read_text())['recovery']
    arrays = a.output.with_suffix('.npz')
    np.savez_compressed(arrays, commands_rad=commands,
        predicted_q=np.stack([s.q for s in states]), predicted_v=np.stack([s.v for s in states]),
        predicted_contacts=np.stack([s.contacts for s in states]))
    result = dict(scope=__doc__, case=prefix['case'], query_tick=row['query_tick'],
        model_plan_reaches_original_end=len(commands)==needed, exported_steps=len(commands),
        native_execution=False, valid_expert_labels=False, blocks=blocks,
        model_kinematic_window=kinematic_window(c.qp.model, states[1:], protocol, .01),
        wall_seconds=time.perf_counter()-started, source_sha256=hashes,
        arrays=str(arrays), arrays_sha256=sha(arrays),
        limitation='Predicted contact and kinematics do not establish native force validity or complete success')
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
