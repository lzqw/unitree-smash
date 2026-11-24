"""Select bounded local PhysX probes and verify the simultaneously evaluated panel."""
import hashlib
import json
import math
import random

from run_supermimic_approach_ball_only import evaluate_command
from run_supermimic_approach_v2 import dump, run
from quadsmash.rl.supermimic_hybrid_native import QUICK_LANES


def read(path):
    return json.loads(path.read_text())


def probe_spec(source, block, holdout):
    eligible = set(range(1, 128, 2)) - set(holdout)
    priorities = {}
    for path in sorted(source.glob('generation_*/approach_search_results.json')):
        history = read(path)
        for offset, index in enumerate(history['case_indices']):
            outcomes = [variant['rows'][offset] for variant in history['variants']]
            priorities[index] = min(priorities.get(index, 3), min(
                0 if row['frame'] else 1 if not row['string'] else 2 if not row['success'] else 3
                for row in outcomes))
    rng = random.Random(882072 + block)
    candidates = sorted(eligible)
    rng.shuffle(candidates)
    indices = sorted(candidates, key=lambda i: priorities.get(i, 4))[:16]
    offsets = [[0.] * 6]
    for _ in range(2):
        direction = [rng.gauss(0., 1.) for _ in range(6)]
        maximum = max(abs(x) for x in direction)
        offsets.append([.016 * x / maximum for x in direction])
    offsets.append([-x for x in offsets[1]])
    return dict(case_indices=indices, offsets=offsets)


def evaluate_boundary(checkpoint, folder, execution, baseline, exploration, block, holdout):
    spec = folder.with_suffix('.spec.json')
    dump(spec, probe_spec(exploration, block, holdout))
    command = evaluate_command(checkpoint, folder, True, execution, seed=882072, a0=False)
    command.remove('--research-fixed128')
    run([*command, '--approach-hybrid-spec', str(spec)], folder.with_suffix('.log'), 38)
    return verify_boundary(checkpoint, folder, baseline)


def quick_actuator_contract(contract):
    """Compare the actual drive profile of each scored lane across layouts."""
    indices = contract['lane_profile'][:QUICK_LANES]
    profiles = contract['profiles']
    if len(indices) != QUICK_LANES or any(type(i) is not int or not 0 <= i < len(profiles) for i in indices):
        raise ValueError('Hybrid panel lacks valid drive profiles for the scored lanes')
    common = {key: value for key, value in contract.items() if key not in ('lane_profile', 'profiles')}
    return dict(common, lane_profiles=[profiles[i] for i in indices])


def verify_boundary(checkpoint, folder, baseline):
    result = read(folder / 'hybrid_boundary.json')
    metrics = read(folder / 'metrics.json')
    complete = read(folder / 'completed.json')
    with checkpoint.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if (complete['episodes'] != 128 or metrics['evaluated_checkpoint_sha256'] != digest
            or metrics['actor_state_sha256'] != result['actor_sha256'] or metrics['numerical_failures']
            or any(metrics['optimizer_updates'].values()) or not metrics['actor_parameters_unchanged']):
        raise ValueError('Hybrid native evaluation did not complete with the frozen candidate')
    initial = read(folder / 'initial_states.json')[:QUICK_LANES]
    physics = read(folder / 'physics.json')
    physics['actual_origins_m'] = physics['actual_origins_m'][:QUICK_LANES]
    prior_physics = read(baseline['directory'] / 'physics.json')
    prior_physics['actual_origins_m'] = prior_physics['actual_origins_m'][:QUICK_LANES]
    if physics != prior_physics:
        raise ValueError('Hybrid panel changed the native physics contract')
    for name in ('policy_execution.json', 'task_contract.json'):
        if read(folder / name) != read(baseline['directory'] / name):
            raise ValueError(f'Hybrid panel changed {name}')
    name = 'actuator_contract.json'
    if quick_actuator_contract(read(folder / name)) != quick_actuator_contract(read(baseline['directory'] / name)):
        raise ValueError('Hybrid panel changed the drive profiles of the scored lanes')
    if len(initial) != QUICK_LANES or initial != read(baseline['directory'] / 'initial_states.json')[:QUICK_LANES]:
        raise ValueError('Hybrid panel changed matched native initial states')
    rows = result['panel']['rows']
    outcomes = {row['case_id']: row['success'] for row in rows}
    if (len(outcomes) != QUICK_LANES or outcomes.keys() != baseline['outcomes'].keys()
            or result['panel_sha256'] != baseline['panel_sha256']):
        raise ValueError('Hybrid boundary lacks the registered paired cases')
    panel = dict(panel_sha256=result['panel_sha256'], directory=folder, outcomes=outcomes,
                 success=sum(outcomes.values()), qualified=sum(row['string'] for row in rows),
                 frame=sum(row['frame'] for row in rows), low=sum(row['low_speed'] for row in rows),
                 target=sum(row['landed_target'] for row in rows),
                 robot_net=sum(row['robot_net'] for row in rows),
                 gains=sorted(key for key in outcomes if outcomes[key] and not baseline['outcomes'][key]),
                 losses=sorted(key for key in outcomes if not outcomes[key] and baseline['outcomes'][key]),
                 teacher_episodes=len(result['examples']['teacher']),
                 preserve_episodes=len(result['examples']['preserve']), elapsed_s=result['elapsed_s'])
    if not math.isfinite(panel['elapsed_s']):
        raise ValueError('Invalid native timing')
    return panel, folder / 'hybrid_native.pt'
