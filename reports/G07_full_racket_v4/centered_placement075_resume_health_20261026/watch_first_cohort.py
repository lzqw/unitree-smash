"""Read-only CPU monitor for every first ground episode after strict resume.

This observes the existing native job. It never starts/stops a GPU process or
changes its code, configuration, cases, weights, ledger or acceptance gates.
"""
import argparse
import csv
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('observed075_resume', HERE.parent / 'centered_placement075_resume32_queue02/run.py')
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)
core = queue.core
PLAN = HERE / 'FIRST_COHORT_MONITOR_PLAN.json'
STATE = HERE / 'first_cohort_monitor_state.json'
RESULT = HERE / 'ACTUAL_ALL2048_FIRST_RESUMED_COHORT_CHECK.json'
OWNER = 1734658
GUARD = 1734667
NATIVE = 1734680


def write(state, **fields):
    value = dict(state=state, observed_at=datetime.now(timezone.utc).isoformat(),
                 monitor_pid=__import__('os').getpid(), existing_native_pid=NATIVE,
                 new_GPU_job_started=False, frozen_policy_performance=False,
                 formal_acceptance=False, goal_complete=False, **fields)
    core.atomic(STATE, value)
    print(json.dumps(value), flush=True)


def require_live_source():
    assert core.live(OWNER, str(queue.HERE / 'run.py')), 'Original observer exited'
    for pid in [GUARD, NATIVE]:
        assert core.live(pid, str(queue.TRAIN)), 'Original native/guard exited'
    active = [v for v in core.read(ROOT / 'status/GPU_LEDGER.json')['runs'] if v['status'] == 'RUNNING']
    assert len(active) == 1 and Path(active[0]['output']) == queue.TRAIN
    assert active[0]['pid'] == GUARD and active[0]['child_pid'] == NATIVE


def prepare():
    assert not PLAN.exists() and not STATE.exists() and not RESULT.exists()
    require_live_source()
    progress = core.read(queue.TRAIN / 'progress.json')
    source = torch.load(queue.SOURCE / 'checkpoint.pt', map_location='cpu', weights_only=False)
    assert source['iter'] == 8 and source['native']['canonical_restart']
    current = torch.load(queue.TRAIN / f'ppo/model_{progress["update"] - 1}.pt', map_location='cpu', weights_only=False)
    initial = source['native']['environment']['tensors']['episode_index']
    terminal = current['native']['environment']['tensors']['episode_index']
    assert initial.shape == terminal.shape == (2048,) and (initial > 0).all()
    assert not (terminal > initial).all(), 'Register before observing all first resumed episodes'
    original = core.read(queue.PLAN)
    queue.check_sources(original['source_hashes'])
    queue.save_once(PLAN, dict(created_at=datetime.now(timezone.utc).isoformat(),
        registered_at_actual_stage_update=progress['update'], original_observer_pid=OWNER,
        original_guard_pid=GUARD, original_native_pid=NATIVE,
        source_checkpoint_sha256=core.sha(queue.SOURCE / 'checkpoint.pt'),
        original_training_plan_sha256=core.sha(queue.PLAN),
        monitor_sha256=core.sha(Path(__file__)), source_iteration=8, count=2048,
        selection='All lanes, first closed source-inclusive episode ID after canonical8 ground resume. No episode0 assumption or outcome-based filtering.',
        boundary='Earliest persisted completed-update diagnostic with every terminal episode ID greater than its source ID, scanning in update order.',
        scope='Changing-policy exploratory TRAIN cohort, not frozen128 or formal A/B performance.',
        maximum_monitor_wait_seconds=7200, poll_seconds=20,
        no_GPU_control_or_runtime_mutation=True, formal_acceptance=False, goal_complete=False))
    print(json.dumps(dict(prepared=str(PLAN), registered_update=progress['update'], all_first_episodes_closed=False)), flush=True)


def verify_cohort(source, native, checkpoint, plan):
    from quadsmash.evaluation_v4 import independent_episode
    assert core.sha(queue.PLAN) == plan['original_training_plan_sha256']
    assert core.sha(Path(__file__)) == plan['monitor_sha256']
    queue.check_sources(core.read(queue.PLAN)['source_hashes'])
    update = native['iter']
    steps = (update - 8) * 262144
    assert 9 <= update <= 40 and not native['native']['canonical_restart']
    assert native['native']['total_timesteps'] == 2097152 + steps
    assert native['native']['environment']['num_envs'] == 2048
    model, previous = native['model_state_dict'], source['model_state_dict']
    assert all(torch.isfinite(v).all() for v in model.values() if torch.is_tensor(v))
    for name in ['actor_obs_normalizer.count', 'critic_obs_normalizer.count']:
        assert int(model[name]) - int(previous[name]) == steps
    a, b = source['optimizer_state_dict']['state'], native['optimizer_state_dict']['state']
    assert set(a) == set(b) and len(b) == 17
    assert all(float(b[k]['step']) - float(a[k]['step']) == 20 * (update - 8) for k in a)
    assert all(torch.isfinite(v).all() for state in b.values() for v in state.values() if torch.is_tensor(v))
    raw = (queue.TRAIN / 'episodes.partial.json').read_bytes()
    rows = json.loads(raw)
    initial = source['native']['environment']['tensors']['episode_index'].tolist()
    terminal = native['native']['environment']['tensors']['episode_index'].tolist()
    boundary = [v for v in rows if v['episode_index'] < terminal[v['env_id']]]
    chains = queue.verify_episode_chains(source, native, boundary, 2048)
    first = sorted([v for v in boundary if v['episode_index'] == initial[v['env_id']]], key=lambda v: v['env_id'])
    assert len(first) == 2048 and [v['env_id'] for v in first] == list(range(2048))
    assert all(v['ground_init'] and v['source_training_index'] == v['env_id'] % 128 for v in first)
    protocol = core.read(ROOT / 'configs/g07_full_racket_eval_v4_r3.json')
    protocol['_root'] = str(ROOT)
    outcomes = [independent_episode(v, protocol) for v in first]
    attacks = []
    for row, outcome in zip(first, outcomes):
        velocity = np.asarray(row['outgoing_velocity_mps'])
        attacks.append(bool(outcome['airborne_contact'] and row['released'] and np.isfinite(velocity).all()
                            and np.linalg.norm(velocity) >= 8 and velocity[0] > 0 and velocity[2] <= -1))

    def counts(ids):
        return dict(episodes=len(ids), air=sum(outcomes[i]['airborne_contact'] for i in ids),
                    conditional_recovery=sum(outcomes[i]['conditional_recovery'] for i in ids),
                    complete=sum(outcomes[i]['complete'] for i in ids),
                    numerical_failures=sum(first[i]['numerical_failures'] for i in ids),
                    attack_outgoing=sum(attacks[i] for i in ids),
                    racket_contacts=sum(first[i]['had_contact'] for i in ids),
                    crossed_net=sum(first[i]['crossed_net'] for i in ids),
                    landed_target=sum(first[i]['landed_target'] for i in ids),
                    body_landing=sum(first[i]['jump']['body_landing'] for i in ids))

    aggregate = counts(list(range(2048)))
    assert aggregate['numerical_failures'] == 0
    table = [dict(TRAIN=case, **counts(list(range(case, 2048, 128)))) for case in range(128)]
    assert all(v['episodes'] == 16 for v in table)
    snapshot = HERE / 'FIRST_COMPLETE_RESUMED_BOUNDARY_EPISODES.json'
    queue.save_once(snapshot, boundary)
    csv_path = HERE / 'FIRST2048_RESUMED_BY_TRAIN_CASE.csv'
    with csv_path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(table)
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), actual_boundary_iteration=update,
        actual_new_job_control_env_steps=steps, actual_stage_total_env_steps=native['native']['total_timesteps'],
        first_cohort_count=2048, all128_original_cases_repeated16=True,
        actual_source_inclusive_episode_chains=chains,
        independent_first_cohort_counts=aggregate,
        first_cohort_overlapping_terminal_flags={k:sum(v['training_terminal_reasons'][k] for v in first)
                                                for k in first[0]['training_terminal_reasons']},
        actor_normalizer_count=int(model['actor_obs_normalizer.count']),
        critic_normalizer_count=int(model['critic_obs_normalizer.count']),
        all17_Adam_states_advanced=20*(update-8),
        source_checkpoint_sha256=plan['source_checkpoint_sha256'],
        actual_boundary_diagnostic_sha256=core.sha(checkpoint),
        boundary_snapshot_sha256=core.sha(snapshot), boundary_snapshot_bytes=snapshot.stat().st_size,
        per_case_table_sha256=core.sha(csv_path),
        source_partial_bytes_sha256=__import__('hashlib').sha256(raw).hexdigest(),
        limitations='All first resumed ground episodes, across changing actors/exploration. Not a frozen-policy score, formal A/B result or causal comparison to the zero-noise source8 evaluation.',
        frozen_policy_performance=False, formal_cases_read=False, formal_acceptance=False, goal_complete=False)
    queue.save_once(RESULT, result)
    return result


def main():
    assert PLAN.exists() and not STATE.exists() and not RESULT.exists()
    plan = core.read(PLAN)
    assert core.sha(queue.SOURCE / 'checkpoint.pt') == plan['source_checkpoint_sha256']
    source = torch.load(queue.SOURCE / 'checkpoint.pt', map_location='cpu', weights_only=False)
    initial = source['native']['environment']['tensors']['episode_index']
    checked = set()
    started = time.monotonic()
    while True:
        require_live_source()
        assert time.monotonic() - started < plan['maximum_monitor_wait_seconds'], 'CPU monitor wait limit reached; native job untouched'
        progress = core.read(queue.TRAIN / 'progress.json')
        for update in range(9, progress['update'] + 1):
            if update in checked:
                continue
            checkpoint = queue.TRAIN / f'ppo/model_{update - 1}.pt'
            if not checkpoint.exists():
                break
            native = torch.load(checkpoint, map_location='cpu', weights_only=False)
            assert native['iter'] == update
            terminal = native['native']['environment']['tensors']['episode_index']
            assert terminal.shape == initial.shape == (2048,) and (terminal >= initial).all()
            closed = int((terminal > initial).sum())
            if closed == 2048:
                result = verify_cohort(source, native, checkpoint, plan)
                write('VERIFIED_ALL2048_FIRST_RESUMED_COHORT', result=str(RESULT),
                      actual_boundary_iteration=update, independent_counts=result['independent_first_cohort_counts'])
                return
            checked.add(update)
            write('WAITING_FOR_ALL2048_FIRST_RESUMED_EPISODES', actual_stage_update=update,
                  first_episodes_closed=closed, first_episodes_still_open=2048-closed)
        time.sleep(plan['poll_seconds'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.prepare_only:
        prepare()
    else:
        try:
            main()
        except BaseException as error:
            (HERE / 'first_cohort_monitor_error.txt').write_text(traceback.format_exc())
            write('CPU_COHORT_MONITOR_FAILED_NATIVE_UNTOUCHED', error=repr(error))
            raise
