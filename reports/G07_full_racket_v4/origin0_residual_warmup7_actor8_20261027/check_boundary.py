"""Inspect actual warmup7/actor8 checkpoints and the complete first TRAIN128 cohort."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
REPORTS = ROOT / 'reports/G07_full_racket_v4'
TRAIN = REPORTS / 'origin0_residual_fresh_n128_updates15_01'
OWNER = REPORTS / 'origin0_residual_after_gain_body_queue01'
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from quadsmash.evaluation_v4 import independent_episode


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')


def main():
    assert Path.cwd() == ROOT and not (HERE / 'RESULT.json').exists()
    held = read(OWNER / 'ADMISSION_DECISION.json')['source_sha256']
    assert len(held) == 519 and all(sha(Path(p)) == s for p, s in held.items())
    initial = torch.load(TRAIN / 'fresh_initial_model_state.pt', map_location='cpu', weights_only=False)
    start = initial['model_state_dict']
    assert initial['optimizer_entries'] == initial['actual_control_env_steps'] == 0
    keys = [name for name in start if name == 'log_std' or name.startswith('actor.')]
    checkpoint_checks = []
    for iteration in [7, 8]:
        path = TRAIN / f'ppo/model_{iteration - 1}.pt'
        native = torch.load(path, map_location='cpu', weights_only=False)
        model, optimizer = native['model_state_dict'], native['optimizer_state_dict']
        assert native['iter'] == iteration and not native['native']['canonical_restart']
        assert native['native']['total_timesteps'] == iteration * 16384
        assert all(torch.isfinite(v).all() for v in model.values())
        assert int(model['actor_obs_normalizer.count']) == int(model['critic_obs_normalizer.count']) == iteration * 16384
        ids = [n for group in optimizer['param_groups'] for n in group['params']]
        names = [n for n in model if n == 'log_std' or n.startswith(('actor.', 'critic.'))]
        assert len(ids) == len(names) == 17
        steps = {name: int(optimizer['state'][i]['step']) for i, name in zip(ids, names) if i in optimizer['state']}
        assert len(steps) == (8 if iteration == 7 else 17)
        assert all(v == (iteration * 20 if name.startswith('critic.') else 20) for name, v in steps.items())
        differences = {name: float((model[name] - start[name]).abs().max()) for name in keys}
        unchanged = all(value == 0 for value in differences.values())
        assert unchanged == (iteration == 7)
        if iteration == 8:
            assert any(differences[n] > 0 for n in keys if n.startswith('actor.'))
        assert int(native['native']['environment']['tensors']['episode_index'].min()) >= 1
        checkpoint_checks.append(dict(iteration=iteration, checkpoint_sha256=sha(path),
            actor_and_log_std_unchanged=unchanged, actor_parameter_max_differences=differences,
            actual_Adam_steps=steps, actual_RMS_count=iteration * 16384,
            all128_initial_episodes_completed=True, canonical_restart=False))
    raw = read(TRAIN / 'episodes.partial.json')
    first = {row['env_id']: row for row in raw if row['episode_index'] == 0}
    assert set(first) == set(range(128))
    rows = [first[i] for i in range(128)]
    for lane, row in enumerate(rows):
        assert row['env_id'] == row['source_training_index'] == row['jump']['env_id'] == lane
        assert row['ground_init'] and row['case_id'] == f'TRAIN{lane}_LANE{lane}_EP0'
    previous = read(REPORTS / 'origin0_residual_early_failures_20261027/COHORT_OBSERVATION.json')
    for old in previous['first_cohort_status_projection']:
        digest = hashlib.sha256(json.dumps(first[old['env_id']], sort_keys=True).encode()).hexdigest()
        assert digest == old['canonical_full_raw_row_sha256']
    protocol_path = ROOT / 'configs/g07_full_racket_eval_v4_r3_origin0.json'
    protocol = read(protocol_path)
    protocol['_root'] = str(ROOT)
    checked = [independent_episode(row, protocol) for row in rows]
    summary = dict(episodes=128, before_launch=sum(not r['launched'] for r in rows),
        airborne_contacts=sum(c['airborne_contact'] for c in checked),
        conditional_recovery_count=sum(c['conditional_recovery'] for c in checked),
        complete=sum(c['complete'] for c in checked), numerical_failures=sum(r['numerical_failures'] for r in rows))
    assert summary['numerical_failures'] == 0
    anomalous = []
    for row, c in zip(rows, checked):
        j = row['jump']
        if c['airborne_contact'] and j['stable_recovery'] and not j['body_landing'] and not j['qualified_jump_landed']:
            anomalous.append(dict(TRAIN_index=row['env_id'], original_episode_index=row['episode_index'],
                actual_airborne_contact=True, stable_recovery=j['stable_recovery'], recovery_s=j['recovery_s'],
                qualified_jump_landed=j['qualified_jump_landed'], qualified_flight_seen=j['qualified_flight_seen'],
                body_landing=j['body_landing'], maximum_body_ground_force_n=j['maximum_body_ground_force_n'],
                last_takeoff_time_s=j['takeoff_time_s'], last_landing_time_s=j['landing_time_s'],
                last_reported_flight_interval_s=j['landing_time_s'] - j['takeoff_time_s'],
                outgoing_speed_mps=float(np.linalg.norm(row['outgoing_velocity_mps'])),
                landed_target=row['landed_target'], original_conditional_recovery=c['conditional_recovery'],
                complete=c['complete'], actual_full100Hz_episode_trace_saved=(TRAIN / f'trajectory_traces/lane{row["env_id"]:04d}_episode000000.npz').exists(),
                scope='Flag relationship needing a recorded event timeline; no recovery score correction or causal proof.'))
    save(HERE / 'FIRST128_EPISODES.json', rows)
    save(HERE / 'PRODUCTION_UPDATE07_CHECK.json', read(OWNER / 'ACTUAL_UPDATE_07.json'))
    paths = [Path(__file__), TRAIN / 'fresh_initial_model_state.pt', TRAIN / 'ppo/model_6.pt',
             TRAIN / 'ppo/model_7.pt', TRAIN / 'INPUTS.json', protocol_path,
             OWNER / 'ACTUAL_UPDATE_07.json', HERE / 'FIRST128_EPISODES.json']
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), checkpoint_checks=checkpoint_checks,
        first128_original_stochastic_training_summary=summary, previous127_raw_rows_unchanged=True,
        recovery_flag_observations=anomalous, source_sha256={str(p): sha(p) for p in paths},
        all519_current_inputs_unchanged=True,
        scope='Actual critic-warmup and first actor update; training episodes and diagnostic checkpoints, no frozen-policy performance or canonical resume claim.',
        scoring_or_training_sources_changed=False, new_native_execution=False,
        formal_cases_read=False, formal_acceptance=False)
    assert all(sha(Path(p)) == s for p, s in held.items())
    save(HERE / 'RESULT.json', result)
    print(json.dumps(dict(first128=summary, actor_changed_at_update8=True,
        recovery_flag_observation_count=len(anomalous), original_scores_retained=True)), flush=True)


if __name__ == '__main__':
    main()
