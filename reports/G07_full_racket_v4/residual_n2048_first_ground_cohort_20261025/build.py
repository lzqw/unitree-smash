"""Preserve the first complete2048-lane ground cohort from the live native run.

This reads the existing job only. It cannot launch, stop or resume GPU work.
Changing-policy training rows are never presented as frozen-policy performance.
"""
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
REPORTS = ROOT / 'reports/G07_full_racket_v4'
RUN = REPORTS / 'residual_ppo_n2048_resume64_01'
sys.path.insert(0, str(ROOT))
from quadsmash.evaluation_v4 import FAULTS, independent_episode


def sha(data):
    return hashlib.sha256(data).hexdigest()


def atomic_bytes(path, data):
    assert not path.exists(), 'Never overwrite preserved evidence: ' + str(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_bytes(data)
    os.replace(temporary, path)


def main():
    assert not (HERE / 'REPORT.json').exists(), 'This cohort has already been preserved'
    protocol_path = ROOT / 'configs/g07_full_racket_eval_v4_r3.json'
    protocol = json.loads(protocol_path.read_bytes())
    protocol['_root'] = str(ROOT)
    for attempt in range(3):
        progress_bytes = (RUN / 'progress.json').read_bytes()
        progress = json.loads(progress_bytes)
        raw = (RUN / 'episodes.partial.json').read_bytes()
        rows = json.loads(raw)
        checked = [independent_episode(row, protocol) for row in rows]
        counts = dict(completed_episodes=len(rows), air=sum(c['airborne_contact'] for c in checked),
            conditional_recovery_count=sum(c['conditional_recovery'] for c in checked),
            complete=sum(c['complete'] for c in checked),
            numerical_failures=sum(row['numerical_failures'] for row in rows))
        if progress_bytes == (RUN / 'progress.json').read_bytes() and all(progress[k] == v for k, v in counts.items()):
            break
        if attempt < 2:
            time.sleep(.25)
    else:
        raise RuntimeError('Progress/rows were changing; re-observe the same job, never restart it')
    first = [row for row in rows if row['episode_index'] == 0]
    if len(first) != 2048:
        raise RuntimeError(f'First cohort still censored: {len(first)}/2048 closed; preserve no incomplete result')
    first.sort(key=lambda row: row['env_id'])
    assert [row['env_id'] for row in first] == list(range(2048))
    assert len({(row['env_id'], row['episode_index']) for row in rows}) == len(rows)
    assert all(row['source_training_index'] == i % 128 and row['ground_init']
        and row['active_rally_launch_count'] == 1 for i, row in enumerate(first))
    checked_first = [independent_episode(row, protocol) for row in first]
    attacking = [bool(c['airborne_contact'] and row['released']
        and np.isfinite(row['outgoing_velocity_mps']).all()
        and np.linalg.norm(row['outgoing_velocity_mps']) >= 8
        and row['outgoing_velocity_mps'][0] > 0 and row['outgoing_velocity_mps'][2] <= -1)
        for row, c in zip(first, checked_first)]
    cohort = dict(episodes=2048, distinct_TRAIN_cases=128, lanes_per_case=16,
        air=sum(c['airborne_contact'] for c in checked_first),
        conditional_recovery=sum(c['conditional_recovery'] for c in checked_first),
        complete=sum(c['complete'] for c in checked_first),
        numerical_failures=sum(row['numerical_failures'] for row in first),
        racket_contacts=sum(row['had_contact'] for row in first),
        frame_contacts=sum(row['frame_hit'] for row in first),
        attack_outgoing=sum(attacking), target=sum(row['landed_target'] for row in first),
        air_attack_and_target=sum(a and row['landed_target'] for a, row in zip(attacking, first)),
        air_attack_and_recovery=sum(a and c['conditional_recovery'] for a, c in zip(attacking, checked_first)),
        body_landing=sum(row['jump']['body_landing'] for row in first),
        early_terminated=sum(row['training_early_termination'] for row in first),
        timeout=sum(not row['training_early_termination'] for row in first),
        fault_counts={name: sum(row[name] for row in first) for name in FAULTS})
    ground_path = RUN / 'initial_ground_states.json'
    ground = json.loads(ground_path.read_bytes())
    assert ground['all_episode_clocks_zero']
    assert ground['source_training_indices'] == list(range(128)) * 16
    roots = np.asarray(ground['root_state_local'])
    assert roots.shape == (2048, 13) and np.isfinite(roots).all()
    assert np.max(abs(roots[:, :3] - np.asarray(protocol['initialization']['root_position_m']))) <= 1e-5
    assert np.max(abs(roots[:, 3:7] - np.array([1., 0., 0., 0.]))) <= 1e-5
    assert np.max(abs(roots[:, 7:])) <= 1e-5
    assert np.max(abs(np.asarray(ground['joint_velocity_rad_s']))) <= 1e-5
    old_ground = json.loads((REPORTS / 'residual_ppo_n512_resume24_01/initial_ground_states.json').read_bytes())
    assert np.max(abs(np.asarray(ground['joint_position_rad']) - np.asarray(old_ground['joint_position_rad'])[0])) <= 1e-5
    checkpoint_path = RUN / f'ppo/model_{progress["update"] - 1}.pt'
    checkpoint_bytes = checkpoint_path.read_bytes()
    model = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    assert model['iter'] == progress['update'] >= 39
    assert model['native']['environment']['num_envs'] == 2048
    assert model['native']['total_timesteps'] == 1703936 + progress['actual_control_env_steps']
    assert all(torch.isfinite(v).all() for v in model['model_state_dict'].values() if torch.is_tensor(v))
    assert all(torch.isfinite(v).all() for state in model['optimizer_state_dict']['state'].values()
               for v in state.values() if torch.is_tensor(v))
    terminal_poses_finite = all(np.isfinite(row['terminal_native_state']['body_link_positions_m']).all()
        and np.isfinite(row['terminal_native_state']['body_link_quaternions_wxyz']).all() for row in first)
    assert terminal_poses_finite
    for name, payload in [('all_closed_training_episodes.json.gz', raw),
                          ('first_ground_episodes.json.gz', (json.dumps(first, separators=(',', ':')) + '\n').encode())]:
        data = gzip.compress(payload, mtime=0)
        atomic_bytes(HERE / name, data)
        assert gzip.decompress((HERE / name).read_bytes()) == payload
    atomic_bytes(HERE / 'progress.json', progress_bytes)
    older = json.loads((REPORTS / 'residual_n512_first_ground_cohort_20261025/REPORT.json').read_bytes())
    result = dict(created_at=datetime.now(timezone.utc).isoformat(),
        scope='Actual first2048 ground training episodes,128 original TRAIN cases each repeated16 times. Changing actor/normalizer states and stochastic exploration; not frozen performance or formal A/B.',
        snapshot_update=progress['update'], current_job_control_env_steps=progress['actual_control_env_steps'],
        cumulative_learner_control_env_steps=model['native']['total_timesteps'],
        all_closed_training_counts_at_snapshot=counts, first_ground_cohort_counts=cohort,
        TRAIN_cases_by_air_contacts_in16_lanes=dict(Counter(sum(checked_first[i]['airborne_contact']
            for i in range(case, 2048, 128)) for case in range(128))),
        training_terminal_reason_counts=dict(Counter(name for row in first
            for name, value in row['training_terminal_reasons'].items() if value)),
        all2048_actual_ground_states_verified=True, all_model_optimizer_tensors_finite=True,
        all_first_episode_terminal_robot_poses_finite=terminal_poses_finite,
        diagnostic_checkpoint_canonical_restart=False, diagnostic_checkpoint_sha256=sha(checkpoint_bytes),
        earlier512_ground_cohort_descriptive_only=older['first_ground_cohort_counts'],
        earlier512_report_sha256=sha((REPORTS / 'residual_n512_first_ground_cohort_20261025/REPORT.json').read_bytes()),
        raw_files={name: dict(bytes=(HERE / name).stat().st_size, sha256=sha((HERE / name).read_bytes()))
            for name in ['all_closed_training_episodes.json.gz', 'first_ground_episodes.json.gz', 'progress.json']},
        sources={str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in [protocol_path,
            ROOT / 'configs/court_net_v4.json', ROOT / 'configs/evaluation_protocol_v2.json',
            ROOT / 'quadsmash/evaluation_v4.py', ground_path, RUN / 'INPUTS.json', RUN / 'source_hashes.json']},
        partial_episode_snapshot_sha256=sha(raw),
        limitations=['Native first episodes include changing actors/normalizers and exploration. No fixed-policy rate, statistical gain or causal batch-size comparison is inferred.',
            'The earlier512 cohort began from iter8; this2048 cohort began from iter32 with another seed and placement. They are descriptive engineering snapshots.',
            'Later episodes from early-reset lanes are excluded from the2048 first-ground cohort but included in the separately scoped all-closed snapshot.',
            'Legal-air, frame and fault flags may overlap; count them using their original definitions.',
            'Diagnostic PPO model files are not canonical environment restarts. Do not resume them directly or start another GPU owner.',
            'Original task thresholds and all400 formal cases remain unchanged and unproven.'],
        formal_cases_read=False, formal_acceptance=False, goal_complete=False)
    atomic_bytes(HERE / 'REPORT.json', (json.dumps(result, indent=2) + '\n').encode())
    print(json.dumps(dict(snapshot_update=progress['update'], first_ground_cohort=cohort,
        all_closed=counts, model_optimizer_tensors_finite=True)), flush=True)


if __name__ == '__main__':
    main()
