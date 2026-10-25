"""Preserve a complete first-ground training cohort without reporting frozen performance."""
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RUN = ROOT / 'reports/G07_full_racket_v4/residual_ppo_n512_resume24_01'
sys.path.insert(0, str(ROOT))
from quadsmash.evaluation_v4 import independent_episode


def main():
    assert not (HERE / 'REPORT.json').exists(), 'Never overwrite a preserved cohort'
    protocol = json.loads((ROOT / 'configs/g07_full_racket_eval_v4_r3.json').read_text())
    protocol['_root'] = str(ROOT)
    for _ in range(3):
        progress_bytes = (RUN / 'progress.json').read_bytes()
        progress = json.loads(progress_bytes)
        raw = (RUN / 'episodes.partial.json').read_bytes()
        rows = json.loads(raw)
        checks = [independent_episode(row, protocol) for row in rows]
        counts = dict(completed_episodes=len(rows), air=sum(v['airborne_contact'] for v in checks),
            conditional_recovery_count=sum(v['conditional_recovery'] for v in checks),
            complete=sum(v['complete'] for v in checks), numerical_failures=sum(v['numerical_failures'] for v in rows))
        if progress_bytes == (RUN / 'progress.json').read_bytes() and all(progress[k] == v for k, v in counts.items()):
            break
        time.sleep(.25)
    else:
        raise RuntimeError('Progress/episode write in flight; re-poll the same live job')
    assert progress['update'] >= 15
    first = {}
    for row in rows:
        first.setdefault(row['env_id'], row)
    assert set(first) == set(range(512))
    first = [first[i] for i in range(512)]
    assert all(row['episode_index'] == 0 and row['source_training_index'] == i % 128 and row['ground_init'] for i, row in enumerate(first))
    checked = [independent_episode(row, protocol) for row in first]
    attacking = [bool(v['airborne_contact'] and row['released'] and np.isfinite(row['outgoing_velocity_mps']).all()
        and np.linalg.norm(row['outgoing_velocity_mps']) >= 8 and row['outgoing_velocity_mps'][0] > 0
        and row['outgoing_velocity_mps'][2] <= -1) for row, v in zip(first, checked)]
    cohort = dict(episodes=512, distinct_TRAIN_cases=128, lanes_per_case=4,
        air=sum(v['airborne_contact'] for v in checked),
        conditional_recovery=sum(v['conditional_recovery'] for v in checked),
        complete=sum(v['complete'] for v in checked), numerical_failures=sum(v['numerical_failures'] for v in first),
        racket_contacts=sum(v['had_contact'] for v in first), frame_contacts=sum(v['frame_hit'] for v in first),
        frame_and_legal_air_overlap=sum(row['frame_hit'] and v['airborne_contact'] for row, v in zip(first, checked)),
        attack_outgoing=sum(attacking), target=sum(v['landed_target'] for v in first),
        body_landing=sum(v['jump']['body_landing'] for v in first), early_terminated=sum(v['training_early_termination'] for v in first))
    assert cohort['air'] == 70 and cohort['conditional_recovery'] == 33 and cohort['complete'] == 0
    checkpoint = RUN / f'ppo/model_{progress["update"] - 1}.pt'
    model = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert model['iter'] == progress['update'] and model['native']['environment']['num_envs'] == 512
    assert model['native']['total_timesteps'] == 131072 + progress['actual_control_env_steps']
    assert all(torch.isfinite(v).all() for v in model['model_state_dict'].values() if torch.is_tensor(v))
    assert all(torch.isfinite(v).all() for s in model['optimizer_state_dict']['state'].values() for v in s.values() if torch.is_tensor(v))
    payload = (json.dumps(first, separators=(',', ':')) + '\n').encode()
    compressed = gzip.compress(payload, mtime=0)
    (HERE / 'first_ground_episodes.json.gz').write_bytes(compressed)
    assert gzip.decompress((HERE / 'first_ground_episodes.json.gz').read_bytes()) == payload
    (HERE / 'progress.json').write_bytes(progress_bytes)
    result = dict(created_at=datetime.now(timezone.utc).isoformat(),
        scope='First512 actual native ground episodes with changing actor/normalizer states and stochastic exploration. Not one frozen policy or formal A/B.',
        snapshot_update=progress['update'], first_cohort_seen_complete_at_update=15,
        current_job_control_env_steps=progress['actual_control_env_steps'], cumulative_learner_control_env_steps=model['native']['total_timesteps'],
        all_closed_training_counts_at_snapshot=counts, first_ground_cohort_counts=cohort,
        TRAIN_cases_by_air_contacts_in_four_lanes=dict(Counter(sum(checked[i]['airborne_contact'] for i in range(case, 512, 128)) for case in range(128))),
        all_model_optimizer_tensors_finite=True, diagnostic_checkpoint_canonical_restart=False,
        diagnostic_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        raw_first_cohort='first_ground_episodes.json.gz', raw_first_cohort_sha256=hashlib.sha256(compressed).hexdigest(),
        uncompressed_json_sha256=hashlib.sha256(payload).hexdigest(), original_partial_snapshot_sha256=hashlib.sha256(raw).hexdigest(),
        first_cohort_native_indices_preserved=True,
        limitations=['Separate trajectories experience exploration, updated weights/normalizers and native placements. No fixed-policy rate or causal improvement is inferred.',
            'Frame and legal-air counts can overlap within a stroke; they are not disjoint categories.',
            'Later episodes from early-reset lanes are excluded from this complete512-episode first-ground cohort.'],
        formal_cases_read=False, formal_acceptance=False, goal_complete=False)
    (HERE / 'REPORT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    print('compressed_bytes', len(compressed), flush=True)


if __name__ == '__main__':
    main()
