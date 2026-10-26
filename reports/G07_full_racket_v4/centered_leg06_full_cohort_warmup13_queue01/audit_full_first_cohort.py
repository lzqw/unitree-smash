"""Capture actual seventh critic-only update and all2048 first ground episodes."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
TRAIN = ROOT/'reports/G07_full_racket_v4/residual_ppo_centered_leg06_full_cohort_warmup7_n2048_probe13_01'
sys.path.insert(0, str(ROOT))
from quadsmash.evaluation_v4 import independent_episode


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def counts(rows, protocol):
    results = [independent_episode(row, protocol) for row in rows]
    return dict(completed_episodes=len(rows), air=sum(r['airborne_contact'] for r in results),
        conditional_recovery_count=sum(r['conditional_recovery'] for r in results),
        complete=sum(r['complete'] for r in results), numerical_failures=sum(row['numerical_failures'] for row in rows))


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert not (HERE/'FIRST_COHORT_CHECK.json').exists()
    before = (TRAIN/'progress.json').read_bytes();progress = json.loads(before)
    assert progress['update'] == 7 and progress['actual_control_env_steps'] == 1835008
    raw = (TRAIN/'episodes.partial.json').read_bytes();rows = json.loads(raw)
    checkpoint = TRAIN/'ppo/model_6.pt'
    native = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert native['iter'] == 7 and native['native']['total_timesteps'] == 1835008
    assert not native['native']['canonical_restart'] and native['native']['environment']['num_envs'] == 2048
    model = native['model_state_dict']
    assert int(model['actor_obs_normalizer.count']) == 12582912+1835008
    assert int(model['critic_obs_normalizer.count']) == 1835008
    seen, first = [0]*2048, {}
    for row in rows:
        lane = row['env_id'];assert 0 <= lane < 2048
        assert row['episode_index'] == seen[lane] and row['ground_init']
        assert row['source_training_index'] == lane % 128
        if row['episode_index'] == 0:first[lane] = row
        seen[lane] += 1
    assert seen == native['native']['environment']['tensors']['episode_index'].tolist()
    assert set(first) == set(range(2048)) and min(seen) >= 1
    protocol = read(ROOT/'configs/g07_full_racket_eval_v4_r3.json');protocol['_root'] = str(ROOT)
    all_counts, first_counts = counts(rows, protocol), counts(list(first.values()), protocol)
    assert all_counts == {key:progress[key] for key in all_counts} and all_counts['numerical_failures'] == 0
    assert (TRAIN/'progress.json').read_bytes() == before
    os.kill(1777471,0)
    (HERE/'ACTUAL_CLOSED_AT_UPDATE7.json').write_bytes(raw)
    (HERE/'ACTUAL_PROGRESS_AT_UPDATE7.json').write_bytes(before)
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), actual_live_native_pid=1777471,
        actual_completed_update=7, actual_current_job_control_env_steps=1835008,
        first_episode_lanes=2048, first_episode_index=0, every_initial_case_repeated16=True,
        all_actual_source_inclusive_episode_chains_exact=True,
        independently_recomputed_first_cohort=first_counts,
        independently_recomputed_all_closed_at_capture=all_counts,
        actor_RMS_count=int(model['actor_obs_normalizer.count']), critic_RMS_count=int(model['critic_obs_normalizer.count']),
        sources={str(path.relative_to(ROOT)):sha(path) for path in [Path(__file__),checkpoint,
            TRAIN/'INPUTS.json',TRAIN/'initial_ground_states.json',HERE/'ACTUAL_CLOSED_AT_UPDATE7.json',
            HERE/'ACTUAL_PROGRESS_AT_UPDATE7.json',ROOT/'configs/g07_full_racket_eval_v4_r3.json']},
        scope='All2048 actual fresh first ground episodes. Changing actor, normalization and exploration; not frozen performance.',
        new_native_execution=False, formal_cases_read=False, formal_acceptance=False, goal_complete=False)
    (HERE/'FIRST_COHORT_CHECK.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key!='sources'}),flush=True)


if __name__ == '__main__':main()
