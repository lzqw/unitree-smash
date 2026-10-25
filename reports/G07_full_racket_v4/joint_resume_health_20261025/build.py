"""Snapshot an actually completed update of the still-live joint40 learner."""
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RUN = ROOT/'reports/G07_full_racket_v4/residual_ppo_joint_placement_actor32_n2048_resume32_01'
SOURCE = ROOT/'reports/G07_full_racket_v4/residual_ppo_joint_placement_actor32_n2048_probe8_01/checkpoint.pt'
sys.path.insert(0, str(ROOT))
from quadsmash.evaluation_v4 import independent_episode


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    assert not (HERE/'REPORT.json').exists()
    for _ in range(5):
        progress_raw = (RUN/'progress.json').read_bytes()
        progress = json.loads(progress_raw)
        update = progress['update']
        checkpoint_path = RUN/'ppo'/f'model_{update-1}.pt'
        checkpoint_raw = checkpoint_path.read_bytes()
        episodes_raw = (RUN/'episodes.partial.json').read_bytes()
        if progress_raw == (RUN/'progress.json').read_bytes():
            break
    else:
        raise RuntimeError('Could not capture a stable observed update; do not restart the native job')
    latest = torch.load(io.BytesIO(checkpoint_raw), map_location='cpu', weights_only=False)
    original = torch.load(SOURCE, map_location='cpu', weights_only=False)
    assert original['iter'] == 8 and latest['iter'] == update
    steps = (update-8)*2048*128
    assert latest['native']['total_timesteps'] == original['native']['total_timesteps']+steps
    assert steps == progress['actual_control_env_steps']
    assert all(torch.isfinite(t).all() for t in latest['model_state_dict'].values() if torch.is_tensor(t))
    current_optimizer = latest['optimizer_state_dict']['state']
    source_optimizer = original['optimizer_state_dict']['state']
    assert current_optimizer.keys() == source_optimizer.keys() and len(current_optimizer) == 17
    increments = []
    for key, state in current_optimizer.items():
        assert all(torch.isfinite(t).all() for t in state.values() if torch.is_tensor(t))
        increments.append(float(state['step']-source_optimizer[key]['step']))
    assert set(increments) == {(update-8)*20.}
    normalizer_increments = {}
    for key in ['actor_obs_normalizer.count', 'critic_obs_normalizer.count']:
        value = int(latest['model_state_dict'][key]-original['model_state_dict'][key])
        assert value == steps
        normalizer_increments[key] = value
    parameter_change = max(float((value-original['model_state_dict'][name]).abs().max())
        for name, value in latest['model_state_dict'].items() if name.startswith('actor.'))
    assert parameter_change > 0.
    protocol = json.loads((ROOT/'configs/g07_full_racket_eval_v4_r3.json').read_text())
    protocol['_root'] = str(ROOT)
    rows = json.loads(episodes_raw)
    assert len({(r['env_id'], r['episode_index']) for r in rows}) == len(rows)
    assert all(r['ground_init'] and r['source_training_index'] == r['env_id']%128 for r in rows)
    checked = [independent_episode(row, protocol) for row in rows]
    counts = dict(completed_episodes=len(rows), air=sum(row['airborne_contact'] for row in checked),
        conditional_recovery_count=sum(row['conditional_recovery'] for row in checked),
        complete=sum(row['complete'] for row in checked), numerical_failures=sum(r['numerical_failures'] for r in rows))
    assert all(progress[k] == v for k,v in counts.items())
    owners = json.loads((ROOT/'reports/G07_full_racket_v4/live_tangent_trial_20261025/PREDECLARED_PLAN.json').read_text())['source_hashes']
    assert all(digest((ROOT/name).read_bytes()) == expected for name, expected in owners.items())
    identities = []
    for pid, needle in [(1638238, 'joint_placement_resume32_queue01/run.py'),
            (1642834, 'run_gpu_job.py'), (1642846, 'train_full_racket_residual_ppo_v4.py'),
            (1651319, 'live_tangent_trial_20261025/queue.py')]:
        path = Path(f'/proc/{pid}')
        state = (path/'stat').read_text().split(') ',1)[1].split()[0]
        command = (path/'cmdline').read_bytes().replace(b'\0', b' ').decode()
        assert state not in ['Z','X'] and needle in command
        identities.append(dict(pid=pid, process_state=state, expected_command_contains=needle, actual_command=command))
    compressed = gzip.compress(episodes_raw, compresslevel=1, mtime=0)
    (HERE/'closed_training_episodes.json.gz').write_bytes(compressed)
    (HERE/'progress.json').write_bytes(progress_raw)
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(),
        scope='Stable actually completed update of an ongoing native learner; all closed TRAIN records independently rescored. Not a frozen policy score or formal acceptance.',
        actual_iteration=update, current_job_control_env_steps=steps,
        cumulative_new_learner_control_env_steps=latest['native']['total_timesteps'],
        finite_model_and_Adam=True, all17_Adam_states_advanced=True,
        Adam_step_increment_per_parameter=sorted(set(increments)),
        normalizer_count_increments=normalizer_increments,
        actor_weight_max_absolute_change_from_iteration8=parameter_change,
        independent_all_closed_changing_policy_counts=counts,
        actual_diagnostic_checkpoint=str(checkpoint_path.relative_to(ROOT)), diagnostic_checkpoint_sha256=digest(checkpoint_raw),
        canonical_source_sha256=digest(SOURCE.read_bytes()),
        all_closed_source_json_sha256=digest(episodes_raw), all_closed_snapshot_gzip_sha256=digest(compressed),
        source_files_verified=len(owners), verified_live_processes=identities,
        source_program_sha256=digest(Path(__file__).read_bytes()),
        actual_native_job_restarted=False, frozen_policy_performance=False,
        formal_cases_read=False, formal_acceptance=False, goal_complete=False)
    (HERE/'REPORT.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(update=update, counts=counts, steps=steps, finite_model_and_Adam=True,
        actor_weight_change=parameter_change, verified_source_files=len(owners))), flush=True)


if __name__ == '__main__':
    main()
