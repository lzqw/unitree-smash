"""Preserve and independently check one full native TRAIN episode during learning."""
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
from quadsmash.evaluation_v4 import FAULTS, independent_episode


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    assert not (HERE / 'REPORT.json').exists(), 'Never overwrite preserved native evidence'
    protocol_path = ROOT / 'configs/g07_full_racket_eval_v4_r3.json'
    protocol = json.loads(protocol_path.read_text())
    protocol['_root'] = str(ROOT)
    for _ in range(5):
        progress_bytes = (RUN / 'progress.json').read_bytes()
        progress = json.loads(progress_bytes)
        raw = (RUN / 'episodes.partial.json').read_bytes()
        rows = json.loads(raw)
        checks = [independent_episode(row, protocol) for row in rows]
        counts = dict(completed_episodes=len(rows), air=sum(v['airborne_contact'] for v in checks),
            conditional_recovery_count=sum(v['conditional_recovery'] for v in checks),
            complete=sum(v['complete'] for v in checks),
            numerical_failures=sum(v['numerical_failures'] for v in rows))
        if progress_bytes == (RUN / 'progress.json').read_bytes() and all(progress[k] == v for k, v in counts.items()):
            break
        time.sleep(.25)
    else:
        raise RuntimeError('Concurrent progress write; re-poll the same live native job')
    selected = [row for row in rows if row['env_id'] == 275 and row['episode_index'] == 1]
    assert len(selected) == 1
    row = selected[0]
    checked = independent_episode(row, protocol)
    assert checked == dict(airborne_contact=True, conditional_recovery=True, complete=True, fault=False)
    assert row['source_training_index'] == 19 and row['case_id'] == 'TRAIN19_LANE275_EP1'
    assert row['contact_count'] == row['active_rally_launch_count'] == 1 and not row['frame_hit']
    assert not any(row[name] for name in FAULTS) and row['numerical_failures'] == 0
    jump = row['jump']
    assert jump['takeoff_time_s'] < row['contact_time_s'] < jump['landing_time_s']
    assert not row['training_early_termination'] and row['terminal_native_state']['time_s'] == 8.
    lane = sorted((r for r in rows if r['env_id'] == 275 and r['episode_index'] <= 1),
        key=lambda r: r['episode_index'])
    assert [r['episode_index'] for r in lane] == [0, 1]
    durations = [round(r['terminal_native_state']['time_s'] / .01) for r in lane]
    assert all(abs(r['terminal_native_state']['time_s'] - n*.01) < 1e-5 for r, n in zip(lane, durations))
    start_tick, end_tick = durations[0], sum(durations)
    inputs = json.loads((RUN / 'INPUTS.json').read_text())
    initial_iteration = inputs['initialization']['initial_iteration']
    rollout_ticks = inputs['runner']['num_steps_per_env']
    first_actor = initial_iteration + start_tick // rollout_ticks
    last_actor = initial_iteration + (end_tick - 1) // rollout_ticks
    contact_tick = start_tick + row['contact_time_s'] / .01
    contact_actor = initial_iteration + int(contact_tick // rollout_ticks)
    assert (start_tick, end_tick, first_actor, last_actor, contact_actor) == (800, 1600, 14, 20, 16)
    source_hashes = json.loads((RUN / 'source_hashes.json').read_text())
    relevant = [str(protocol_path.relative_to(ROOT)), 'scripts/train_full_racket_residual_ppo_v4.py',
        'quadsmash/evaluation_v4.py', 'quadsmash/tasks/stand/full_racket_train_env_v4.py',
        'quadsmash/tasks/stand/full_racket_env_v4.py', 'quadsmash/tasks/stand/coupled_env.py',
        'quadsmash/tasks/stand/jump_env.py']
    for name in relevant:
        assert source_hashes[name] == sha(ROOT / name), 'Runtime source changed: ' + name
    for name, expected in protocol['physical_configs'].items():
        assert sha(ROOT / name) == expected
    assert sha(ROOT / protocol['asset']['path']) == protocol['asset']['sha256']
    lineage = []
    # Actor states encountered during this episode, plus the first checkpoint
    # saved after it closed. These are diagnostic, not canonical native resumes.
    for iteration in range(first_actor, last_actor + 2):
        checkpoint = RUN / f'ppo/model_{iteration - 1}.pt'
        model = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert model['iter'] == iteration
        finite = all(torch.isfinite(v).all() for v in model['model_state_dict'].values() if torch.is_tensor(v))
        finite = finite and all(torch.isfinite(v).all() for s in model['optimizer_state_dict']['state'].values()
            for v in s.values() if torch.is_tensor(v))
        assert finite
        lineage.append(dict(path=str(checkpoint.relative_to(ROOT)), sha256=sha(checkpoint), iteration=iteration,
            model_optimizer_tensors_finite=True, canonical_native_resume=False))
    latest_checkpoint = RUN / f'ppo/model_{progress["update"] - 1}.pt'
    latest = torch.load(latest_checkpoint, map_location='cpu', weights_only=False)
    assert latest['iter'] == progress['update']
    assert latest['native']['total_timesteps'] == 131072 + progress['actual_control_env_steps']
    assert latest['native']['environment']['num_envs'] == 512
    snapshot = gzip.compress(raw, mtime=0)
    (HERE / 'all_closed_training_episodes.json.gz').write_bytes(snapshot)
    assert gzip.decompress(snapshot) == raw
    (HERE / 'progress.json').write_bytes(progress_bytes)
    save('complete_episode.json', row)
    save('same_lane_previous_and_complete.json', lane)
    save('all_closed_TRAIN19_lanes.json', [r for r in rows if r['source_training_index'] == 19])
    save('weight_lineage.json', lineage)
    contact_record_path = RUN / 'contacts.json'
    contacts = None
    if contact_record_path.exists():
        contacts = [c for c in json.loads(contact_record_path.read_text())
            if c['env_id'] == 275 and c['episode_index'] == 1]
        assert contacts
        save('native_contacts.json', contacts)
    velocity = np.asarray(row['outgoing_velocity_mps'])
    report = dict(created_at=datetime.now(timezone.utc).isoformat(),
        scope='One independently rechecked complete TRAIN19 native episode during changing-policy stochastic PPO learning. No frozen-policy performance or formal A/B claim.',
        native_case_id=row['case_id'], env_id=275, episode_index=1, source_training_index=19,
        independent_original_gates=checked, contact_point_m=row['contact_point_m'],
        outgoing_velocity_mps=row['outgoing_velocity_mps'], outgoing_speed_mps=float(np.linalg.norm(velocity)),
        crossing_point_m=row['crossing_point_m'], net_margin_m=row['net_margin_m'],
        first_landing_m=row['first_landing_m'], recovery_s=jump['recovery_s'],
        total_com_rise_m=jump['total_com_rise_m'], airborne_s=jump['airborne_s'],
        event_times_s={k: row[k] for k in ['contact_time_s', 'release_time_s', 'crossing_time_s', 'first_landing_time_s']},
        jump=jump, all_closed_training_counts_at_snapshot=counts, snapshot_update=progress['update'],
        current_job_control_env_steps=progress['actual_control_env_steps'],
        cumulative_learner_control_env_steps=latest['native']['total_timesteps'],
        job_control_ticks=dict(start=start_tick, end=end_tick),
        actor_iterations_encountered=list(range(first_actor, last_actor + 1)),
        contact_rollout_actor_iteration=contact_actor,
        diagnostic_checkpoint_at_snapshot=dict(path=str(latest_checkpoint.relative_to(ROOT)),
            sha256=sha(latest_checkpoint), canonical_native_resume=False),
        current_source_matches_native_snapshot=True,
        runtime_source_hashes={k: source_hashes[k] for k in relevant},
        protocol_sha256=sha(protocol_path), asset_sha256=protocol['asset']['sha256'],
        physical_configs=protocol['physical_configs'],
        full_closed_snapshot_sha256=sha(HERE / 'all_closed_training_episodes.json.gz'),
        full_closed_uncompressed_sha256=hashlib.sha256(raw).hexdigest(),
        native_contact_record_match_pending=contacts is None,
        limitations=[
            'Actor and observation normalizers changed during this episode; it does not demonstrate a single frozen actor can reproduce it.',
            'The exact contact records are flushed at the end of the active native training job; match them after that existing job exits.',
            'No full50Hz body trajectory was saved by this PPO trainer, so an exact-motion video of this episode cannot be reconstructed from these event and terminal records.',
            'Ground initialization of this later episode is backed by the unchanged native reset implementation and its ground flag; an actual episode1 initial frame was not separately logged.',
            'The terminal ball position is after first ground and dead-ball rolling. Use first_landing_m for the scored landing and hide the dead ball in terminal illustrations.',
            'Peak arm torque and speed are aggregate episode metrics; no per-joint time history or real-hardware calibration is inferred.',
            'Earlier complete full-racket CEM episodes were both TRAIN65; this is an additional TRAIN case, not broad-distribution acceptance.'],
        formal_cases_read=False, formal_acceptance=False, goal_complete=False)
    save('REPORT.json', report)
    print(json.dumps({k: report[k] for k in ['native_case_id', 'snapshot_update',
        'all_closed_training_counts_at_snapshot', 'outgoing_speed_mps', 'first_landing_m',
        'recovery_s', 'native_contact_record_match_pending']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
