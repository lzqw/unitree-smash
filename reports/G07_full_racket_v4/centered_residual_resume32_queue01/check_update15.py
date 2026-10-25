"""Check real resumed update15 and the first completed2048 ground episodes."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from quadsmash.evaluation_v4 import independent_episode


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = HERE/'ROOT_ACTUAL_UPDATE15_FIRST_COHORT_CHECK.json'
    assert not output.exists(), 'Preserve previous observation'
    reports = ROOT/'reports/G07_full_racket_v4'
    run = reports/'residual_ppo_centered_base8_n2048_resume32_01'
    source_path = reports/'residual_ppo_centered_base8_n2048_probe8_02/checkpoint.pt'
    # RSL diagnostic filename14 is saved after its actual completed update15.
    diagnostic_path = run/'ppo/model_14.pt'
    source = torch.load(source_path, map_location='cpu', weights_only=False)
    diagnostic = torch.load(diagnostic_path, map_location='cpu', weights_only=False)
    assert source['iter'] == 8 and diagnostic['iter'] == 15
    assert source['native']['canonical_restart'] and not diagnostic['native']['canonical_restart']
    assert source['native']['training_signature'] == diagnostic['native']['training_signature']
    steps = 7*2048*128
    assert diagnostic['native']['total_timesteps'] == source['native']['total_timesteps'] + steps == 3932160
    assert all(torch.isfinite(value).all() for value in diagnostic['model_state_dict'].values() if torch.is_tensor(value))
    before, after = source['optimizer_state_dict']['state'], diagnostic['optimizer_state_dict']['state']
    assert set(before) == set(after) and len(after) == 17
    assert all(float(after[key]['step'])-float(before[key]['step']) == 140 for key in before)
    assert all(torch.isfinite(value).all() for state in after.values() for value in state.values() if torch.is_tensor(value))
    names = ['actor_obs_normalizer.count', 'critic_obs_normalizer.count']
    assert all(int(diagnostic['model_state_dict'][name])-int(source['model_state_dict'][name]) == steps for name in names)
    actor_changed = any(not torch.equal(value, source['model_state_dict'][name]) for name, value
                        in diagnostic['model_state_dict'].items() if name.startswith('actor.') and torch.is_tensor(value))
    assert actor_changed
    initial = source['native']['environment']['tensors']['episode_index'].cpu().numpy()
    boundary = diagnostic['native']['environment']['tensors']['episode_index'].cpu().numpy()
    assert initial.shape == boundary.shape == (2048,)
    raw = (run/'episodes.partial.json').read_bytes()
    rows = json.loads(raw)
    by_lane = [[] for _ in range(2048)]
    for row in rows:
        by_lane[row['env_id']].append(row)
    for lane, closed in enumerate(by_lane):
        ids = [row['episode_index'] for row in closed]
        assert ids == list(range(int(initial[lane]), int(initial[lane])+len(ids)))
        assert int(boundary[lane]) <= int(initial[lane])+len(ids)
        assert ids, 'First ground cohort must have actually closed on every lane'
    first = [lane[0] for lane in by_lane]
    assert len(first) == 2048
    assert all(row['ground_init'] and row['source_training_index'] == row['env_id'] % 128 for row in rows)
    assert all(row['episode_index'] == int(initial[row['env_id']]) for row in first)
    protocol = read(ROOT/'configs/g07_full_racket_eval_v4_r3.json'); protocol['_root'] = str(ROOT)

    def counts(selected):
        checked = [independent_episode(row, protocol) for row in selected]
        return dict(actual_closed_episodes=len(selected), actual_racket_contacts=sum(row['had_contact'] for row in selected),
            air=sum(value['airborne_contact'] for value in checked),
            conditional_recovery=sum(value['conditional_recovery'] for value in checked),
            complete=sum(value['complete'] for value in checked),
            qualified_jump_and_recovery=sum(row['jump']['success'] for row in selected),
            numerical_failures=sum(row['numerical_failures'] for row in selected))

    overall, cohort = counts(rows), counts(first)
    assert overall['numerical_failures'] == cohort['numerical_failures'] == 0
    per_case = [dict(TRAIN=index, **counts([row for row in first if row['source_training_index'] == index])) for index in range(128)]
    assert all(row['actual_closed_episodes'] == 16 for row in per_case)
    progress = read(run/'progress.json')
    assert progress['update'] >= 15 and progress['trajectory_recording']['enabled']
    assert progress['trajectory_recording']['disabled_error'] is None
    assert progress['numerical_failures'] == 0
    launch_state = read(HERE/'state.json')
    ledger = read(ROOT/'status/GPU_LEDGER.json')
    active = [row for row in ledger['runs'] if row['status'] == 'RUNNING']
    assert len(active) == 1 and Path(active[0]['output']) == run
    pids = [launch_state['queue_pid'], active[0]['pid'], active[0]['child_pid']]
    for pid in pids:
        os.kill(pid, 0)
    assert str(run).encode() in Path('/proc', str(pids[2]), 'cmdline').read_bytes()
    assert b'CUDA_VISIBLE_DEVICES=0' in Path('/proc', str(pids[2]), 'environ').read_bytes().split(b'\0')
    pinned_counts = []
    for path in [reports/'centered_residual_pilot8_queue02/PREDECLARED_PLAN.json', HERE/'PREDECLARED_PLAN.json']:
        expected = read(path)['source_hashes']
        for name, digest in expected.items():
            assert sha(ROOT/name) == digest, 'Active source changed: '+name
        pinned_counts.append(len(expected))
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(),
        previous_goal_turn_classification='Verified wait: confirmed actual native1687066 and workflow1678986 live; delivered the already uploaded latest real video.',
        source_iteration=8, independently_checked_completed_iteration=15,
        diagnostic_is_canonical_restart=False, actual_current_job_steps_at_update15=steps,
        actual_cumulative_new_learner_steps_at_update15=3932160,
        all17_Adam_states_preserved_and_advanced140_steps=True, both_normalizer_counts_advanced=steps,
        all_model_and_Adam_states_finite=True, actor_weights_changed=True,
        exact_source_inclusive_episode_ID_prefixes_all2048_lanes=True,
        source_episode_ID_min=int(initial.min()), source_episode_ID_max=int(initial.max()),
        first_actual2048_ground_episode_cohort=cohort, first_cohort_per_original_TRAIN_case=per_case,
        independent_observed_closed_changing_policy_counts=overall,
        actual_partial_snapshot_sha256=hashlib.sha256(raw).hexdigest(),
        actual_partial_snapshot_note='Live atomic producer file may advance; final raw episodes must contain this exact source-inclusive prefix. No claim that all raw rows ended at the diagnostic update boundary.',
        progress_observed={name:progress[name] for name in ['update','actual_control_env_steps','completed_episodes','air','conditional_recovery_count','complete','numerical_failures','trajectory_recording']},
        native_process_verified_live=dict(workflow_pid=pids[0], guard_pid=pids[1], native_pid=pids[2], gpu=0),
        pinned_source_counts_verified=pinned_counts,
        source_sha256={str(path.relative_to(ROOT)):sha(path) for path in [Path(__file__), source_path, diagnostic_path]},
        scope='Actual changing-policy native training only. First cohort uses actual source IDs, not a false episode0 assumption. No frozen performance inherited.',
        frozen_policy_performance=False, formal_cases_read=False, formal_acceptance=False, goal_complete=False)
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({name:result[name] for name in ['independently_checked_completed_iteration','actual_current_job_steps_at_update15',
        'first_actual2048_ground_episode_cohort','independent_observed_closed_changing_policy_counts','native_process_verified_live','pinned_source_counts_verified']}, indent=2))


if __name__ == '__main__':
    main()
