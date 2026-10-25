"""Verify the actual first actor update after two critic-only warmups.

Closed TRAIN rows at this early boundary are biased toward early failures.
They do not measure one frozen actor or replace the pending full assessment.
"""
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from quadsmash.evaluation_v4 import independent_episode


def main():
    output = HERE/'ACTUAL_UPDATE3_POST_WARMUP_CHECK.json'
    assert not output.exists()
    path = ROOT/'reports/G07_full_racket_v4/centered_placement075_pilot8_queue01/run.py'
    spec = importlib.util.spec_from_file_location('active075',path)
    queue = importlib.util.module_from_spec(spec);spec.loader.exec_module(queue)
    core = queue.core;plan = core.read(queue.PLAN)
    expected = dict(plan['source_hashes']);expected[str(queue.PLAN.relative_to(ROOT))] = core.sha(queue.PLAN)
    queue.template.check_sources(expected)
    inputs = core.read(queue.TRAIN/'INPUTS.json');init = queue.validate_training_inputs(inputs)
    progress_bytes = (queue.TRAIN/'progress.json').read_bytes();progress = json.loads(progress_bytes)
    assert progress['update'] == 3 and progress['actor_warmup'] is False
    assert progress['actual_control_env_steps'] == 3*2048*128
    native_path = queue.TRAIN/'ppo/model_2.pt'
    native = torch.load(native_path,map_location='cpu',weights_only=False)
    assert native['iter'] == 3 and not native['native']['canonical_restart']
    assert native['native']['total_timesteps'] == 786432 and native['native']['environment']['num_envs'] == 2048
    model = native['model_state_dict']
    assert all(torch.isfinite(v).all() for v in model.values() if torch.is_tensor(v))
    assert int(model['actor_obs_normalizer.count']) == init['source_actor_normalizer_count']+786432
    assert int(model['critic_obs_normalizer.count']) == 786432
    steps = queue.optimizer_steps(native)
    assert len(steps) == len(native['optimizer_state_dict']['state']) == 17
    assert all(value == (60 if name.startswith('critic.') else 20) for name,value in steps.items())
    assert all(torch.isfinite(v).all() for state in native['optimizer_state_dict']['state'].values()
               for v in state.values() if torch.is_tensor(v))
    source = torch.load(Path(plan['selected']['policy']),map_location='cpu',weights_only=False)
    actor = torch.jit.load(io.BytesIO(source['residual_actor_jit']),map_location='cpu').eval().state_dict()
    changes = {name:float((model['actor.'+name[2:]]-value).abs().max())
               for name,value in actor.items() if name.startswith('1.')}
    assert any(value > 0 for value in changes.values())
    raw_bytes = (queue.TRAIN/'episodes.partial.json').read_bytes();rows = json.loads(raw_bytes)
    seen = [0]*2048
    for row in rows:
        lane = row['env_id']
        assert 0 <= lane < 2048 and row['episode_index'] == 0 and row['ground_init']
        assert row['source_training_index'] == lane%128
        seen[lane] += 1
    assert seen == native['native']['environment']['tensors']['episode_index'].tolist()
    assert set(seen) == {0,1}
    protocol = core.read(ROOT/'configs/g07_full_racket_eval_v4_r3.json');protocol['_root'] = str(ROOT)
    outcomes = [independent_episode(row,protocol) for row in rows]
    counts = dict(completed_episodes=len(rows),air=sum(v['airborne_contact'] for v in outcomes),
        conditional_recovery_count=sum(v['conditional_recovery'] for v in outcomes),
        complete=sum(v['complete'] for v in outcomes),numerical_failures=sum(r['numerical_failures'] for r in rows))
    assert all(progress[key] == value for key,value in counts.items()) and counts['numerical_failures'] == 0
    reasons = {key:sum(bool(r['training_terminal_reasons'][key]) for r in rows)
               for key in rows[0]['training_terminal_reasons']}
    recorder = progress['trajectory_recording']
    assert recorder['enabled'] and recorder['disabled_error'] is None
    assert recorder['verified_full_public_rows'] == 786432 and recorder['closed_episodes_observed'] == len(rows)
    assert core.finite(progress['losses'])
    active = [r for r in core.read(ROOT/'status/GPU_LEDGER.json')['runs'] if r['status'] == 'RUNNING']
    assert len(active) == 1 and Path(active[0]['output']) == queue.TRAIN
    for name in ['pid','child_pid']:assert core.live(active[0][name],str(queue.TRAIN))
    assert core.live(1722226,'centered_placement075_pilot8_queue01/run.py')
    progress_snapshot,rows_snapshot = HERE/'UPDATE3_PROGRESS_SNAPSHOT.json',HERE/'UPDATE3_RAW_EPISODES_SNAPSHOT.json'
    for path,data in [(progress_snapshot,progress_bytes),(rows_snapshot,raw_bytes)]:
        with path.open('xb') as file:file.write(data)
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(),actual_queue_pid=1722226,
        actual_guard_pid=active[0]['pid'],actual_native_pid=active[0]['child_pid'],actual_stage_update=3,
        actual_new_learner_control_env_steps=786432,canonical_restart=False,
        actual_diagnostic_sha256=core.sha(native_path),actual_source_public_policy_sha256=init['source_policy_sha256'],
        actor_normalizer_count=int(model['actor_obs_normalizer.count']),critic_normalizer_count=int(model['critic_obs_normalizer.count']),
        source_actor_normalizer_count=init['source_actor_normalizer_count'],all17_Adam_states_finite=True,
        actual_actor_logstd_Adam_steps=20,actual_critic_Adam_steps=60,
        copied_actor_tensor_max_changes=changes,at_least_one_actual_actor_tensor_changed=True,
        exact_fresh_episode_chains_at_actual_update3=True,independent_partial_changing_policy_counts=counts,
        lanes_with_first_episode_closed=sum(seen),lanes_with_first_episode_still_open=seen.count(0),
        overlapping_actual_early_terminal_reasons=reasons,trace_recording=recorder,
        actual_losses={key:value for key,value in progress['losses'].items() if key not in ['behavior_actions','post_update_actions']},
        immutable_progress_snapshot_sha256=core.sha(progress_snapshot),immutable_raw_episode_snapshot_sha256=core.sha(rows_snapshot),
        source_sha256_count=len(expected),all_held_sources_unchanged=True,
        limitations='This boundary contains early-terminated episodes while756 original first episodes remain open. No frozen-policy rate or performance improvement is inferred; source8/40 scores are not inherited.',
        frozen_policy_performance=False,formal_cases_read=False,formal_acceptance=False,goal_complete=False)
    queue.template.save_once(output,result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
