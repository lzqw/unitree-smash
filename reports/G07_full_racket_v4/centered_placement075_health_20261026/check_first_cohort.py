"""Check every actual first ground episode after the seven-update boundary."""
from datetime import datetime, timezone
import csv
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from quadsmash.evaluation_v4 import independent_episode


def main():
    output = HERE/'ACTUAL_ALL2048_FIRST_COHORT_CHECK.json';assert not output.exists()
    path = ROOT/'reports/G07_full_racket_v4/centered_placement075_pilot8_queue01/run.py'
    spec = importlib.util.spec_from_file_location('active075cohort',path)
    queue = importlib.util.module_from_spec(spec);spec.loader.exec_module(queue)
    core = queue.core;plan = core.read(queue.PLAN)
    expected = dict(plan['source_hashes']);expected[str(queue.PLAN.relative_to(ROOT))] = core.sha(queue.PLAN)
    queue.template.check_sources(expected)
    init = queue.validate_training_inputs(core.read(queue.TRAIN/'INPUTS.json'))
    checkpoint = queue.TRAIN/'ppo/model_6.pt'
    native = torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert native['iter'] == 7 and not native['native']['canonical_restart']
    assert native['native']['total_timesteps'] == 1835008 and native['native']['environment']['num_envs'] == 2048
    model = native['model_state_dict']
    assert all(torch.isfinite(v).all() for v in model.values() if torch.is_tensor(v))
    assert int(model['actor_obs_normalizer.count']) == init['source_actor_normalizer_count']+1835008
    assert int(model['critic_obs_normalizer.count']) == 1835008
    steps = queue.optimizer_steps(native);assert len(steps) == 17
    assert all(v == (140 if k.startswith('critic.') else 100) for k,v in steps.items())
    assert all(torch.isfinite(v).all() for state in native['optimizer_state_dict']['state'].values()
               for v in state.values() if torch.is_tensor(v))
    terminal_ids = native['native']['environment']['tensors']['episode_index'].tolist()
    assert len(terminal_ids) == 2048 and min(terminal_ids) >= 1
    source_bytes = (queue.TRAIN/'episodes.partial.json').read_bytes();rows = json.loads(source_bytes)
    boundary_rows = [r for r in rows if r['episode_index'] < terminal_ids[r['env_id']]]
    seen = [0]*2048
    for r in boundary_rows:
        assert r['episode_index'] == seen[r['env_id']] and r['source_training_index'] == r['env_id']%128 and r['ground_init']
        seen[r['env_id']] += 1
    assert seen == terminal_ids
    first = sorted([r for r in boundary_rows if r['episode_index'] == 0],key=lambda r:r['env_id'])
    assert len(first) == 2048 and [r['env_id'] for r in first] == list(range(2048))
    protocol = core.read(ROOT/'configs/g07_full_racket_eval_v4_r3.json');protocol['_root'] = str(ROOT)
    outcomes = [independent_episode(r,protocol) for r in first]
    attacks = []
    for row,outcome in zip(first,outcomes):
        v = np.asarray(row['outgoing_velocity_mps'])
        attacks.append(bool(outcome['airborne_contact'] and row['released'] and np.isfinite(v).all()
            and np.linalg.norm(v) >= 8 and v[0] > 0 and v[2] <= -1))
    def counts(indices):
        return dict(episodes=len(indices),air=sum(outcomes[i]['airborne_contact'] for i in indices),
            conditional_recovery=sum(outcomes[i]['conditional_recovery'] for i in indices),complete=sum(outcomes[i]['complete'] for i in indices),
            numerical_failures=sum(first[i]['numerical_failures'] for i in indices),racket_contacts=sum(first[i]['had_contact'] for i in indices),
            attack_outgoing=sum(attacks[i] for i in indices),crossed_net=sum(first[i]['crossed_net'] for i in indices),
            landed_target=sum(first[i]['landed_target'] for i in indices),body_landing=sum(first[i]['jump']['body_landing'] for i in indices))
    aggregate = counts(list(range(2048)));assert aggregate['numerical_failures'] == 0
    per_case = [dict(TRAIN=case,**counts([i for i in range(2048) if i%128 == case])) for case in range(128)]
    assert all(row['episodes'] == 16 for row in per_case)
    snapshot = HERE/'FIRST2048_EPISODES_SNAPSHOT.json'
    with snapshot.open('x') as file:json.dump(first,file);file.write('\n')
    csv_path = HERE/'FIRST2048_BY_TRAIN_CASE.csv'
    with csv_path.open('x',newline='') as file:
        writer = csv.DictWriter(file,fieldnames=list(per_case[0]),lineterminator='\n');writer.writeheader();writer.writerows(per_case)
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(),actual_boundary_iteration=7,actual_new_job_control_env_steps=1835008,
        actual_boundary_diagnostic_sha256=core.sha(checkpoint),all_2048_first_ground_episodes_closed=True,
        every_original_TRAIN_case_repeated16_times=True,all_lane_episode_chains_exact_at_boundary7=True,
        all_boundary_closed_episodes=len(boundary_rows),independent_first_cohort_counts=aggregate,
        first_cohort_overlapping_terminal_flags={key:sum(r['training_terminal_reasons'][key] for r in first) for key in first[0]['training_terminal_reasons']},
        actor_normalizer_count=int(model['actor_obs_normalizer.count']),critic_normalizer_count=int(model['critic_obs_normalizer.count']),
        source_actor_normalizer_count=init['source_actor_normalizer_count'],all17_Adam_states_finite=True,actor_logstd_Adam_steps=100,critic_Adam_steps=140,
        first_cohort_snapshot_sha256=core.sha(snapshot),first_cohort_snapshot_bytes=snapshot.stat().st_size,
        per_case_table_sha256=core.sha(csv_path),source_current_episode_file_sha256=__import__('hashlib').sha256(source_bytes).hexdigest(),
        limitations='All first TRAIN episodes, but across changing actors/exploration. This is not a frozen-policy score, a new formal A/B result or a matched comparison to zero-noise source40.',
        source_sha256_count=len(expected),all_held_sources_unchanged=True,frozen_policy_performance=False,
        formal_cases_read=False,formal_acceptance=False,goal_complete=False)
    queue.template.save_once(output,result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':main()
