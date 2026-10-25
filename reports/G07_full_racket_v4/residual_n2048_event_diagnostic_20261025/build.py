"""Recompute a real recorded TRAIN prefix, keeping incomplete cohorts explicit."""
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RUN = ROOT / 'reports/G07_full_racket_v4/residual_ppo_n2048_resume64_01'
OBSERVATION = ROOT / 'reports/G07_full_racket_v4/residual_iter96_formal_queue01/NATIVE_PROGRESS_AT_REGISTRATION.json'
sys.path.insert(0, str(ROOT))
from quadsmash.evaluation_v4 import FAULTS, independent_episode


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not (HERE / 'REPORT.json').exists(), 'Preserve existing diagnostic evidence'
    observation = json.loads(OBSERVATION.read_bytes())
    length = observation['independently_recomputed_training_counts']['completed_episodes']
    source = RUN / 'episodes.json'
    if not source.exists():
        source = RUN / 'episodes.partial.json'
    rows = json.loads(source.read_bytes())[:length]
    reconstructed = (json.dumps(rows, indent=2) + '\n').encode()
    assert hashlib.sha256(reconstructed).hexdigest() == observation['partial_episodes_sha256']
    protocol_path = ROOT / 'configs/g07_full_racket_eval_v4_r3.json'
    protocol = json.loads(protocol_path.read_bytes())
    protocol['_root'] = str(ROOT)
    checked = [independent_episode(row, protocol) for row in rows]
    assert len({(r['env_id'], r['episode_index']) for r in rows}) == len(rows)
    assert all(r['source_training_index'] == r['env_id'] % 128 for r in rows)
    velocity = np.asarray([r['outgoing_velocity_mps'] for r in rows])
    speed = np.linalg.norm(velocity, axis=1)
    air = np.asarray([v['airborne_contact'] for v in checked])
    recovered = np.asarray([v['conditional_recovery'] for v in checked])
    complete = np.asarray([v['complete'] for v in checked])
    target = np.asarray([r['landed_target'] for r in rows])
    attack = (air & np.asarray([r['released'] for r in rows]) & np.isfinite(velocity).all(1)
              & (speed >= 8) & (velocity[:, 0] > 0) & (velocity[:, 2] <= -1))
    counts = dict(completed_episodes=len(rows), air=int(air.sum()),
        conditional_recovery_count=int(recovered.sum()), complete=int(complete.sum()),
        numerical_failures=sum(r['numerical_failures'] for r in rows))
    assert counts == observation['independently_recomputed_training_counts']
    cohorts = []
    complete_cohort_indices = []
    for index in sorted({r['episode_index'] for r in rows}):
        mask = np.asarray([r['episode_index'] == index for r in rows])
        lane_ids = {r['env_id'] for r in rows if r['episode_index'] == index}
        all_closed = int(mask.sum()) == 2048 and lane_ids == set(range(2048))
        if all_closed:
            complete_cohort_indices.append(index)
        cohorts.append(dict(episode_index=index, closed=int(mask.sum()), all2048_lanes_closed=all_closed,
            air=int((air & mask).sum()), conditional_recovery=int((recovered & mask).sum()),
            air_attacks=int((attack & mask).sum()), targets=int((target & mask).sum()),
            complete=int((complete & mask).sum())))
    assert complete_cohort_indices == list(range(len(complete_cohort_indices)))
    per_case = []
    fully_closed = np.asarray([r['episode_index'] in complete_cohort_indices for r in rows])
    for case in range(128):
        mask = np.asarray([r['source_training_index'] == case for r in rows]) & fully_closed
        assert int(mask.sum()) == len(complete_cohort_indices) * 16
        per_case.append(dict(source_TRAIN_index=case, executions=int(mask.sum()),
            air=int((air & mask).sum()), conditional_recovery=int((recovered & mask).sum()),
            air_attacks=int((attack & mask).sum()), targets=int((target & mask).sum()),
            complete=int((complete & mask).sum())))
    landing = np.asarray([r['first_landing_m'] for r in rows])
    court = json.loads((ROOT / 'configs/court_net_v4.json').read_bytes())
    low, high = court['target_x_limits_m']
    distance = np.sqrt(np.maximum(low - landing[:, 0], 0)**2
        + np.maximum(landing[:, 0] - high, 0)**2
        + np.maximum(abs(landing[:, 1]) - court['target_y_half_width_m'], 0)**2)
    near = sorted(np.where(attack & recovered & np.isfinite(distance) & ~complete)[0],
                  key=lambda i: distance[i])[:5]
    overlap = np.where(attack & target)[0]
    selected = sorted(set(near) | set(overlap))
    detail = [dict(case_id=rows[i]['case_id'], source_TRAIN_index=rows[i]['source_training_index'],
        env_id=rows[i]['env_id'], episode_index=rows[i]['episode_index'],
        selection=('attack_target_overlap' if i in overlap else 'nearest_attack_recovery_target_miss'),
        outgoing_speed_mps=float(speed[i]), first_landing_m=landing[i].tolist(),
        target_distance_m=float(distance[i]), conditional_recovery=bool(recovered[i]),
        recovery_s=rows[i]['jump']['recovery_s'], complete=bool(complete[i]),
        fault_flags=[flag for flag in FAULTS if rows[i][flag]]) for i in selected]
    (HERE / 'SELECTED_NATIVE_EPISODES.json').write_text(
        json.dumps([rows[i] for i in selected], indent=2) + '\n')
    with (HERE / 'COMPLETE_COHORTS_PER_TRAIN_CASE.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_case[0]))
        writer.writeheader()
        writer.writerows(per_case)
    finding = dict(air_attacks=int(attack.sum()), targets=int(target.sum()),
        attack_target_overlap=int((attack & target).sum()),
        target_hits_below_attack_speed=int((target & (speed < 8)).sum()),
        attack_target_without_qualified_recovery=int((attack & target & ~recovered).sum()),
        body_landings=sum(r['jump']['body_landing'] for r in rows),
        body_landing_before_rally_preparation_ends=sum(r['jump']['body_landing']
            and r['terminal_native_state']['time_s'] < protocol['preparation_seconds'] for r in rows),
        fully_closed_cohort_indices=complete_cohort_indices,
        fully_closed_cohort_executions=int(fully_closed.sum()),
        fully_closed_cohort_TRAIN_cases_without_any_air=sum(r['air'] == 0 for r in per_case),
        per_case_air_count_distribution=dict(sorted(Counter(r['air'] for r in per_case).items())),
        fault_counts={flag: sum(r[flag] for r in rows) for flag in FAULTS})
    sources = [protocol_path, ROOT / 'configs/court_net_v4.json',
        ROOT / 'quadsmash/evaluation_v4.py', ROOT / 'quadsmash/physics/rally_rules_v4.py',
        ROOT / 'quadsmash/control/full_racket_residual_reward_v4.py',
        ROOT / 'quadsmash/tasks/stand/full_racket_residual_env_v4.py',
        RUN / 'INPUTS.json', OBSERVATION, Path(__file__).resolve()]
    artifacts = ['SELECTED_NATIVE_EPISODES.json', 'COMPLETE_COHORTS_PER_TRAIN_CASE.csv']
    result = dict(created_at=datetime.now(timezone.utc).isoformat(),
        scope='Real recorded changing-policy TRAIN event prefix, independently rescored. No new robot execution, candidate reward application, frozen-policy result or formal acceptance.',
        snapshot_update=observation['update'], snapshot_control_env_steps=observation['actual_control_env_steps'],
        original_snapshot_rows=length, original_snapshot_sha256=observation['partial_episodes_sha256'],
        exact_prefix_reconstructed_from=str(source.relative_to(ROOT)),
        reproduction='Read the first original_snapshot_rows from the same run episodes.partial.json or final episodes.json; json.dumps(indent=2)+newline must exactly match the recorded SHA before analysis.',
        independently_recomputed_original_counts=counts, findings=finding, cohorts=cohorts,
        selected_native_cases=detail, sources={str(p.relative_to(ROOT)): sha(p) for p in sources},
        artifacts={name: dict(sha256=sha(HERE / name), bytes=(HERE / name).stat().st_size) for name in artifacts},
        next_action='Keep the real original native workflow and source owners intact. Reassess frozen iter96 independently before promoting a changed-reward learner. The attack/target gap supports testing the prepared joint placement hypothesis if the fixed result remains below original thresholds; it does not prove candidate benefit or solve low contact/recovery coverage.',
        limitations=['All cohorts use changing actors, normalizers and exploration; their counts are not frozen-policy or distribution acceptance rates.',
            'Only cohorts explicitly marked all2048_lanes_closed contain every lane; incomplete later cohorts are censored and must not be interpreted as performance collapse.',
            'Targets and attacks overlap in five observed TRAIN episodes at this snapshot; three meet every complete gate and two lack qualified recovery. Keep these separate.',
            'A terminal body landing count does not by itself locate the physical cause or prove that any proposed reward change will improve recovery.',
            'Selected native episodes are posthoc diagnostic cases, not preselected video cases or a replay of missing full motion.',
            'No formal A/B cases were opened. All original model, physics, actuation, target bounds and acceptance thresholds are unchanged.'],
        formal_cases_read=False, formal_acceptance=False, goal_complete=False)
    temporary = HERE / 'REPORT.json.tmp'
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    temporary.replace(HERE / 'REPORT.json')
    print(json.dumps(dict(snapshot_update=observation['update'], counts=counts, findings=finding,
                         report=str((HERE / 'REPORT.json').relative_to(ROOT)))))


if __name__ == '__main__':
    main()
