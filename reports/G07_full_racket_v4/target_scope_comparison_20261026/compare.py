"""Read-only target-scope comparison of the existing fixed TRAIN128 execution."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path('/home/autodrive/LZQW/unitree_smash')
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from quadsmash.evaluation_v4 import FAULTS, independent_episode


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    plan = json.loads((OUT / 'PLAN.json').read_text())
    for path, expected in plan['input_sha256'].items():
        assert sha(ROOT / path) == expected, path
    assert not (OUT / 'RESULT.json').exists(), 'Preserve prior evidence'
    run = ROOT / plan['source_run']
    rows = json.loads((run / 'episodes.json').read_text())
    protocol = json.loads((ROOT / 'configs/g07_full_racket_eval_v4_r3.json').read_text())
    protocol['_root'] = str(ROOT)
    court = json.loads((ROOT / 'configs/court_net_v4.json').read_text())
    assert len(rows) == 128
    assert [r['source_training_index'] for r in rows] == list(range(128))
    assert all(r['episode_index'] == 0 for r in rows)
    canonical = [independent_episode(r, protocol) for r in rows]
    scopes = plan['proposed_scopes']
    tables = []
    summaries = {}
    for scope, bounds in scopes.items():
        complete_indices = []
        geometric_indices = []
        for row, checked in zip(rows, canonical):
            landing = np.asarray(row['first_landing_m'], dtype=float)
            velocity = np.asarray(row['outgoing_velocity_mps'], dtype=float)
            inside = bool(np.isfinite(landing).all()
                and bounds['x_limits_m'][0] <= landing[0] <= bounds['x_limits_m'][1]
                and abs(landing[1]) <= bounds['y_half_width_m']
                and court['opponent_x_limits_m'][0] <= landing[0] <= court['opponent_x_limits_m'][1]
                and abs(landing[1]) <= court['singles_width_m'] / 2)
            times = [row[k] for k in ('contact_time_s', 'release_time_s',
                                     'crossing_time_s', 'first_landing_time_s')]
            contact, release, crossing, first_landing = times
            tolerance = protocol['physics_dt_s']
            ordered = bool(np.isfinite(times).all()
                and contact >= protocol['preparation_seconds'] - tolerance
                and release >= contact and crossing + tolerance >= contact
                and first_landing + tolerance >= max(crossing, release))
            attack = bool(checked['airborne_contact'] and row['released']
                and np.isfinite(velocity).all() and np.linalg.norm(velocity) >= 8
                and velocity[0] > 0 and velocity[2] <= -1)
            # conditional_recovery already retains the original recovery and every fault check.
            complete = bool(row['ground_init'] and row['active_rally_launch_count'] == 1
                and attack and row['crossed_net'] and inside and ordered
                and checked['conditional_recovery'])
            if scope == 'original_precision_square':
                assert complete == checked['complete'], row['case_id']
            index = row['source_training_index']
            if complete:
                complete_indices.append(index)
            if inside:
                geometric_indices.append(index)
            tables.append(dict(scope=scope, source_training_index=index, case_id=row['case_id'],
                original_complete=int(checked['complete']), complete=int(complete),
                first_landing_inside_scope=int(inside), airborne_contact=int(checked['airborne_contact']),
                conditional_recovery=int(checked['conditional_recovery']), attack=int(attack),
                fault=int(checked['fault']), **{f'first_landing_{axis}_m': float(landing[i])
                                              for i, axis in enumerate('xyz')}))
        summaries[scope] = dict(bounds=bounds, complete=len(complete_indices),
            complete_rate=len(complete_indices) / 128, complete_training_indices=complete_indices,
            geometric_first_landings=len(geometric_indices),
            geometric_count_is_not_complete_success=True)
    assert [s['complete'] for s in summaries.values()] == [2, 12, 13]
    with (OUT / 'EPISODES.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(tables[0]))
        writer.writeheader()
        writer.writerows(tables)
    for path, expected in plan['input_sha256'].items():
        assert sha(ROOT / path) == expected, path
    result = dict(created_at=datetime.now(timezone.utc).isoformat(),
        scope='Offline target-only rescoring of one existing frozen teacher TRAIN128 execution; no new simulation.',
        source_run=plan['source_run'], input_sha256=plan['input_sha256'],
        unchanged=dict(airborne_contacts=sum(c['airborne_contact'] for c in canonical),
            conditional_recovery_count=sum(c['conditional_recovery'] for c in canonical),
            numerical_failures=sum(r['numerical_failures'] for r in rows),
            speed_min_mps=8, outgoing_vx_positive=True, outgoing_vz_max_mps=-1,
            rally_faults=list(FAULTS), original_recovery_and_event_order=True),
        comparisons=summaries, original_canonical_complete_matches=True,
        csv_sha256=sha(OUT / 'EPISODES.csv'), script_sha256=sha(Path(__file__)),
        formal_cases_read=False, formal_acceptance=False, original_runtime_configuration_modified=False,
        user_scope_choice_pending=True,
        provenance='The court config initially included the 2x2m research precision target in commit526d411; ACCEPTANCE.md describes G07 targets as research goals. No source was found identifying these exact bounds as a badminton rule or a user-specified numeric requirement.')
    (OUT / 'RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(complete={k:v['complete'] for k,v in summaries.items()},
                          unchanged=result['unchanged'], formal_acceptance=False)))


if __name__ == '__main__':
    main()
