"""Describe actual matched TRAIN128 failures without changing training or gates."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from quadsmash.evaluation_v4 import independent_episode


def main():
    protocol = json.loads((ROOT / 'configs/g07_full_racket_eval_v4_r3.json').read_text())
    protocol['_root'] = str(ROOT)
    reports = ROOT / 'reports/G07_full_racket_v4'
    names = ['residual_ppo_centered_base8_n2048_iter40_fixed_train128_01',
             'residual_ppo_centered_placement075_n2048_probe8_fixed_train128_01']
    data = []
    sources = {}
    for name in names:
        path = reports / name / 'episodes.json'
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows = json.loads(path.read_text())
        assert len(rows) == 128 and [v['env_id'] for v in rows] == list(range(128))
        checked = []
        for row in rows:
            o = independent_episode(row, protocol)
            v = np.asarray(row['outgoing_velocity_mps'])
            first = np.asarray(row['first_landing_m'])
            attack = bool(o['airborne_contact'] and row['released'] and np.isfinite(v).all()
                          and np.linalg.norm(v) >= 8 and v[0] > 0 and v[2] <= -1)
            checked.append(dict(TRAIN=row['env_id'], **o, actual_racket_contact=row['had_contact'],
                frame_hit=row['frame_hit'], attack=attack, crossed_net=row['crossed_net'],
                landed_target=row['landed_target'], body_landing=row['jump']['body_landing'],
                speed_mps=float(np.linalg.norm(v)) if np.isfinite(v).all() else None,
                outgoing_velocity_mps=v.tolist() if np.isfinite(v).all() else None,
                first_landing_m=first.tolist() if np.isfinite(first).all() else None))
        data.append(checked)
    before, after = data
    paired = {}
    for key in ['airborne_contact', 'conditional_recovery', 'attack', 'complete']:
        paired[key] = dict(gained=[a['TRAIN'] for b, a in zip(before, after) if a[key] and not b[key]],
                           lost=[a['TRAIN'] for b, a in zip(before, after) if b[key] and not a[key]],
                           retained=[a['TRAIN'] for b, a in zip(before, after) if b[key] and a[key]])
    attacks = [v for v in after if v['attack']]
    assert len(attacks) == 5 and all(v['crossed_net'] and not v['fault'] for v in attacks)
    assert all(v['first_landing_m'][0] > 4 for v in attacks)
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), source_sha256=sources,
        matched_case_scope='Same originalTRAIN128, zero reset noise and seed724201. Training seed, critic initialization and reward differ; this does not isolate a causal reward effect.',
        paired_changes=paired, all_current_airborne_attacks=attacks,
        current_airborne_contact_cases_with_target=[v['TRAIN'] for v in after if v['airborne_contact'] and v['landed_target']],
        current_airborne_contact_cases_without_recovery=[v['TRAIN'] for v in after if v['airborne_contact'] and not v['conditional_recovery']],
        current_all128=after,
        implication='All five qualifying airborne attacks cross legally but first land beyond the original4m target boundary; three recover. Recovery improved, while both contact coverage and target placement remain unresolved. Wait for the already running strict continuation frozen assessment before selecting another method.',
        formal_cases_read=False, formal_acceptance=False, goal_complete=False)
    output = HERE / 'PAIRED_FROZEN_FAILURE_DIAGNOSTIC.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(paired_changes=paired, attacks=attacks, air_and_target=result['current_airborne_contact_cases_with_target']), ensure_ascii=False))


if __name__ == '__main__':
    main()
