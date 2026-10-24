#!/usr/bin/env python3
"""Compare TRAIN selection rules on unchanged completed native episode records."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quadsmash.control.full_racket_contact_curriculum_v4 import contact_search_tiers, usable_air_contact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a fresh output directory')
    comparisons, hashes = [], {}
    for run in args.runs:
        metrics_path = run / 'metrics.json'
        metrics = json.loads(metrics_path.read_text())
        hashes[str(metrics_path)] = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
        for entry in metrics['history']:
            path = run / f"episodes_{entry['generation']:03d}.json"
            rows = json.loads(path.read_text())
            assert len(rows) == entry['episodes'] and all(r['episode_index'] == 0 for r in rows)
            ids = sorted({r['candidate_id'] for r in rows})
            groups = [[r for r in rows if r['candidate_id'] == i] for i in ids]
            old = contact_search_tiers(groups)
            new = contact_search_tiers(groups, retain_recovery=True)
            scores = np.array([.7 * min(r['training_score'] for r in g)
                               + .3 * np.mean([r['training_score'] for r in g]) for g in groups])
            assert np.isfinite(scores).all()
            if 'contact_search_tiers' in entry:
                np.testing.assert_array_equal(old, entry['contact_search_tiers'])
            choices = {}
            for mode, tiers in [('historical', old), ('retain_same_episode_recovery', new)]:
                index = max(range(len(groups)), key=lambda i: (int(tiers[i]), float(scores[i])))
                group = groups[index]
                choices[mode] = dict(candidate_id=ids[index], tier=int(tiers[index]),
                    score=float(scores[index]), usable_air_contacts=sum(usable_air_contact(r) for r in group),
                    same_episode_air_contact_recovery=sum(usable_air_contact(r) and r['recovery_success'] for r in group),
                    complete_smashes=sum(r['complete_smash'] for r in group),
                    cases=[{k: r[k] for k in ['env_id', 'source_training_index', 'airborne_string_hit',
                        'recovery_success', 'outgoing_velocity_mps', 'crossed_net', 'landed_target', 'complete_smash']}
                        for r in group])
            comparisons.append(dict(run=str(run), generation=entry['generation'], native_episodes=len(rows),
                choices=choices, available_same_episode_contact_recovery=sum(usable_air_contact(r) and r['recovery_success'] for r in rows)))
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in [Path(__file__), ROOT / 'quadsmash/control/full_racket_contact_curriculum_v4.py']:
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = dict(created_at=datetime.now(timezone.utc).isoformat(),
        scope='Alternative TRAIN ranking of unchanged historical completed native records; no new dynamics, no future outcomes, no formal panels. Native labels are never changed.',
        comparisons=comparisons, original_flags_and_data_preserved=True,
        changed_ranking_native_execution=False, formal_acceptance=False, sources_sha256=hashes)
    args.output.mkdir(parents=True)
    (args.output / 'REPORT.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps([dict(run=Path(r['run']).name, generation=r['generation'],
        old_candidate=r['choices']['historical']['candidate_id'],
        new_candidate=r['choices']['retain_same_episode_recovery']['candidate_id'],
        old_joined=r['choices']['historical']['same_episode_air_contact_recovery'],
        new_joined=r['choices']['retain_same_episode_recovery']['same_episode_air_contact_recovery'])
        for r in comparisons]), flush=True)


if __name__ == '__main__':
    main()
