"""Inspect the selected policy's real independent128, without inheriting search scores."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from analyze_components import ROOT, read, sha, independent_episode

HERE = Path(__file__).resolve().parent
COLD = HERE.parent/'windowed_swing7_best_cold128_01'


def main():
    proof = read(HERE/'ACTUAL_INDEPENDENT128_CHECK.json')
    assert all(sha(Path(p)) == h for p, h in proof['source_sha256'].items())
    protocol_path = ROOT/'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json'
    protocol = read(protocol_path)
    protocol['_root'] = str(ROOT)
    rows_path = COLD/'episodes.json'
    rows = read(rows_path)
    assert len(rows) == 128
    checked = [independent_episode(r, protocol) for r in rows]
    air = [r for r, c in zip(rows, checked) if c['airborne_contact']]
    speed = [math.sqrt(sum(v*v for v in r['outgoing_velocity_mps'])) for r in air]
    assert len(air) == proof['independent_actual128_summary']['airborne_contacts']
    assert all(math.isfinite(v) for v in speed)
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), all_original_cases=128,
        selected_candidate_id=proof['selected_candidate_id'], actual_air_count=len(air),
        actual_air_TRAIN_indices=[r['source_training_index'] for r in air],
        actual_outgoing_ball_speed_range_m_s=[min(speed), max(speed)],
        speed_at_least_original8m_s_count=sum(v >= 8. for v in speed),
        air_first_target_landings=sum(r['landed_target'] for r in air),
        air_aggregate_nonfoot_ground_flags=sum(r['contact_flight_recovery']['body_landing'] for r in air),
        actual_independent_complete_count=sum(c['complete'] for c in checked),
        actual_independent_recovery_count=sum(c['conditional_recovery'] for c in checked),
        measured_outgoing_ball_velocity_not_preimpact_racket_velocity=True,
        numerical_failure_count=proof['independent_actual128_summary']['numerical_failures'],
        new_native_execution=False, formal_cases_read=False, formal_acceptance=False,
        source_sha256={str(p): sha(p) for p in [Path(__file__), HERE/'analyze_components.py',
            HERE/'ACTUAL_INDEPENDENT128_CHECK.json', rows_path, protocol_path, ROOT/'quadsmash/evaluation_v4.py']})
    with (HERE/'INDEPENDENT128_FAILURE_COMPONENT_CHECK.json').open('x') as f:
        f.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'source_sha256'}))


if __name__ == '__main__':
    main()
