"""Summarize actual outgoing-ball and recovery failures in all7 sealed trials."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2].parent/'unitree_smash_contact_recovery'
RUN = HERE.parent/'windowed_swing7_n128_01'
sys.path.insert(0, str(ROOT))
from quadsmash.evaluation_v4 import FAULTS, independent_episode


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    check = read(HERE/'ACTUAL_COMPLETED_SEARCH_CHECK.json')
    assert all(sha(Path(p)) == h for p, h in check['source_sha256'].items())
    protocol_path = ROOT/'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json'
    protocol = read(protocol_path)
    protocol['_root'] = str(ROOT)
    files = [Path(__file__), HERE/'ACTUAL_COMPLETED_SEARCH_CHECK.json', protocol_path,
             ROOT/'quadsmash/evaluation_v4.py']
    results = []
    for entry in check['results']:
        path = RUN/f"episodes_{entry['candidate_id']:03d}.json"
        rows = read(path)
        assert len(rows) == 128
        air = [r for r in rows if independent_episode(r, protocol)['airborne_contact']]
        assert len(air) == entry['summary']['airborne_contacts'] and air
        speed = [math.sqrt(sum(v*v for v in r['outgoing_velocity_mps'])) for r in air]
        assert all(math.isfinite(v) for v in speed)
        attacks = [r for r, value in zip(air, speed) if r['released'] and value >= 8.
                   and r['outgoing_velocity_mps'][0] > 0. and r['outgoing_velocity_mps'][2] <= -1.]
        assert len(attacks) == entry['summary']['attack_outgoing']
        results.append(dict(candidate_id=entry['candidate_id'], all_original_cases=128,
            air_TRAIN_indices=[r['source_training_index'] for r in air], air_count=len(air),
            actual_outgoing_ball_speed_range_m_s=[min(speed), max(speed)],
            speed_at_least_original8m_s_count=sum(v >= 8. for v in speed), original_attack_count=len(attacks),
            air_first_target_landings=sum(r['landed_target'] for r in air),
            air_aggregate_nonfoot_ground_flags=sum(r['contact_flight_recovery']['body_landing'] for r in air),
            air_qualified_recovery_count=sum(independent_episode(r, protocol)['conditional_recovery'] for r in air),
            all128_fault_counts={name: sum(r[name] for r in rows) for name in FAULTS}))
        files.append(path)
    base = set(results[0]['air_TRAIN_indices'])
    selected = next(r for r in results if r['candidate_id'] == check['selected_candidate_id'])
    record = dict(checked_at=datetime.now(timezone.utc).isoformat(), results=results,
        selected_candidate_id=check['selected_candidate_id'], selected_components=selected,
        selected_gained_air_TRAIN_indices=sorted(set(selected['air_TRAIN_indices'])-base),
        selected_lost_air_TRAIN_indices=sorted(base-set(selected['air_TRAIN_indices'])),
        aggregate_contact_subset_count=sum(r['air_count'] for r in results),
        maximum_actual_air_outgoing_ball_speed_m_s=max(r['actual_outgoing_ball_speed_range_m_s'][1] for r in results),
        contact_subset_counts_are_repeated_executions_not_distinct_cases=True,
        nonfoot_flag_not_attributed_to_a_specific_body_part=True,
        point_velocity_mps_not_used_as_preimpact_velocity=True,
        independent_selected_cold128_still_separate=True, native_scores_changed=False,
        new_native_execution=False, formal_cases_read=False, formal_acceptance=False,
        source_sha256={str(p): sha(p) for p in files})
    with (HERE/'ACTUAL_FAILURE_COMPONENT_CHECK.json').open('x') as f:
        f.write(json.dumps(record, indent=2)+'\n')
    print(json.dumps(dict(selected_components=selected,
        total_repeated_air_events=record['aggregate_contact_subset_count'],
        maximum_actual_air_outgoing_ball_speed_m_s=record['maximum_actual_air_outgoing_ball_speed_m_s'],
        gained=record['selected_gained_air_TRAIN_indices'], lost=record['selected_lost_air_TRAIN_indices'])))


if __name__ == '__main__':
    main()
