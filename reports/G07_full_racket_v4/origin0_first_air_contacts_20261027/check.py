"""Independently inspect the completed first origin0 candidate with air strings."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import numpy as np
from quadsmash.control.full_racket_body_search_v4 import summarize_candidate
from quadsmash.control.full_racket_train_evaluation_v4 import training_cases
from quadsmash.evaluation_v4 import independent_episode, FAULTS


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not (HERE/'RESULT.json').exists()
    run = ROOT/'reports/G07_full_racket_v4/prepose_origin0_8_sequential_n128_01'
    protocol = read(ROOT/'configs/g07_full_racket_eval_v4_r3_origin0.json'); protocol['_root'] = str(ROOT)
    cases, indices = training_cases(ROOT, protocol)
    entry = next(x for x in read(run/'metrics.json')['history'] if x['candidate_id'] == 6)
    rows, initial = read(run/'episodes_006.json'), read(run/'initial_ground_states_006.json')
    summary = summarize_candidate(rows, protocol)
    assert summary == entry['summary'] and summary['airborne_contacts'] == 4 and summary['complete'] == summary['numerical_failures'] == 0
    assert len(rows) == 128 and initial['all_clocks_zero'] and not np.asarray(initial['actual_environment_origins_m']).any()
    for lane, row in enumerate(rows):
        assert row['env_id'] == row['jump']['env_id'] == row['source_training_index'] == indices[lane] == lane
        assert row['episode_index'] == initial['episode_ids'][lane] and row['case_id'] == cases[lane]['case_id']
        assert row['ground_init'] and row['candidate_id'] == 6
    contacts = read(run/'contacts_006.json'); selected = []
    for row in rows:
        outcome = independent_episode(row, protocol)
        if not outcome['airborne_contact']:
            continue
        actual = [event for event in contacts if event['env_id'] == row['env_id']
                  and event['episode_index'] == row['episode_index'] and event['string_contact']]
        assert actual and all(np.isfinite(event['point_m']).all() for event in actual)
        selected.append(dict(case_id=row['case_id'], env_id=row['env_id'], episode_index=row['episode_index'],
            contact_time_s=row['contact_time_s'], outgoing_velocity_mps=row['outgoing_velocity_mps'],
            actual_outgoing_speed_mps=float(np.linalg.norm(row['outgoing_velocity_mps'])),
            original_attack_speed_pass=bool(np.linalg.norm(row['outgoing_velocity_mps']) >= 8.),
            actual_string_contact_events=len(actual), crossed_net=row['crossed_net'],
            landed_legal_court=row['landed_legal_court'], landed_target=row['landed_target'],
            body_landing=row['jump']['body_landing'], stable_recovery=row['jump']['stable_recovery'],
            qualified_jump_landed=row['jump']['qualified_jump_landed'],
            actual_faults=[name for name in FAULTS if row[name]], independently_complete=outcome['complete']))
    assert [x['env_id'] for x in selected] == [69, 100, 112, 123]
    assert not any(x['original_attack_speed_pass'] or x['independently_complete'] for x in selected)
    paths = [run/name for name in ['episodes_006.json', 'initial_ground_states_006.json', 'contacts_006.json',
                                  'all_first_episodes_006.npz', 'public_and_actions_006.npz', 'INPUTS.json']]
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), actual_completed_candidate_id=6,
        candidate=entry['candidate'], original128_summary=summary, all128_case_and_native_episode_ids_checked=True,
        actual_air_contact_cases=selected, actual_crossed_and_legal_court=sum(x['landed_legal_court'] for x in selected),
        actual_target_first_landings=sum(x['landed_target'] for x in selected),
        body_landing_among_air_contacts=sum(x['body_landing'] for x in selected),
        source_sha256={str(path):sha(path) for path in paths+[Path(__file__)]},
        source_control_replay_pending_overall_owner_check=True,
        scope='Completed search candidate6 only. Four real legal air strings are not attacks or full successes. Whole search, selected policy cold128 and formal panels remain pending.',
        new_native_execution=False, formal_cases_read=False, formal_acceptance=False)
    (HERE/'RESULT.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(summary=summary, actual_air_contact_cases=selected)), flush=True)


if __name__ == '__main__':
    main()
