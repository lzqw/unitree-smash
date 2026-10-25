"""Verify a preserved native TRAIN prefix against its actual source rows."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(HERE))
from preserve_incremental_completions import digest, load_prefix, serialized
from quadsmash.evaluation_v4 import FAULTS, independent_episode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('update', type=int)
    number = parser.parse_args().update
    assert number == 96, 'Final snapshot verifier is reserved for the actual update96 data'
    check_path = HERE / f'INCREMENTAL_STORAGE_CHECKS_UPDATE{number}.json'
    note_path = HERE / f'CASE_NOTES_UPDATE{number}.json'
    assert not check_path.exists() and not note_path.exists(), 'Never overwrite evidence'
    report_path = HERE / f'SUPPLEMENT_UPDATE{number}.json'
    report = json.loads(report_path.read_bytes())
    assert report['snapshot_update'] == number
    assert digest((HERE / 'preserve_incremental_completions.py').read_bytes()) == report['preservation_program_sha256']
    rows = load_prefix(report_path)
    native_path = ROOT / 'reports/G07_full_racket_v4/residual_ppo_n2048_resume64_01/episodes.partial.json'
    native = json.loads(native_path.read_bytes())
    assert len(native) >= len(rows)
    preserved_sha = digest(serialized(rows))
    assert preserved_sha == report['reconstructed_original_partial_sha256']
    assert digest(serialized(native[:len(rows)])) == preserved_sha
    print(f'Exact preserved/native prefix match: {len(rows)} real rows', flush=True)
    protocol = json.loads((ROOT / 'configs/g07_full_racket_eval_v4_r3.json').read_bytes())
    protocol['_root'] = str(ROOT)
    checked = [independent_episode(row, protocol) for row in rows]
    counts = dict(completed_episodes=len(rows), air=sum(c['airborne_contact'] for c in checked),
        conditional_recovery_count=sum(c['conditional_recovery'] for c in checked),
        complete=sum(c['complete'] for c in checked),
        numerical_failures=sum(row['numerical_failures'] for row in rows))
    progress = json.loads((HERE / f'progress_update{number}.json').read_bytes())
    assert counts == report['all_closed_training_counts']
    assert all(progress[key] == value for key, value in counts.items())
    complete = [row for row, check in zip(rows, checked) if check['complete']]
    assert serialized(complete) == (HERE / report['complete_episode_file']).read_bytes()
    for name, metadata in report['preserved_files'].items():
        blob = (HERE / name).read_bytes()
        assert len(blob) == metadata['bytes'] and digest(blob) == metadata['sha256']
    prefix = json.loads((HERE / report['prefix_report']).read_bytes())
    old_complete = json.loads((HERE / prefix['complete_episode_file']).read_bytes())
    old_ids = {row['case_id'] for row in old_complete}
    indexed = {row['case_id']: row for row in complete}
    assert len(indexed) == len(complete)
    assert all(json.dumps(indexed[row['case_id']], sort_keys=True) == json.dumps(row, sort_keys=True)
               for row in old_complete)
    new = [row for row in complete if row['case_id'] not in old_ids]
    lineages = {item['case_id']: item for item in report['actor_lineage']}
    cases = []
    for row in new:
        velocity = row['outgoing_velocity_mps']
        cases.append(dict(native_case_id=row['case_id'], source_TRAIN_case=row['source_training_index'],
            first_string_contact_m=row['first_string_contact_point_m'],
            outgoing_velocity_mps=velocity, outgoing_speed_mps=sum(v*v for v in velocity)**.5,
            first_landing_m=row['first_landing_m'], recovery_s=row['jump']['recovery_s'],
            net_margin_m=row['net_margin_m'], crossing_point_m=row['crossing_point_m'],
            contact_time_s=row['contact_time_s'], release_time_s=row['release_time_s'],
            crossing_time_s=row['crossing_time_s'], first_landing_time_s=row['first_landing_time_s'],
            shuttle_net_contact=row['shuttle_net_contact'], frame_hit=row['frame_hit'],
            fault_flags={name: row[name] for name in FAULTS},
            body_landing=row['jump']['body_landing'],
            maximum_body_ground_force_n=row['jump']['maximum_body_ground_force_n'],
            ground_init=row['ground_init'], terminal_time_s=row['terminal_native_state']['time_s'],
            actor_lineage=lineages[row['case_id']],
            scope='Independently scored whole-chain native TRAIN completion under changing actors, normalizers and exploration; no frozen-policy performance.'))
    chain = []
    path = report_path
    while True:
        item = json.loads(path.read_bytes())
        chain.append(dict(report=path.name, sha256=digest(path.read_bytes())))
        if path.name == 'REPORT.json':
            break
        path = HERE / item.get('prefix_report', 'REPORT.json')
    observed_receipts = {}
    receipt_paths = [
        ROOT / 'reports/G07_full_racket_v4/residual_ppo_n2048_resume64_01/job.json',
        HERE / 'FINAL_CONTACT_CONFIRMATION.json',
        ROOT / 'reports/G07_full_racket_v4/residual_ppo_fixed_train128_iter96_n2048_01/job.json',
    ]
    for path in receipt_paths:
        blob = path.read_bytes() if path.is_file() else None
        observed_receipts[str(path.relative_to(ROOT))] = dict(
            observed_at=datetime.now(timezone.utc).isoformat(), exists=blob is not None,
            bytes=len(blob) if blob is not None else None,
            sha256=digest(blob) if blob is not None else None)
    notes = dict(created_at=datetime.now(timezone.utc).isoformat(), snapshot_update=number,
        source_report=report_path.name, source_report_sha256=digest(report_path.read_bytes()),
        all_closed_training_counts=counts, new_cases_since_prefix=cases,
        negative_margin_case_ids=[case['native_case_id'] for case in cases if case['net_margin_m'] < 0],
        net_margin_definition='At the trailing-surface crossing, the recorded center is x=net_plane_x+radius. net_margin=center_z-radius-strip_top. This diagnostic is not the minimum geometric clearance of the moving sphere from the finite net band.',
        crossing_rule='The unchanged rule requires the measured trailing crossing center above the actual strip top, within net width, after legal racket contact and without invalidating faults. Native solid-band collision governs response; shuttle/net touch itself is allowed.',
        clearance_limit='No full training trajectory was saved. A negative recorded margin alone neither measures minimum clearance nor proves net contact. Preserve its sign; do not describe it as positive clearance.',
        rule_source='quadsmash/physics/rally_rules_v4.py',
        rule_source_sha256=digest((ROOT / 'quadsmash/physics/rally_rules_v4.py').read_bytes()),
        native_contact_matching_performed_by_this_program=False, full_training_motion_saved=False,
        frozen_policy_assessment_performed_by_this_program=False,
        observed_receipts=observed_receipts,
        receipt_scope='File observations only. This storage verifier does not validate contact or fixed-policy receipts; use the separate native queue and delivery checks.',
        formal_acceptance=False, goal_complete=False)
    checks = dict(created_at=datetime.now(timezone.utc).isoformat(), snapshot_update=number,
        scope='CPU verification of actual preserved native TRAIN records and storage. No new simulation or frozen-policy acceptance.',
        source_report=report_path.name, source_report_sha256=digest(report_path.read_bytes()),
        verification_program_sha256=digest(Path(__file__).read_bytes()), prefix_chain_reports=chain,
        reconstructed_original_partial_sha256=preserved_sha, actual_native_prefix_rows=len(rows),
        native_rows_observed_at_recheck=len(native), independently_recomputed_counts=counts,
        compression_crc_and_manifest_sha_verified=True, exact_actual_native_prefix_reconstructed=True,
        preserved_complete_rows_exactly_match_native_prefix=True,
        previous_complete_rows_unchanged=True, snapshot_progress_counts_match=True,
        original_scoring_and_thresholds_unchanged=True, GPU_jobs_started=False,
        formal_acceptance=False, goal_complete=False)
    for path, value in [(note_path, notes), (check_path, checks)]:
        temporary = path.with_suffix('.json.tmp')
        assert not temporary.exists()
        temporary.write_text(json.dumps(value, indent=2) + '\n')
        temporary.replace(path)
    print(json.dumps(dict(update=number, counts=counts, new_cases=[case['native_case_id'] for case in cases],
        storage_checks=str(check_path.relative_to(ROOT)), case_notes=str(note_path.relative_to(ROOT)))))


if __name__ == '__main__':
    main()
