"""Apply the preserved additional-complete audit to the actual fourth trace."""
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
REPORTS = ROOT/'reports/G07_full_racket_v4'
SNAPSHOT = REPORTS/'video_snapshot_placement075_fourth_complete_20261026'
AUDITOR = REPORTS/'placement075_additional_complete_training_20261026/audit.py'


def main():
    output = HERE/'FOURTH_COMPLETE_NATIVE_EPISODE_CHECK.json'
    assert not output.exists()
    spec = importlib.util.spec_from_file_location('preserved075_complete_trace_audit', AUDITOR)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    audit.SNAPSHOT = SNAPSHOT  # Data input only; the preserved audit code is unchanged.
    manifest = audit.read(SNAPSHOT/'cases.json')
    assert not manifest['source_job_complete'] and manifest['includes_failure']
    assert [(c['env_id'], c['episode_index']) for c in manifest['cases']] == [(1466, 6), (23, 2), (8, 3)]
    directory = SNAPSHOT/manifest['cases'][0]['run']
    raw_path, trace_path = directory/'observed_episodes.json', directory/'observed_trace_manifest.json'
    assert audit.sha(raw_path) == manifest['observed_raw_publication_sha256']
    assert audit.sha(trace_path) == manifest['observed_trace_manifest_sha256']
    rows, recorded = audit.read(raw_path), audit.read(trace_path)
    indexed = {(r['env_id'], r['episode_index']): r for r in rows}
    assert len(indexed) == len(rows)
    protocol_path = ROOT/'configs/g07_full_racket_eval_v4_r3.json'
    protocol = audit.read(protocol_path)
    protocol['_root'] = str(ROOT)
    checked, paths = audit.audit_case(manifest['cases'][0], protocol, indexed, recorded)
    assert checked['TRAIN'] == 58 and checked['actor_iterations_encountered'] == list(range(19, 26))
    outcomes = [audit.independent_episode(r, protocol) for r in rows]
    counts = dict(closed_episodes=len(rows), air=sum(o['airborne_contact'] for o in outcomes),
        conditional_recovery=sum(o['conditional_recovery'] for o in outcomes),
        complete=sum(o['complete'] for o in outcomes), numerical_failures=sum(r['numerical_failures'] for r in rows))
    assert counts['complete'] >= 4 and counts['numerical_failures'] == 0
    assert all(not audit.independent_episode(c['row'], protocol)['complete'] for c in manifest['cases'][1:])
    dependencies = [Path(__file__), AUDITOR, SNAPSHOT/'cases.json', raw_path, trace_path,
        protocol_path, directory/'geometry.json', ROOT/'quadsmash/evaluation_v4.py',
        ROOT/'quadsmash/tasks/stand/observation_contract.py', *paths]
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), actual_fourth_complete_episode=checked,
        all_observed_closed_changing_policy_counts=counts, retained_failure_keys=[[23, 2], [8, 3]],
        scope='Actual fourth complete ground-start TRAIN episode of075 strict continuation, using the unchanged preserved100Hz auditor. Actor19..25, not a frozen-policy result.',
        source_job_complete_at_capture=False,
        source_sha256={str(p.relative_to(ROOT)): audit.sha(p) for p in dependencies},
        formal_cases_read=False, new_native_execution=False, frozen_policy_performance=False,
        formal_acceptance=False, goal_complete=False)
    for path in paths:
        destination = HERE/path.name
        assert not destination.exists()
        shutil.copy2(path, destination)
        assert audit.sha(destination) == audit.sha(path)
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: checked[key] for key in ['TRAIN', 'env_id', 'episode_index',
        'actual_outgoing_speed_mps', 'actual_first_landing_m', 'actual_target_far_x_margin_m',
        'actual_recovery_s', 'actual_point_speed_mps']}, indent=2))


if __name__ == '__main__':
    main()
