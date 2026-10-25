"""Append to the latest preserved real TRAIN prefix, without rewriting history."""
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RUN = ROOT / 'reports/G07_full_racket_v4/residual_ppo_n2048_resume64_01'
sys.path.insert(0, str(ROOT))
from quadsmash.evaluation_v4 import independent_episode


def digest(data):
    return hashlib.sha256(data).hexdigest()


def serialized(rows):
    return (json.dumps(rows, indent=2) + '\n').encode()


def load_prefix(path, ancestors=()):
    """Read both original-prefix supplements and chained incremental ones."""
    assert path.parent == HERE and path.name not in ancestors
    report = json.loads(path.read_bytes())
    if path.name == 'REPORT.json':
        blob = (HERE / 'all_closed_training_episodes.json.gz').read_bytes()
        assert digest(blob) == report['full_closed_snapshot_sha256']
        raw = gzip.decompress(blob)
        assert digest(raw) == report['full_closed_uncompressed_sha256']
        rows = json.loads(raw)
        assert len(rows) == report['all_closed_training_counts_at_snapshot']['completed_episodes']
        return rows
    prefix_name = report.get('prefix_report', 'REPORT.json')
    assert Path(prefix_name).name == prefix_name
    prefix_path = HERE / prefix_name
    if 'prefix_report_sha256' in report:
        assert digest(prefix_path.read_bytes()) == report['prefix_report_sha256']
    prefix = load_prefix(prefix_path, (*ancestors, path.name))
    name = report['delta_file']
    assert Path(name).name == name
    blob = (HERE / name).read_bytes()
    assert digest(blob) == report['preserved_files'][name]['sha256']
    delta = json.loads(gzip.decompress(blob))
    assert len(delta) == report['additional_closed_episodes']
    rows = prefix + delta
    assert len(rows) == report['all_closed_training_counts']['completed_episodes']
    assert digest(serialized(rows)) == report['reconstructed_original_partial_sha256']
    return rows


def main():
    progress_bytes = (RUN / 'progress.json').read_bytes()
    progress = json.loads(progress_bytes)
    raw = (RUN / 'episodes.partial.json').read_bytes()
    rows = json.loads(raw)
    assert progress_bytes == (RUN / 'progress.json').read_bytes(), 'Publication in flight; reobserve same job'
    number = progress['update']
    output = HERE / f'SUPPLEMENT_UPDATE{number}.json'
    assert not output.exists(), 'Never overwrite a preserved supplement'
    candidates = [p for p in HERE.glob('SUPPLEMENT_UPDATE*.json')
                  if json.loads(p.read_bytes())['snapshot_update'] < number]
    prefix_path = max(candidates, key=lambda p: json.loads(p.read_bytes())['snapshot_update'])
    prefix = load_prefix(prefix_path)
    assert len(rows) > len(prefix)
    assert digest(serialized(rows[:len(prefix)])) == digest(serialized(prefix))
    assert serialized(rows) == raw
    protocol = json.loads((ROOT / 'configs/g07_full_racket_eval_v4_r3.json').read_bytes())
    protocol['_root'] = str(ROOT)
    checked = [independent_episode(row, protocol) for row in rows]
    counts = dict(completed_episodes=len(rows), air=sum(c['airborne_contact'] for c in checked),
        conditional_recovery_count=sum(c['conditional_recovery'] for c in checked),
        complete=sum(c['complete'] for c in checked),
        numerical_failures=sum(row['numerical_failures'] for row in rows))
    assert all(progress[key] == value for key, value in counts.items())
    complete = [row for row, check in zip(rows, checked) if check['complete']]
    old_complete = json.loads((HERE / json.loads(prefix_path.read_bytes())['complete_episode_file']).read_bytes())
    indexed = {(r['env_id'], r['episode_index']): r for r in rows}
    assert len(indexed) == len(rows)
    assert all(json.dumps(indexed[(r['env_id'], r['episode_index'])], sort_keys=True)
        == json.dumps(r, sort_keys=True) for r in old_complete)
    initial = json.loads((RUN / 'INPUTS.json').read_bytes())['initialization']['initial_iteration']
    lineage, required_models = [], {number}
    for row in complete:
        lane = sorted((r for r in rows if r['env_id'] == row['env_id']
                       and r['episode_index'] <= row['episode_index']), key=lambda r: r['episode_index'])
        assert [r['episode_index'] for r in lane] == list(range(row['episode_index'] + 1))
        ticks = [round(r['terminal_native_state']['time_s'] / .01) for r in lane]
        assert all(abs(r['terminal_native_state']['time_s'] - count * .01) < 1e-5
                   for r, count in zip(lane, ticks))
        start, end = sum(ticks[:-1]), sum(ticks)
        first, last = initial + start // 128, initial + (end - 1) // 128
        assert row['active_rally_launch_count'] == 1 and not row['training_early_termination']
        assert row['terminal_native_state']['time_s'] == 8.
        lineage.append(dict(case_id=row['case_id'], source_TRAIN_case=row['source_training_index'],
            env_id=row['env_id'], episode_index=row['episode_index'], job_ticks=[start, end],
            actor_iterations_encountered=list(range(first, last + 1)),
            contact_rollout_actor_iteration=initial + int((start + row['contact_time_s'] / .01) // 128),
            first_string_rollout_actor_iteration=initial + int((start + row['first_string_time_s'] / .01) // 128)))
        required_models.update(range(first, last + 2))
    cached = {item['iteration']: item for item in json.loads((HERE / 'weight_lineage.json').read_bytes())}
    for path in candidates:
        for item in json.loads(path.read_bytes())['diagnostic_model_checks']:
            assert item['iteration'] not in cached or cached[item['iteration']]['sha256'] == item['sha256']
            cached[item['iteration']] = item
    model_checks = []
    for iteration in sorted(required_models):
        path = RUN / f'ppo/model_{iteration - 1}.pt'
        checksum = digest(path.read_bytes())
        if iteration in cached:
            assert checksum == cached[iteration]['sha256']
        else:
            model = torch.load(path, map_location='cpu', weights_only=False)
            assert model['iter'] == iteration
            assert all(torch.isfinite(v).all() for v in model['model_state_dict'].values() if torch.is_tensor(v))
            assert all(torch.isfinite(v).all() for state in model['optimizer_state_dict']['state'].values()
                       for v in state.values() if torch.is_tensor(v))
        model_checks.append(dict(iteration=iteration, path=str(path.relative_to(ROOT)), sha256=checksum,
            model_optimizer_tensors_finite=True, canonical_native_resume=False))
    delta_name = f'additional_closed_training_episodes_update{number}.json.gz'
    complete_name, progress_name = f'complete_episodes_update{number}.json', f'progress_update{number}.json'
    assert not any((HERE / name).exists() for name in [delta_name, complete_name, progress_name])
    delta = rows[len(prefix):]
    blob = gzip.compress((json.dumps(delta, separators=(',', ':')) + '\n').encode(), compresslevel=6, mtime=0)
    assert json.dumps(json.loads(gzip.decompress(blob)), separators=(',', ':')) == json.dumps(delta, separators=(',', ':'))
    (HERE / delta_name).write_bytes(blob)
    (HERE / complete_name).write_text(json.dumps(complete, indent=2) + '\n')
    (HERE / progress_name).write_bytes(progress_bytes)
    result = dict(created_at=datetime.now(timezone.utc).isoformat(), snapshot_update=number,
        scope='Independently rescored changing-policy native TRAIN snapshot, incrementally preserved. No frozen-policy performance or formal acceptance.',
        preservation_program_sha256=digest(Path(__file__).read_bytes()),
        prefix_report=prefix_path.name, prefix_report_sha256=digest(prefix_path.read_bytes()),
        preserved_prefix_episodes=len(prefix), additional_closed_episodes=len(delta),
        all_closed_training_counts=counts, complete_case_ids=[r['case_id'] for r in complete],
        distinct_completed_TRAIN_cases=sorted({r['source_training_index'] for r in complete}),
        complete_episode_file=complete_name, delta_file=delta_name, actor_lineage=lineage,
        diagnostic_model_checks=model_checks, original_prefix_unchanged=True,
        exact_reconstructed_episode_bytes_match=True, reconstructed_original_partial_sha256=digest(raw),
        preserved_files={name: dict(bytes=(HERE / name).stat().st_size,
            sha256=digest((HERE / name).read_bytes())) for name in [delta_name, complete_name, progress_name]},
        native_contact_record_match_pending=True, full50Hz_training_motion_saved=False,
        limitations=['Recursively reconstruct prefix_report, append the decoded delta and serialize json.dumps(indent=2)+newline to match the original native partial SHA.',
            'Actors, normalizers and exploration change within episodes; no completion belongs to a single frozen actor.',
            'Finalized native contacts remain pending. Later episode initial frames and full motion were not separately saved; the unchanged native ground-reset contract is not a reconstructed trajectory.',
            'This changes storage only; actual raw rows, original scoring, physical configuration and all thresholds are unchanged.'],
        formal_cases_read=False, formal_acceptance=False, goal_complete=False)
    temporary = output.with_suffix('.json.tmp')
    assert not temporary.exists()
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(output)
    print(json.dumps(dict(snapshot_update=number, counts=counts, prefix_report=prefix_path.name,
        prefix_episodes=len(prefix), additional_episodes=len(delta), delta_bytes=len(blob),
        complete_case_ids=result['complete_case_ids'])))


if __name__ == '__main__':
    main()
