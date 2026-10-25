"""Append a verifiable native TRAIN snapshot delta; never rewrite the first report."""
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


def main():
    pb = (RUN / 'progress.json').read_bytes()
    progress = json.loads(pb)
    raw = (RUN / 'episodes.partial.json').read_bytes()
    rows = json.loads(raw)
    assert pb == (RUN / 'progress.json').read_bytes(), 'Publication in flight; re-poll the same run'
    number = progress['update']
    out = HERE / f'SUPPLEMENT_UPDATE{number}.json'
    assert not out.exists(), 'Never overwrite a preserved supplement'
    original = json.loads((HERE / 'REPORT.json').read_bytes())
    base_path = HERE / 'all_closed_training_episodes.json.gz'
    assert digest(base_path.read_bytes()) == original['full_closed_snapshot_sha256']
    base = json.loads(gzip.decompress(base_path.read_bytes()))
    canonical = lambda value: json.dumps(value, separators=(',', ':'), sort_keys=True).encode()
    assert len(rows) > len(base) and canonical(rows[:len(base)]) == canonical(base)
    delta = rows[len(base):]
    rebuilt = base + delta
    assert (json.dumps(rebuilt, indent=2) + '\n').encode() == raw
    protocol = json.loads((ROOT / 'configs/g07_full_racket_eval_v4_r3.json').read_bytes())
    protocol['_root'] = str(ROOT)
    checked = [independent_episode(row, protocol) for row in rows]
    counts = dict(completed_episodes=len(rows), air=sum(v['airborne_contact'] for v in checked),
        conditional_recovery_count=sum(v['conditional_recovery'] for v in checked),
        complete=sum(v['complete'] for v in checked),
        numerical_failures=sum(row['numerical_failures'] for row in rows))
    assert all(progress[k] == value for k, value in counts.items())
    complete = [row for row, result in zip(rows, checked) if result['complete']]
    assert len(complete) >= 3
    inputs = json.loads((RUN / 'INPUTS.json').read_bytes())
    initial = inputs['initialization']['initial_iteration']
    rollout = inputs['runner']['num_steps_per_env']
    lineage = []
    required_models = {number}
    for row in complete:
        lane = sorted((v for v in rows if v['env_id'] == row['env_id'] and v['episode_index'] <= row['episode_index']),
            key=lambda value: value['episode_index'])
        assert [v['episode_index'] for v in lane] == list(range(row['episode_index'] + 1))
        durations = [round(v['terminal_native_state']['time_s'] / .01) for v in lane]
        assert all(abs(v['terminal_native_state']['time_s'] - ticks * .01) < 1e-5 for v, ticks in zip(lane, durations))
        start, end = sum(durations[:-1]), sum(durations)
        first_actor, last_actor = initial + start // rollout, initial + (end - 1) // rollout
        contact_actor = initial + int((start + row['contact_time_s'] / .01) // rollout)
        assert row['jump']['takeoff_time_s'] < row['contact_time_s'] < row['jump']['landing_time_s']
        assert row['contact_count'] == row['active_rally_launch_count'] == 1
        assert not row['frame_hit'] and not row['training_early_termination']
        assert row['terminal_native_state']['time_s'] == 8.
        lineage.append(dict(case_id=row['case_id'], source_TRAIN_case=row['source_training_index'],
            env_id=row['env_id'], episode_index=row['episode_index'], job_ticks=[start, end],
            actor_iterations_encountered=list(range(first_actor, last_actor + 1)),
            contact_rollout_actor_iteration=contact_actor))
        required_models.update(range(first_actor, last_actor + 2))
    old_checks = {v['iteration']: v for v in json.loads((HERE / 'weight_lineage.json').read_bytes())}
    model_checks = []
    for iteration in sorted(required_models):
        path = RUN / f'ppo/model_{iteration - 1}.pt'
        checksum = digest(path.read_bytes())
        if iteration in old_checks:
            assert old_checks[iteration]['sha256'] == checksum
        else:
            model = torch.load(path, map_location='cpu', weights_only=False)
            assert model['iter'] == iteration
            assert all(torch.isfinite(v).all() for v in model['model_state_dict'].values() if torch.is_tensor(v))
            assert all(torch.isfinite(v).all() for s in model['optimizer_state_dict']['state'].values()
                       for v in s.values() if torch.is_tensor(v))
        model_checks.append(dict(iteration=iteration, path=str(path.relative_to(ROOT)), sha256=checksum,
            model_optimizer_tensors_finite=True, canonical_native_resume=False))
    delta_name = f'additional_closed_training_episodes_update{number}.json.gz'
    complete_name = f'complete_episodes_update{number}.json'
    progress_name = f'progress_update{number}.json'
    assert not any((HERE / name).exists() for name in [delta_name, complete_name, progress_name])
    compressed = gzip.compress((json.dumps(delta, separators=(',', ':')) + '\n').encode(), mtime=0)
    (HERE / delta_name).write_bytes(compressed)
    assert json.loads(gzip.decompress(compressed))
    (HERE / complete_name).write_text(json.dumps(complete, indent=2) + '\n')
    (HERE / progress_name).write_bytes(pb)
    result = dict(created_at=datetime.now(timezone.utc).isoformat(), snapshot_update=number,
        scope='Additional changing-policy TRAIN completions; original complete gates independently recomputed. No frozen-policy or formal acceptance.',
        first_report='REPORT.json', first_report_sha256=digest((HERE / 'REPORT.json').read_bytes()),
        all_closed_training_counts=counts, complete_case_ids=[v['case_id'] for v in complete],
        distinct_completed_TRAIN_cases=sorted({v['source_training_index'] for v in complete}),
        original_prefix_episodes=len(base), additional_closed_episodes=len(delta),
        original_prefix_unchanged=True, exact_reconstructed_episode_bytes_match=True,
        reconstructed_original_partial_sha256=digest(raw), reconstructed_canonical_rows_sha256=digest(canonical(rows)),
        complete_episode_file=complete_name, delta_file=delta_name, actor_lineage=lineage,
        diagnostic_model_checks=model_checks,
        preserved_files={name: dict(bytes=(HERE / name).stat().st_size, sha256=digest((HERE / name).read_bytes()))
            for name in [delta_name, complete_name, progress_name]},
        native_contact_record_match_pending=True, full50Hz_training_motion_saved=False,
        limitations=['Actors, normalizers and exploration changed during these episodes. Do not attribute each success to a single frozen actor.',
            'The first7766-record snapshot is retained; append the decoded delta and serialize with json.dumps(indent=2)+newline to recover the exact9015-record partial snapshot when snapshot_update48.',
            'Match all completed env/episode pairs against contacts.json after the existing native guard actually ends.',
            'Later-episode ground starts follow the unchanged real reset path; their individual initial frames and full motion were not separately saved.',
            'Distinct successful TRAIN cases do not establish distribution-level rates or formal A/B acceptance.'],
        formal_cases_read=False, formal_acceptance=False, goal_complete=False)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(snapshot_update=number, counts=counts, complete_case_ids=result['complete_case_ids'],
        distinct_TRAIN_cases=result['distinct_completed_TRAIN_cases'], delta_bytes=len(compressed))))


if __name__ == '__main__':
    main()
