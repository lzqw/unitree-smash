"""Independently count the actual midpoint prefix; never score a frozen actor."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
REPORTS = ROOT / 'reports/G07_full_racket_v4'
RUN = REPORTS / 'residual_ppo_n2048_resume64_01'
sys.path.insert(0, str(ROOT))
from quadsmash.evaluation_v4 import independent_episode


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    assert not (HERE / 'REPORT.json').exists(), 'Preserve the first midpoint receipt'
    progress_bytes = (RUN / 'progress.json').read_bytes()
    progress = json.loads(progress_bytes)
    assert progress['update'] == 64 and progress['actual_control_env_steps'] == 8388608
    episode_bytes = (RUN / 'episodes.partial.json').read_bytes()
    rows = json.loads(episode_bytes)
    assert progress_bytes == (RUN / 'progress.json').read_bytes(), 'Publication changed; reobserve same job'
    assert (json.dumps(rows, indent=2) + '\n').encode() == episode_bytes
    protocol_path = ROOT / 'configs/g07_full_racket_eval_v4_r3.json'
    protocol = json.loads(protocol_path.read_bytes())
    protocol['_root'] = str(ROOT)
    checked = [independent_episode(row, protocol) for row in rows]
    counts = dict(completed_episodes=len(rows), air=sum(v['airborne_contact'] for v in checked),
        conditional_recovery_count=sum(v['conditional_recovery'] for v in checked),
        complete=sum(v['complete'] for v in checked),
        numerical_failures=sum(r['numerical_failures'] for r in rows))
    assert all(progress[key] == value for key, value in counts.items())
    assert len({(r['env_id'], r['episode_index']) for r in rows}) == len(rows)
    assert all(r['source_training_index'] == r['env_id'] % 128 for r in rows)
    cohorts = []
    for episode in sorted({r['episode_index'] for r in rows}):
        pairs = [(r, c) for r, c in zip(rows, checked) if r['episode_index'] == episode]
        complete_cohort = len(pairs) == 2048 and {r['env_id'] for r, _ in pairs} == set(range(2048))
        cohorts.append(dict(episode_index=episode, closed_episodes=len(pairs), all2048_lanes_closed=complete_cohort,
            air=sum(c['airborne_contact'] for _, c in pairs),
            conditional_recovery=sum(c['conditional_recovery'] for _, c in pairs),
            complete=sum(c['complete'] for _, c in pairs)))
    assert all(cohorts[i]['episode_index'] == i and cohorts[i]['all2048_lanes_closed'] for i in range(5))
    milestone = REPORTS / 'residual_n2048_first_complete_20261025'
    preserved = json.loads((milestone / 'complete_episodes_update63.json').read_bytes())
    actual = {(r['env_id'], r['episode_index']): r for r in rows}
    assert all(json.dumps(actual[(r['env_id'], r['episode_index'])], sort_keys=True)
               == json.dumps(r, sort_keys=True) for r in preserved)
    model_path = RUN / 'ppo/model_63.pt'
    model_bytes = model_path.read_bytes()
    model = torch.load(model_path, map_location='cpu', weights_only=False)
    assert model['iter'] == 64
    assert all(torch.isfinite(v).all() for v in model['model_state_dict'].values() if torch.is_tensor(v))
    assert all(torch.isfinite(v).all() for state in model['optimizer_state_dict']['state'].values()
               for v in state.values() if torch.is_tensor(v))
    native_spec = importlib.util.spec_from_file_location('midpoint_native_checks',
        REPORTS / 'residual_iter96_contact_queue01/run.py')
    q = importlib.util.module_from_spec(native_spec)
    native_spec.loader.exec_module(q)
    assert q.hashes() == q.core.read(q.STATE)['source_hashes']
    assert q.native.hashes() == q.core.read(q.NATIVE / 'state.json')['source_hashes']
    owners = [(1576853, 'residual_resize2048_queue01/run.py'), (1582969, str(RUN)), (1582980, str(RUN))]
    for name in ['residual_iter96_video_queue01', 'residual_iter96_train58_video_queue01',
                 'residual_iter96_formal_queue01', 'residual_iter96_contact_queue01']:
        state = q.core.read(REPORTS / name / 'state.json')
        owners.append((state['queue_pid'], name + '/run.py'))
        assert all(q.core.sha(path) == expected for path, expected in state['source_hashes'].items())
    assert all(q.core.live(pid, needle) for pid, needle in owners)
    q.core.atomic(HERE / 'progress_update64.json', progress)
    assert (HERE / 'progress_update64.json').read_bytes() == progress_bytes
    source_paths = [protocol_path, ROOT / 'quadsmash/evaluation_v4.py', ROOT / 'configs/court_net_v4.json',
        RUN / 'INPUTS.json', RUN / 'source_snapshot.zip', RUN / 'source_hashes.json',
        milestone / 'SUPPLEMENT_UPDATE63.json', Path(__file__).resolve()]
    result = dict(verified_at=datetime.now(timezone.utc).isoformat(),
        scope='Actual changing-policy native TRAIN midpoint. Exactly32of64 additional updates completed; canonical96/frozen assessment and all formal gates remain unmeasured.',
        update=64, additional_updates_completed=32, additional_updates_planned=64,
        current_job_control_env_steps=8388608,
        cumulative_learner_control_env_steps=q.native.PREVIOUS_CUMULATIVE_STEPS + 8388608,
        independently_recomputed_counts=counts, cohorts=cohorts,
        complete_case_ids=[r['case_id'] for r, c in zip(rows, checked) if c['complete']],
        raw_episode_prefix_length=len(rows), raw_episode_prefix_sha256=digest(episode_bytes),
        reproduction='Take the first raw_episode_prefix_length rows from this same run final episodes.json and serialize json.dumps(indent=2)+newline; verify its SHA before rescoring with the pinned evaluation source. Later incomplete cohorts are censored.',
        preserved_progress_sha256=digest(progress_bytes),
        diagnostic_model=str(model_path.relative_to(ROOT)), diagnostic_model_sha256=digest(model_bytes),
        model_optimizer_tensors_finite=True, diagnostic_model_is_canonical_restart=False,
        all_seven_original_source_owners_actually_live=True, source_fingerprints_unchanged=True,
        source_sha256={str(p.relative_to(ROOT)): q.core.sha(p) for p in source_paths},
        finalized_native_contact_matching_pending=True, full_training_motion_saved=False,
        formal_cases_read=False, formal_acceptance=False, goal_complete=False)
    q.core.atomic(HERE / 'REPORT.json', result)
    print(json.dumps(dict(update=64, counts=counts, complete_cohorts=[c['episode_index'] for c in cohorts
        if c['all2048_lanes_closed']], model_optimizer_tensors_finite=True, goal_complete=False)))


if __name__ == '__main__':
    main()
