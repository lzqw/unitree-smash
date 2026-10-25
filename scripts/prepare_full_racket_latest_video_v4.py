#!/usr/bin/env python3
"""Freeze selected completed native training or evaluation episodes for video."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/G07_full_racket_v4'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.spec.read_text())
    manifest.update(created_at=datetime.now(timezone.utc).isoformat(),
                    formal_acceptance=False, original_source_hashes={})
    cases = []
    for index, spec in enumerate(manifest['cases']):
        run = (REPORTS / spec['run']).resolve()
        assert run.parent == REPORTS.resolve(), 'Expected a native run in the report directory'
        evaluation=spec.get('source_type')=='evaluation'
        generation, env_id = spec.get('generation'), spec['env_id']
        suffix = '' if evaluation else f'_{generation:03}'
        episode_path = run / f'episodes{suffix}.json'
        contact_path = run / f'contacts{suffix}.json'
        pose_path = run / f'all_first_episodes{suffix}.npz'
        rows = json.loads(episode_path.read_text())
        matching = [row for row in rows if row['env_id'] == env_id and row['episode_index'] == 0]
        assert len(matching) == 1, 'Only a completed, independent first episode is supported'
        row = matching[0]
        contacts = [c for c in json.loads(contact_path.read_text())
                    if c['env_id'] == env_id and c['episode_index'] == 0]
        event = next((c for c in contacts if c['string_contact']), contacts[0] if contacts else None)
        with np.load(pose_path) as data:
            valid = data['valid_first_episode'][:, env_id]
            ids = np.flatnonzero(valid)
            assert len(ids) and np.array_equal(ids, np.arange(len(ids)))
            selected = {key: data[key][ids, env_id].copy() for key in data.files}
        assert selected['valid_first_episode'].all()
        assert np.isfinite(selected['link_poses']).all()
        assert np.allclose(np.diff(selected['episode_clock_s']), .02, atol=2e-6)
        if not evaluation:
            assert selected['episode_clock_s'][-1] <= row['terminal_native_state']['time_s'] + 1e-5
        np.savez_compressed(output / f'case{index}.npz', **selected)
        destination = output / spec['run']
        destination.mkdir(exist_ok=True)
        required = ['geometry.json', 'geometry.npz', f'episodes{suffix}.json',
                    f'contacts{suffix}.json',
                    'source_snapshot.zip', 'source_hashes.json', 'asset_dependencies.json',
                    'dirty.patch']
        required += ['metrics.json','initial_states.json'] if evaluation else [f'population{suffix}.pt']
        for name in required:
            source = run / name
            assert source.is_file(), source
            target = destination / name
            if not target.exists():
                shutil.copy2(source, target)
            assert sha(target) == sha(source)
            manifest['original_source_hashes'][str(source.relative_to(ROOT))] = sha(source)
        manifest['original_source_hashes'][str(pose_path.relative_to(ROOT))] = sha(pose_path)
        if (run / 'job.json').exists():
            shutil.copy2(run / 'job.json', destination / 'job.json')
        case = dict(spec, id=index, episode_index=0, row=row, contact=event,
                    selected_source_frames=ids.tolist(),
                    whole_source_job_complete=(run / 'completed.json').is_file())
        if evaluation:
            # Evaluator rows have no exact terminal pose. Preserve the row
            # and explicitly identify the last 50Hz native sample instead.
            case['last_recorded_state']=dict(time_s=float(selected['episode_clock_s'][-1]),
                body_link_positions_m=selected['link_poses'][-1,:,:3].tolist(),
                body_link_quaternions_wxyz=selected['link_poses'][-1,:,3:].tolist(),
                ball_position_m=selected['ball_position_m'][-1].tolist(),
                scope='Last valid native 50Hz sample; not exact terminal substep.')
        cases.append(case)
    manifest['cases'] = cases
    manifest['original_source_hashes'][str(args.spec.resolve().relative_to(ROOT))] = sha(args.spec)
    (output / 'cases.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(snapshot=str(output.relative_to(ROOT)),
                          frames=[len(c['selected_source_frames']) for c in cases]), ensure_ascii=False))


if __name__ == '__main__':
    main()
