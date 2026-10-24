#!/usr/bin/env python3
"""Freeze selected completed native TRAIN episodes for the video renderer."""
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
        generation, env_id = spec['generation'], spec['env_id']
        suffix = f'{generation:03}'
        episode_path = run / f'episodes_{suffix}.json'
        contact_path = run / f'contacts_{suffix}.json'
        pose_path = run / f'all_first_episodes_{suffix}.npz'
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
        assert selected['episode_clock_s'][-1] <= row['terminal_native_state']['time_s'] + 1e-5
        np.savez_compressed(output / f'case{index}.npz', **selected)
        destination = output / spec['run']
        destination.mkdir(exist_ok=True)
        required = ['geometry.json', 'geometry.npz', f'episodes_{suffix}.json',
                    f'contacts_{suffix}.json', f'population_{suffix}.pt',
                    'source_snapshot.zip', 'source_hashes.json', 'asset_dependencies.json',
                    'dirty.patch']
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
        cases.append(case)
    manifest['cases'] = cases
    manifest['original_source_hashes'][str(args.spec.resolve().relative_to(ROOT))] = sha(args.spec)
    (output / 'cases.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(snapshot=str(output.relative_to(ROOT)),
                          frames=[len(c['selected_source_frames']) for c in cases]), ensure_ascii=False))


if __name__ == '__main__':
    main()
