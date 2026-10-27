"""Restore packaged original paths below a fresh destination, without overwrites."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import zipfile


def mapped(root, name):
    path = PurePosixPath(name)
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('Require an original absolute path without parent traversal')
    return root.joinpath(*path.parts[1:])


def restore(archive_path, destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError('Use a new destination; never overwrite a live project')
    destination.mkdir(parents=True)
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        manifest = json.loads(archive.read('MANIFEST.json'))
        assert manifest['schema'] == 'g07-original-path-input-bundle-v1'
        for record in manifest['files']:
            if record['link_target_original'] is not None:
                continue
            target = mapped(destination, record['original_path'])
            payload = archive.read('objects/' + record['sha256'])
            assert len(payload) == record['bytes']
            assert hashlib.sha256(payload).hexdigest() == record['sha256']
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(payload)
        for record in manifest['files']:
            if record['link_target_original'] is None:
                continue
            target = mapped(destination, record['original_path'])
            canonical = mapped(destination, record['link_target_original'])
            assert canonical.is_file()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(os.path.relpath(canonical, target.parent))
        for record in manifest['files']:
            target = mapped(destination, record['original_path'])
            assert target.is_file() and target.stat().st_size == record['bytes']
            assert hashlib.sha256(target.read_bytes()).hexdigest() == record['sha256']
    return dict(restored_original_paths=len(manifest['files']),
                original_held_input_paths=manifest['held_input_count'],
                unique_payloads=manifest['unique_payload_count'], all_file_hashes_verified=True,
                original_link_resolution_preserved_within_destination=True,
                destination=str(destination), native_execution=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(restore(args.archive, args.destination)))
