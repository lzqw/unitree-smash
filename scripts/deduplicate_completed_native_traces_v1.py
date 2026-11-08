"""Reclaim duplicate immutable native traces without removing any evidence path.

Only successfully completed native_* runs and manifest-verified first-episode
NPZ files qualify. Byte-equal files on the same filesystem are atomically
hardlinked. Never use this for mutable checkpoints or active run directories.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def equal_bytes(a, b):
    with a.open('rb') as x, b.open('rb') as y:
        while True:
            u, v = x.read(1024 * 1024), y.read(1024 * 1024)
            if u != v:
                return False
            if not u:
                return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve existing deduplication ledger')
    root = args.reports.resolve(); canonical = {}; rows = []; reclaimed = 0
    free_before = shutil.disk_usage(root).free
    for folder in sorted(root.glob('native_*')):
        job_path = folder / 'job.json'
        manifest_path = folder / 'fixed_policy_traces/manifest.json'
        if not job_path.is_file() or not manifest_path.is_file():
            continue
        job = json.loads(job_path.read_text())
        if job['status'] != 'SUCCEEDED' or job.get('exit_code') != 0 or not (folder/'completed.json').is_file():
            continue
        if any(Path('/proc') .joinpath(str(job.get(k, -1))).exists() for k in ['pid', 'child_pid']):
            continue
        manifest = json.loads(manifest_path.read_text())
        for row in manifest['episodes']:
            path = manifest_path.parent / row['data_file']
            if path.is_symlink() or path.resolve().parent != manifest_path.parent.resolve() or path.suffix != '.npz':
                raise ValueError('Unexpected evidence path ' + str(path))
            st = path.stat(); mode = stat.S_IMODE(st.st_mode)
            checksum = digest(path)
            if checksum != row['data_sha256']:
                raise ValueError('Evidence hash differs ' + str(path))
            key = (checksum, st.st_size, st.st_dev, mode)
            other = canonical.setdefault(key, path)
            if os.path.samefile(other, path):
                continue
            if not equal_bytes(other, path):
                raise ValueError('Equal-hash traces have unequal bytes')
            released = st.st_blocks * 512 if st.st_nlink == 1 else 0
            if args.apply:
                temp = path.with_name(path.name + '.dedup-link')
                os.link(other, temp)
                try:
                    if path.stat() != st or digest(other) != checksum:
                        raise ValueError('Immutable evidence changed during operation')
                    os.replace(temp, path)
                finally:
                    temp.unlink(missing_ok=True)
                if digest(path) != checksum:
                    raise ValueError('Post-link hash differs')
            rows.append(dict(path=str(path), canonical=str(other), sha256=checksum,
                content_bytes=st.st_size, estimated_reclaimed_bytes=released))
            reclaimed += released
            if reclaimed >= 3 * 1024**3:
                break
        if reclaimed >= 3 * 1024**3:
            break
    result = dict(scope=__doc__, applied=args.apply, paths_retained=True,
        source_manifest_hashes_unchanged=True, free_before_bytes=free_before,
        free_after_bytes=shutil.disk_usage(root).free, reclaimed_bytes=reclaimed, files=rows)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['files','scope']}))


if __name__ == '__main__':
    main()
