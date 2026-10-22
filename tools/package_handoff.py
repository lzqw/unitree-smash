#!/usr/bin/env python3
"""Package existing evidence and selected runtime files without launching Isaac.

No credentials or entire vendor checkouts are included. Existing binary files
are copied byte-for-byte; absolute USD references are documented, not rewritten.
"""
import argparse
import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def files_under(path):
    return sorted(p for p in path.rglob('*') if p.is_file() and not p.is_symlink()
                  and not {'.git', '__pycache__', '.pytest_cache'}.intersection(p.parts))


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--go2-root', type=Path, default=Path('/home/autodrive/Isaac/unitree_model/Go2'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / 'handoff_manifest.json'
    if manifest_path.exists():
        parser.error('use a new output directory to preserve the previous handoff')
    evidence = [(p, str(p.relative_to(ROOT))) for p in files_under(ROOT / 'reports')]
    records = [(p, str(p.relative_to(ROOT))) for p in files_under(ROOT / 'runs') if p.suffix != '.pt']
    records += [(p, str(p.relative_to(ROOT))) for p in files_under(ROOT / 'artifacts')]
    records += [(p, str(p.relative_to(ROOT))) for p in (ROOT / 'runs/G04').glob('corrected_*/native/checkpoint.pt')]
    runtime = [(p, str(p.relative_to(ROOT))) for p in files_under(ROOT / 'assets') if 'vendor' not in p.relative_to(ROOT).parts]
    source_manifest = json.loads((ROOT / 'assets/standard_piper_manifest.json').read_text())
    source_paths = [source_manifest['source'], *source_manifest['mesh_hashes']]
    runtime += [(ROOT / name, name) for name in source_paths]
    vendor = ROOT / 'assets/vendor/piper_isaac_sim'
    runtime += [(p, str(p.relative_to(ROOT))) for p in vendor.iterdir()
                if p.is_file() and (p.name.lower().startswith(('readme', 'license', 'copying')))]
    go2 = files_under(args.go2_root)
    if not go2 or not (args.go2_root / 'usd/go2.usd').is_file():
        parser.error('the measured Go2 runtime assets are required')
    runtime += [(p, 'external/Go2/' + str(p.relative_to(args.go2_root))) for p in go2]
    bundles = {'evidence.tar.gz': evidence,
               'checkpoints-and-training-records.tar.gz': records,
               'runtime-assets.tar.gz': runtime}
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(),
        repository='lzqw/unitree-smash', release_tag='handoff-20261022',
        purpose='Current failed G04 research handoff; no claim of trained task success',
        absolute_reference_mapping={'external/Go2': str(args.go2_root),
            'project_root_in_original_USD_and_logs': str(ROOT)},
        exclusions=['per-update model_*.pt in runs (approximately 18 GB locally)',
            'Python/Isaac installation, caches, credentials and complete vendor Git repositories'],
        bundles={})
    for name, entries in bundles.items():
        destination = args.output / name
        if destination.exists():
            parser.error(f'archive already exists: {destination}')
        unique = {archive_name: source for source, archive_name in entries}
        file_records = []
        with tarfile.open(destination, 'w:gz', compresslevel=4) as archive:
            for archive_name, source in sorted(unique.items()):
                if source.is_symlink() or not source.is_file():
                    raise ValueError(f'expected a regular file: {source}')
                archive.add(source, arcname=archive_name, recursive=False)
                file_records.append(dict(path=archive_name, bytes=source.stat().st_size, sha256=sha256(source)))
        manifest['bundles'][name] = dict(bytes=destination.stat().st_size, sha256=sha256(destination), files=file_records)
        print(json.dumps(dict(archive=name, files=len(file_records), bytes=destination.stat().st_size)), flush=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    checksums = [f'{item["sha256"]}  {name}' for name, item in manifest['bundles'].items()]
    checksums.append(f'{sha256(manifest_path)}  {manifest_path.name}')
    (args.output / 'SHA256SUMS').write_text('\n'.join(checksums) + '\n')


if __name__ == '__main__':
    main()
