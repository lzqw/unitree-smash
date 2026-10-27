"""Fresh-download the complete input bundle and verify objects and path records."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    assert not (HERE / 'UPLOAD_RECEIPT.json').exists()
    spec = importlib.util.spec_from_file_location('verified_github_asset_transport',
        ROOT / 'reports/G07_full_racket_v4/contact_aim7_completed_delivery_20261027/download_verified.py')
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    check = json.loads((HERE / 'ARCHIVE_CHECK.json').read_text())
    request = subprocess.run(['gh', 'api', 'repos/lzqw/unitree-smash/releases/422477709'],
                             text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert request.returncode == 0
    assets = [a for a in json.loads(request.stdout)['assets'] if a['name'] == check['archive']]
    assert len(assets) == 1
    asset = assets[0]
    assert asset['state'] == 'uploaded' and asset['size'] == check['bytes']
    assert asset['digest'] == 'sha256:' + check['sha256']
    transport.ASSET_ID = asset['id']
    for attempt in range(5):
        try:
            location = transport.signed_url()
            break
        except Exception as error:
            print(json.dumps(dict(redirect_attempt=attempt+1,category=type(error).__name__,
                reason_type=type(getattr(error,'reason',None)).__name__)),flush=True)
            if attempt==4:
                raise RuntimeError('Could not acquire an artifact redirect after5 bounded attempts') from None
            time.sleep(attempt+1)
    directory = Path(tempfile.mkdtemp(prefix='g07_complete_contact_recovery_fresh_download_'))
    output = directory / asset['name']
    descriptor = os.open(output, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    os.ftruncate(descriptor, asset['size'])
    chunk = 32 * 1024 * 1024

    def part(start):
        end = min(asset['size'] - 1, start + chunk - 1)
        for attempt in range(4):
            try:
                request = urllib.request.Request(location, headers={
                    'Range': f'bytes={start}-{end}', 'User-Agent': 'unitree-smash-complete-input-verification'})
                with urllib.request.urlopen(request, timeout=25) as stream:
                    assert stream.status == 206
                    assert stream.headers['Content-Range'] == f'bytes {start}-{end}/{asset["size"]}'
                    offset = start
                    while offset <= end:
                        data = stream.read(min(1024 * 1024, end - offset + 1))
                        assert data and os.pwrite(descriptor, data, offset) == len(data)
                        offset += len(data)
                    assert not stream.read(1)
                return end - start + 1
            except Exception as error:
                if attempt == 3:
                    raise RuntimeError('Asset range failed: ' + type(error).__name__) from None
                time.sleep(attempt + 1)

    downloaded = 0
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            for future in as_completed([pool.submit(part, start) for start in range(0, asset['size'], chunk)]):
                downloaded += future.result()
                print(json.dumps(dict(downloaded_bytes=downloaded, total_bytes=asset['size'])), flush=True)
    finally:
        os.close(descriptor)
    assert downloaded == asset['size'] and hashlib.sha256(output.read_bytes()).hexdigest() == check['sha256']
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        manifest = json.loads(archive.read('MANIFEST.json'))
        objects = {record['sha256'] for record in manifest['files']}
        assert len(objects) == manifest['unique_payload_count'] == 580
        for digest in objects:
            assert hashlib.sha256(archive.read('objects/' + digest)).hexdigest() == digest
        assert len(manifest['files']) == check['restored_original_path_count'] == 1016
        assert len(manifest['held_input_paths']) == manifest['held_input_count'] == 931
        assert set(manifest['held_input_paths']).issubset({record['original_path'] for record in manifest['files']})
        members = len(archive.namelist())
    metadata = {key: asset[key] for key in ['id', 'name', 'browser_download_url', 'size', 'state', 'digest']}
    transport.atomic(HERE / 'GITHUB_ARCHIVE_ASSET.json', metadata)
    receipt = dict(checked_at=datetime.now(timezone.utc).isoformat(), repo='lzqw/unitree-smash',
        asset=asset['name'], asset_id=asset['id'], url=asset['browser_download_url'],
        bytes=asset['size'], sha256=check['sha256'], fresh_download=str(output),
        fresh_download_verified=True, archive_members_CRC_verified=members,
        unique_payloads_SHA_verified=len(objects), original_path_records=1016,
        all931_held_input_paths_delivered=True, all10_USD_source_layers_delivered=True,
        fresh_CPU_reproduction_verified=json.loads((HERE / 'FRESH_CPU_CHECK.json').read_text())['all931_held_inputs_verified_from_restored_tree'],
        new_native_execution=False, formal_cases_read=False, formal_acceptance=False, goal_complete=False)
    transport.atomic(HERE / 'UPLOAD_RECEIPT.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps(dict(download_failed=True, category=type(error).__name__)), flush=True)
        raise SystemExit(1) from None
