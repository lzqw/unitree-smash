"""Fresh-download the published body8/cold128 archive and check every payload."""
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
TAG = 'g07-full-racket-v4-progress-20261024'


def main():
    assert not (HERE / 'UPLOAD_RECEIPT.json').exists()
    helper_path = HERE.parent / 'contact_aim7_completed_delivery_20261027/download_verified.py'
    spec = importlib.util.spec_from_file_location('bounded_artifact_transport', helper_path)
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    check = json.loads((HERE / 'ARCHIVE_CHECK.json').read_text())
    checked = json.loads((HERE / 'ACTUAL_COMPLETED_DATA_CHECK.json').read_text())
    assert checked['actual_search_rows'] == 1024 and checked['actual_cold_rows'] == 128
    assert checked['CPU_vs_actual_GPU_max_action_error'] <= 5e-4
    name, size = Path(check['archive']).name, check['bytes']
    response = subprocess.run(['gh', 'api', 'repos/lzqw/unitree-smash/releases/422477709'],
                              text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert response.returncode == 0
    assets = [a for a in json.loads(response.stdout)['assets'] if a['name'] == name]
    assert len(assets) == 1
    asset = assets[0]
    assert asset['state'] == 'uploaded' and asset['size'] == size
    assert asset['digest'] == 'sha256:' + check['sha256']
    transport.ASSET_ID = asset['id']
    location = transport.signed_url()
    directory = Path(tempfile.mkdtemp(prefix='g07_gain_body8_completed_fresh_'))
    partial = directory / (name + '.partial')
    descriptor = os.open(partial, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    os.ftruncate(descriptor, size)
    chunk = 32 * 1024 * 1024

    def part(start):
        end = min(size - 1, start + chunk - 1)
        for attempt in range(4):
            try:
                request = urllib.request.Request(location, headers={
                    'Range': f'bytes={start}-{end}', 'User-Agent': 'unitree-smash-artifact-verification'})
                with urllib.request.urlopen(request, timeout=25) as stream:
                    assert stream.status == 206
                    assert stream.headers['Content-Range'] == f'bytes {start}-{end}/{size}'
                    offset = start
                    while offset <= end:
                        block = stream.read(min(1024 * 1024, end - offset + 1))
                        assert block and offset + len(block) <= end + 1
                        assert os.pwrite(descriptor, block, offset) == len(block)
                        offset += len(block)
                    assert not stream.read(1)
                return end - start + 1
            except Exception as error:
                if attempt == 3:
                    raise RuntimeError(f'Artifact range failed: {type(error).__name__}') from None
                time.sleep(attempt + 1)

    downloaded = 0
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            pending = [pool.submit(part, start) for start in range(0, size, chunk)]
            for future in as_completed(pending):
                downloaded += future.result()
                print(json.dumps(dict(downloaded_bytes=downloaded, expected_bytes=size)), flush=True)
    finally:
        os.close(descriptor)
    assert downloaded == size and partial.stat().st_size == size
    assert hashlib.sha256(partial.read_bytes()).hexdigest() == check['sha256']
    with zipfile.ZipFile(partial) as archive:
        assert archive.testzip() is None
        manifest = json.loads(archive.read('MANIFEST.json'))
        payloads = manifest['payloads']
        assert len(archive.namelist()) == len(set(archive.namelist())) == len(payloads) + 1
        assert set(archive.namelist()) == set(payloads) | {'MANIFEST.json'}
        for member, record in payloads.items():
            data = archive.read(member)
            assert len(data) == record['bytes'] and hashlib.sha256(data).hexdigest() == record['sha256'], member
        assert manifest['original128_cold_summary'] == checked['original128_cold_summary']
        assert manifest['best_policy_sha256'] == checked['source_policy_sha256']
        assert all(f'search8/episodes_{i:03d}.json' in payloads for i in range(8))
        assert all(f'search8/all_first_episodes_{i:03d}.npz' in payloads for i in range(8))
        assert 'cold128/episodes.json' in payloads and 'cold128/all_first_episodes.npz' in payloads
        assert not manifest['formal_acceptance']
    final = directory / name
    partial.rename(final)
    metadata = {k: asset[k] for k in ['id', 'name', 'browser_download_url', 'size', 'state', 'digest']}
    transport.atomic(HERE / 'GITHUB_ARCHIVE_ASSET.json', metadata)
    receipt = dict(checked_at=datetime.now(timezone.utc).isoformat(), repo='lzqw/unitree-smash',
        release=TAG, asset=name, GitHub_asset_id=asset['id'], url=asset['browser_download_url'],
        bytes=size, sha256=check['sha256'], fresh_download=str(final), fresh_download_verified=True,
        archive_members_CRC_verified=len(payloads) + 1, archive_payload_SHA_verified=len(payloads),
        actual1024_search_plus128_cold_delivered=True, all_original_cases_retained=True,
        helper_source_sha256=hashlib.sha256(helper_path.read_bytes()).hexdigest(),
        native_replay_in_fresh_extraction=False, new_native_execution=False,
        formal_cases_read=False, formal_acceptance=False, goal_complete=False)
    transport.atomic(HERE / 'UPLOAD_RECEIPT.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps(dict(download_failed=True, category=type(error).__name__)), flush=True)
        raise SystemExit(1) from None
