"""Fresh-download the actual residual15 video and full archive; verify all payloads."""
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
VIDEO_REPORT = ROOT.parent / 'unitree_smash_video_framing/reports/G07_full_racket_v4/origin0_residual15_cold_video_20261027'


def main():
    assert not (HERE / 'UPLOAD_RECEIPT.json').exists()
    spec = importlib.util.spec_from_file_location('bounded_verified_transport',
        HERE.parent / 'contact_aim7_completed_delivery_20261027/download_verified.py')
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    checks = [json.loads((VIDEO_REPORT / 'DECODE_AND_NATIVE_INPUT_CHECK.json').read_text()),
              json.loads((HERE / 'ARCHIVE_CHECK.json').read_text())]
    response = subprocess.run(['gh', 'api', 'repos/lzqw/unitree-smash/releases/422477709'],
                              text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert response.returncode == 0
    assets = json.loads(response.stdout)['assets']
    receipts = []
    for number, check in enumerate(checks):
        name = Path(check.get('video') or check['archive']).name
        matches = [a for a in assets if a['name'] == name]
        assert len(matches) == 1
        asset = matches[0]
        assert asset['state'] == 'uploaded' and asset['size'] == check['bytes']
        assert asset['digest'] == 'sha256:' + check['sha256']
        transport.ASSET_ID = asset['id']
        location = transport.signed_url()
        size = check['bytes']
        directory = Path(tempfile.mkdtemp(prefix='g07_residual15_completed_fresh_'))
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
                            data = stream.read(min(1024 * 1024, end - offset + 1))
                            assert data and offset + len(data) <= end + 1
                            assert os.pwrite(descriptor, data, offset) == len(data)
                            offset += len(data)
                        assert not stream.read(1)
                    return end - start + 1
                except Exception as error:
                    if attempt == 3:
                        raise RuntimeError(f'Artifact range failed: {type(error).__name__}') from None
                    time.sleep(attempt + 1)

        downloaded = 0
        try:
            with ThreadPoolExecutor(max_workers=4) as pool:
                for future in as_completed([pool.submit(part, start) for start in range(0, size, chunk)]):
                    downloaded += future.result()
                    print(json.dumps(dict(asset=name, downloaded_bytes=downloaded, total_bytes=size)), flush=True)
        finally:
            os.close(descriptor)
        assert downloaded == size and partial.stat().st_size == size
        assert hashlib.sha256(partial.read_bytes()).hexdigest() == check['sha256']
        payload_count = 0
        if number == 1:
            with zipfile.ZipFile(partial) as archive:
                assert archive.testzip() is None
                manifest = json.loads(archive.read('MANIFEST.json'))
                payloads = manifest['payloads']
                assert len(archive.namelist()) == len(set(archive.namelist())) == len(payloads) + 1
                assert set(archive.namelist()) == set(payloads) | {'MANIFEST.json'}
                for member, record in payloads.items():
                    content = archive.read(member)
                    assert len(content) == record['bytes'] and hashlib.sha256(content).hexdigest() == record['sha256'], member
                assert manifest['original128_cold_summary']['complete'] == 0
                assert manifest['original128_cold_summary']['airborne_contacts'] == 0
                assert manifest['actual_training_counts']['completed_episodes'] == 721
                assert manifest['actual_training_counts']['air'] == 7
                assert all(f'training15/ppo/model_{i}.pt' in payloads for i in range(15))
                assert manifest['current_policy_file'] in payloads and manifest['canonical_checkpoint_file'] in payloads
                assert 'cold128/episodes.json' in payloads and 'cold128/all_first_episodes.npz' in payloads
                assert not manifest['formal_acceptance']
                payload_count = len(payloads)
        final = directory / name
        partial.rename(final)
        metadata = {k: asset[k] for k in ['id', 'name', 'browser_download_url', 'size', 'state', 'digest']}
        transport.atomic(HERE / ('GITHUB_VIDEO_ASSET.json' if number == 0 else 'GITHUB_ARCHIVE_ASSET.json'), metadata)
        receipts.append(dict(asset=name, id=asset['id'], url=asset['browser_download_url'],
            bytes=size, sha256=check['sha256'], fresh_download=str(final), fresh_download_verified=True,
            archive_members_CRC_verified=payload_count + 1 if payload_count else None,
            archive_payload_SHA_verified=payload_count if payload_count else None))
    receipt = dict(checked_at=datetime.now(timezone.utc).isoformat(), repo='lzqw/unitree-smash',
        tag='g07-full-racket-v4-progress-20261024', assets=receipts, upload_succeeded=True,
        fresh_downloads_verified=True, actual15_training_and_original128_cold_delivered=True,
        all_original_cases_retained=True, new_native_execution=False,
        native_replay_in_fresh_extraction=False, formal_cases_read=False, formal_acceptance=False, goal_complete=False)
    transport.atomic(HERE / 'UPLOAD_RECEIPT.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps(dict(download_failed=True, category=type(error).__name__)), flush=True)
        raise SystemExit(1) from None
