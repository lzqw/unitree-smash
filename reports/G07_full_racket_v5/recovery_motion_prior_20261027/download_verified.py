"""Download the sealed small proposal packet; do not confer native performance."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent
MAIN = HERE.parents[2]


def main():
    expected = json.loads((HERE / 'ARCHIVE_CHECK.json').read_text())
    asset = json.loads((HERE / 'GITHUB_ASSET.json').read_text())
    assert asset['name'] == Path(expected['archive']).name and asset['size'] == expected['bytes']
    assert asset['state'] == 'uploaded' and asset['digest'] == 'sha256:' + expected['sha256']
    path = MAIN / 'reports/G07_full_racket_v4/contact_aim7_completed_delivery_20261027/download_verified.py'
    spec = importlib.util.spec_from_file_location('verified_release_transport', path)
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    transport.ASSET_ID = asset['id']
    for attempt in range(5):
        try:
            location = transport.signed_url()
            request = urllib.request.Request(location, headers={'User-Agent': 'unitree-smash-proposal-verification'})
            with urllib.request.urlopen(request, timeout=25) as response:
                assert response.status == 200
                data = response.read(expected['bytes'] + 1)
            assert len(data) == expected['bytes'] and hashlib.sha256(data).hexdigest() == expected['sha256']
            break
        except Exception as error:
            print(json.dumps(dict(attempt=attempt + 1, category=type(error).__name__)), flush=True)
            if attempt == 4:
                raise RuntimeError('Proposal artifact download failed after five bounded attempts') from None
            time.sleep(attempt + 1)
    directory = Path(tempfile.mkdtemp(prefix='g07_recovery_prior_fresh_download_'))
    archive_path = directory / asset['name']
    archive_path.write_bytes(data)
    extracted = directory / 'extracted'
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        manifest = json.loads(archive.read('MANIFEST.json'))
        payload = manifest['payloads']
        assert len(archive.namelist()) == len(set(archive.namelist())) == expected['members'] == len(payload) + 1
        assert set(archive.namelist()) == set(payload) | {'MANIFEST.json'}
        for name, record in payload.items():
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
            content = archive.read(name)
            assert len(content) == record['bytes'] and hashlib.sha256(content).hexdigest() == record['sha256']
        assert manifest['proposed_family_native_execution'] is False and manifest['seed_complete_smash'] is False
        archive.extractall(extracted)
    receipt = dict(checked_at=datetime.now(timezone.utc).isoformat(), repo='lzqw/unitree-smash',
        release='g07-full-racket-v4-progress-20261024', asset=asset['name'], GitHub_asset_id=asset['id'],
        url=asset['browser_download_url'], bytes=expected['bytes'], sha256=expected['sha256'],
        fresh_download=str(archive_path), fresh_extracted=str(extracted),
        fresh_downloaded_sha256=expected['sha256'], upload_succeeded=True, download_verified=True,
        members_CRC_verified=len(payload) + 1, payload_SHA_verified=len(payload),
        seed_actual_native_recovery=True, seed_complete_smash=False,
        proposed_family_native_execution=False, source_performance_inherited=False,
        required_runtime_companion=manifest['required_runtime_companion'],
        program_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        native_reexecution=False, formal_acceptance=False, goal_complete=False)
    assert not (HERE / 'UPLOAD_RECEIPT.json').exists()
    transport.atomic(HERE / 'UPLOAD_RECEIPT.json', receipt)
    print(json.dumps({key: receipt[key] for key in ['bytes', 'sha256', 'members_CRC_verified',
        'payload_SHA_verified', 'fresh_extracted', 'proposed_family_native_execution']}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps(dict(download_failed=True, category=type(error).__name__)), flush=True)
        raise SystemExit(1) from None
